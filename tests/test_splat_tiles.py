"""Splat tiles: partition by what the cameras see, the per-tile clean, a tiled run with Stop and Resume.

The trainer and ns-export are the stand-ins of test_splat_pause (no GPU)."""

import json
import os

import numpy as np
import plyfile
import pytest

from modelforge import jobs, splat, splat_plan, splat_tiles
from tests.test_splat_pause import fake_ns  # noqa: F401  (fixture)
from tests.test_splat_to_enu import make_splat


def write_scene(project, n_side=8, extent=100.0, seed=0):
    """Cameras on an n_side² grid at 50 m looking straight down over a ground of sparse points."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, (x, y) in enumerate((x, y) for x in np.linspace(-extent, extent, n_side)
                               for y in np.linspace(-extent, extent, n_side)):
        T = np.eye(4)  # OpenGL camera looking along -z: down
        T[:3, 3] = (x, y, 50.0)
        frames.append({"file_path": f"images/f{i:03d}.jpg", "transform_matrix": T.tolist()})
    json.dump({"fl_x": 500.0, "fl_y": 500.0, "cx": 500.0, "cy": 375.0, "w": 1000, "h": 750,
               "camera_model": "OPENCV", "frames": frames, "ply_file_path": "sparse_pc.ply"},
              open(os.path.join(project, "transforms.json"), "w"))
    xyz = np.c_[rng.uniform(-extent - 60, extent + 60, (4000, 2)), rng.normal(0, 0.5, 4000)].astype("f4")
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"], v["z"] = xyz.T
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(os.path.join(project, "sparse_pc.ply"))
    os.makedirs(os.path.join(project, "images"), exist_ok=True)
    return len(frames)


def test_partition(tmp_path):
    n = write_scene(str(tmp_path))
    tiles, box = splat_tiles.partition(str(tmp_path), photos=16)
    assert len(tiles) == 4 and all(t["core"] == 16 for t in tiles)
    covered = sorted({i for t in tiles for i in t["frames"]})
    assert covered == list(range(n))  # every photo trains somewhere
    assert all(len(t["frames"]) > t["core"] for t in tiles)  # plus neighbours seeing the cell
    centre = np.array([t["cell"] for t in tiles])
    assert np.isinf(centre).sum() == 8  # 2 x 2 cells: the outer sides open
    assert box[0] < -100 and box[2] > 100


def test_clean():
    v, _ = make_splat(np.random.default_rng(0), n=6, degree=0)
    v["x"] = [5, 15, 5, 5, -50, 5]
    v["y"] = 5
    for i in range(3):
        v[f"scale_{i}"] = np.log(0.1)
    v["scale_2"][2] = np.log(20.0)  # 3 sigma = 60 m > margin 10 m: a background blob
    keep = splat_tiles.clean(v, (0, 0, 10, 10), 10.0, (-40, -40, 40, 40))
    assert keep.tolist() == [True, False, False, True, False, True]  # out of cell, blob, out of the box


def test_plan_switches_to_tiles():
    s = {"images": 2000, "train_images": 1800, "source": "x", "width": 3840, "height": 2160, "initial_points": 1}
    assert splat_plan.suggest(s, vram=24)["tiles"]
    assert not splat_plan.suggest(s, vram=24, tiles="off")["tiles"]
    assert not splat_plan.suggest(dict(s, train_images=400), vram=24)["tiles"]


def test_tiled_run_stop_resume(fake_ns, tmp_path):  # noqa: F811
    project, _ = fake_ns
    write_scene(project)
    v, _ = make_splat(np.random.default_rng(1), n=500)  # what every fake training "exports" (identity frame)
    v["x"], v["y"] = np.random.default_rng(2).uniform(-250, 250, (2, 500))
    for i in range(3):
        v[f"scale_{i}"] = np.log(0.2)
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(str(tmp_path / "src.ply"))

    pct, job = [], jobs.Token()

    def progress(step, p):
        pct.append(p)
        if p > 35:  # inside tile 2 of 4
            job.cancel()
    args = dict(iterations=30, downscale=1, camera_opt="off")
    with pytest.raises(splat.StoppedPartial, match="tile 2 / 4"):
        splat_tiles.run(project, "splatfacto", args, on_line=lambda m: None, on_progress=progress, job=job,
                        cache="disk", photos=16)
    state = json.load(open(os.path.join(project, splat_tiles.STATE)))
    assert [t.get("status") for t in state["tiles"]] == ["done", "partial", None, None]
    info = splat.resume_info(project)
    assert info and info["tiles"] and info["tile"] == 1 and info["of"] == 4
    partial = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    assert 0 < len(partial) < 500  # tiles 1 + 2 merged

    stats = splat.resume(project, on_line=lambda m: None, on_tile=lambda k, n, it: None)
    merged = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    box = state["box"]
    inside = (v["x"] >= box[0]) & (v["x"] < box[2]) & (v["y"] >= box[1]) & (v["y"] < box[3])
    assert len(merged) == stats["num_gaussians"] == int(inside.sum())  # each Gaussian in exactly one cell
    assert stats["tiles"] == 4 and all(t["status"] == "done" for t in stats["per_tile"])
    assert splat.resume_info(project) is None
    t = json.load(open(os.path.join(splat_tiles.tile_dir(project, 0), "transforms.json")))
    assert len(t["frames"]) == len(state["tiles"][0]["frames"])
    assert os.path.islink(os.path.join(splat_tiles.tile_dir(project, 0), "images"))


def test_live_preview_before_splat_folder(tmp_path):
    """The first tile's preview comes before any merge, so splat/ may not exist yet."""
    v, _ = make_splat(np.random.default_rng(3), n=10, degree=0)
    path, ident = splat_tiles._write_live(str(tmp_path), [splat_tiles._band0(v)], 500)
    d = np.load(path)
    assert int(d["step"]) == 500 and d["means"].shape == (10, 3) and os.path.exists(ident)

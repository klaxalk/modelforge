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
    assert len(tiles) == 4 and sum(t["core"] for t in tiles) == n
    assert all(8 <= t["core"] <= 24 for t in tiles)  # cuts move to cheap gaps within the size slack
    covered = sorted({i for t in tiles for i in t["frames"]})
    assert covered == list(range(n))  # every photo trains somewhere
    assert all(len(t["frames"]) > t["core"] for t in tiles)  # plus neighbours seeing the cell
    centre = np.array([t["cell"] for t in tiles])
    assert np.isinf(centre).sum() == 8  # 2 x 2 cells: the outer sides open
    assert box[0] < -100 and box[2] > 100


def write_ground_mesh(project, half=300.0):
    """A flat ground quad at z = 0 as the SfM mesh."""
    v = np.array([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    f = np.array([([0, 1, 2],), ([0, 2, 3],)], dtype=[("vertex_indices", "i4", (3,))])
    os.makedirs(os.path.join(project, "sfmmesh"), exist_ok=True)
    path = os.path.join(project, "sfmmesh", "mesh.ply")
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex"), plyfile.PlyElement.describe(f, "face")]).write(path)
    return path


@pytest.mark.skipif(splat.available() is not None, reason="needs nerfstudio's venv (Open3D)")
def test_photos_join_the_tiles_they_show(tmp_path):
    """The SfM mesh raycast gives each photo's pixel share per tile; every photo showing a tile in >= MIN_PIXELS
    of its image trains it."""
    from modelforge import tile_regions
    project = str(tmp_path)
    n = write_scene(project)
    mesh = write_ground_mesh(project)
    # two labels split at x = 0; the first camera (x = -100) sees x in [-150, -50]: all label 0; one at x = -100 / 7
    # sees [-64.3, 35.7]: 35.7 % label 1
    labels = tile_regions.Regions((-400.0, -400.0), 1.0, np.repeat([[0] * 400 + [1] * 400], 800, 0))
    share = splat_tiles.mesh_shares(project, mesh, labels, on_line=lambda m: None)
    assert share.shape == (n, 2)
    assert np.allclose(share[0], [1, 0], atol=0.01)
    xs = np.array([json.load(open(os.path.join(project, "transforms.json")))["frames"][i]["transform_matrix"][0][3]
                   for i in range(n)])
    i = int(np.argmin(np.abs(xs + 100 / 7)))
    assert abs(share[i, 1] - 35.7 / 100) < 0.01
    old, _ = splat_tiles.partition(project, photos=16)
    new, _ = splat_tiles.partition(project, photos=16, mesh=mesh, on_line=lambda m: None)
    cells = splat_tiles._cell_labels([(t["cell"], None) for t in new], np.array([[-200.0, -200], [200, 200]]))
    pixels = splat_tiles.mesh_shares(project, mesh, cells, on_line=lambda m: None)
    for k, (a, b) in enumerate(zip(old, new)):
        assert set(a["frames"]) <= set(b["frames"])
        assert set(np.flatnonzero(pixels[:, k] >= splat_tiles.MIN_PIXELS)) <= set(b["frames"])
        assert b["by_pixels"] == len(b["frames"]) - len(a["frames"])
    assert sum(t["by_pixels"] for t in new) > 0


def test_cut_avoids_an_orbited_object():
    """Photos spread evenly along x, one object at the balanced cut seen by many of them: the cut moves off it,
    staying within the size slack."""
    rng = np.random.default_rng(0)
    looks = np.c_[np.linspace(0, 100, 400), rng.uniform(0, 100, 400)]
    ground = np.c_[rng.uniform(0, 100, (5000, 2))]
    obj = np.c_[rng.normal(50, 1.5, 500), rng.normal(50, 1.5, 500)]
    pts = np.r_[ground, obj]
    views = np.r_[np.full(len(ground), 5.0), np.full(len(obj), 80.0)]
    m = splat_tiles._cut(looks, 0, 0.5, pts, views)
    assert abs(m - 50) > 5  # off the object (sigma 1.5 m)
    below = (looks[:, 0] < m).mean()
    assert 0.5 - splat_tiles.SIZE_SLACK * 0.5 - 0.01 <= below <= 0.5 + splat_tiles.SIZE_SLACK * 0.5 + 0.01
    assert abs(splat_tiles._cut(looks, 0, 0.5, ground, np.full(len(ground), 5.0)) - 50) < 15  # no object: near


def test_clean():
    v, _ = make_splat(np.random.default_rng(0), n=6, degree=0)
    v["x"] = [5, 15, 5, 5, -50, 5]
    v["y"] = 5
    for i in range(3):
        v[f"scale_{i}"] = np.log(0.1)
    v["scale_2"][2] = np.log(20.0)  # 3 sigma = 60 m > margin 10 m: a background blob
    keep = splat_tiles.clean(v, (0, 0, 10, 10), 10.0, (-40, -40, 40, 40))
    assert keep.tolist() == [True, False, False, True, False, True]  # out of the cell; a blob over the neighbours

    # an outer tile (open to the west): background far out stays, also as a large blob; only the shared sides cut
    v["x"] = [-150, -150, 5, -5, -1e4, -150]
    v["y"] = [5, 5, 5, 5, 5, 95]
    v["scale_0"][1] = np.log(30.0)  # 3 sigma = 90 m, 150 m away from the shared east side: kept
    v["scale_0"][2] = np.log(30.0)  # the same blob 5 m from the shared side: it would cover the neighbour
    v["scale_0"][5] = np.log(30.0)  # far west but 5 m from the shared north side
    keep = splat_tiles.clean(v, (-np.inf, -np.inf, 10, 100), 10.0, (-40, -40, 40, 40))
    assert keep.tolist() == [True, True, False, True, False, False]  # ..., -1e4: beyond the far limit


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
                        cache="disk", photos=16, round_steps=0)
    state = json.load(open(os.path.join(project, splat_tiles.STATE)))
    assert [t.get("status") for t in state["tiles"]] == ["done", "partial", "pending", "pending"]
    info = splat.resume_info(project)
    assert info and info["tiles"] and info["tile"] == 1 and info["of"] == 4
    partial = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    assert 0 < len(partial) < 500  # tiles 1 + 2 merged

    stats = splat.resume(project, on_line=lambda m: None, on_tile=lambda *a: None)
    merged = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    assert len(merged) == stats["num_gaussians"] == len(v)  # each Gaussian in exactly one cell (outer ones open)
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


def test_rounds(fake_ns, tmp_path):  # noqa: F811
    """round_steps: every tile advances that many steps per round from its own checkpoint, merged after each round;
    Stop in a later round and Resume continue from there."""
    project, bin_dir = fake_ns
    write_scene(project)
    tiles, lines, pct = [], [], []
    args = dict(iterations=50, downscale=1, camera_opt="off")
    job = jobs.Token()

    def progress(step, p):
        pct.append(p)
        if len(tiles) == 6 and step >= 25 and not job.cancelled:  # round 2, tile 2: stop it while it trains
            job.cancel()
    with pytest.raises(splat.StoppedPartial, match="tile 2 / 4"):
        splat_tiles.run(project, "splatfacto", args, on_line=lines.append, cache="disk", photos=16, round_steps=20,
                        on_tile=lambda k, n, target, label: tiles.append((k, target, label)), job=job,
                        on_progress=progress)
    assert tiles[:4] == [(k, 20, "round 1 / 3") for k in range(4)]
    assert any("round 1 / 3: all 4 tiles at step 20" in m for m in lines)
    state = json.load(open(os.path.join(project, splat_tiles.STATE)))
    assert [t["status"] for t in state["tiles"]] == ["round", "partial", "round", "round"]
    assert state["tiles"][0]["step"] == 40 and 25 <= state["tiles"][1]["step"] < 40 and state["tiles"][2]["step"] == 20

    tiles[:] = []
    stats = splat.resume(project, on_line=lines.append, on_tile=lambda k, n, target, label: tiles.append((k, target, label)),
                         on_progress=lambda s, p: pct.append(p))
    assert tiles == [(1, 40, "round 2 / 3"), (2, 40, "round 2 / 3"), (3, 40, "round 2 / 3")] + \
        [(k, 50, "round 3 / 3") for k in range(4)]
    assert stats["tiles"] == 4 and all(t["status"] == "done" for t in stats["per_tile"])
    args_last = json.load(open(os.path.join(bin_dir, "last_args.json")))
    assert "--load-dir" in args_last  # later rounds continue the tile's checkpoint
    assert splat.resume_info(project) is None


def _spread_source(tmp_path):
    """What every fake training exports: 500 Gaussians over the scene (identity frame)."""
    v, _ = make_splat(np.random.default_rng(1), n=500)
    v["x"], v["y"] = np.random.default_rng(2).uniform(-250, 250, (2, 500))
    for i in range(3):
        v[f"scale_{i}"] = np.log(0.2)
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(str(tmp_path / "src.ply"))
    return v


def test_setup_and_one_tile(fake_ns, tmp_path):  # noqa: F811
    """Set up tiles without training; train all; then one tile alone: the others stay, all are merged again."""
    project, _ = fake_ns
    write_scene(project)
    v = _spread_source(tmp_path)
    args = dict(iterations=30, downscale=1, camera_opt="off")
    with pytest.raises(RuntimeError, match="set up the tiles first"):
        splat_tiles.run(project, "splatfacto", args, on_line=lambda m: None, cache="disk", photos=16, only=1)
    state = splat_tiles.setup(project, 16, on_line=lambda m: None)
    assert [t["status"] for t in state["tiles"]] == ["pending"] * 4
    assert all(os.path.exists(os.path.join(splat_tiles.tile_dir(project, k), "transforms.json")) for k in range(4))
    assert splat_tiles.overlay(project, 99)[1] is False  # the View shows the set-up cells

    trained = []
    on_tile = lambda k, n, target, label: trained.append(k)  # noqa: E731
    splat_tiles.run(project, "splatfacto", args, on_line=lambda m: None, cache="disk", photos=16, round_steps=0,
                    on_tile=on_tile)
    assert trained == [0, 1, 2, 3]
    assert splat_tiles.merged_tiles(project) == [0, 1, 2, 3]
    times = [os.path.getmtime(os.path.join(splat_tiles.tile_dir(project, k), "splat", "clean_enu.ply"))
             for k in range(4)]

    trained[:] = []
    stats = splat_tiles.run(project, "splatfacto", dict(args, iterations=40), on_line=lambda m: None, cache="disk",
                            photos=16, on_tile=on_tile, only=2)
    assert trained == [2]  # in one go, no rounds
    state = splat_tiles.load(project)
    assert [t["iterations"] for t in state["tiles"]] == [30, 30, 40, 30]  # its own new settings
    after = [os.path.getmtime(os.path.join(splat_tiles.tile_dir(project, k), "splat", "clean_enu.ply"))
             for k in range(4)]
    assert after[:2] == times[:2] and after[3] == times[3] and after[2] > times[2]
    assert stats["num_gaussians"] == len(v) and splat_tiles.merged_tiles(project) == [0, 1, 2, 3]

    # the merged splat changes elsewhere: no longer the tiles' merge
    os.utime(os.path.join(project, "splat", "splat_enu.ply"))
    assert splat_tiles.merged_tiles(project) is None


def test_semantic_per_tile(fake_ns, tmp_path, monkeypatch):  # noqa: F811
    """Semantic features of a tiled splat: one PCA, per tile from its photos (its cleaned part), merged in the
    splat's order; a retrained tile alone is recomputed."""
    from modelforge import semantic
    project, _ = fake_ns
    write_scene(project)
    _spread_source(tmp_path)
    args = dict(iterations=30, downscale=1, camera_opt="off")
    splat_tiles.run(project, "splatfacto", args, on_line=lambda m: None, cache="disk", photos=16, round_steps=0)
    calls = []

    def fake_work(job, arrays, out_dir, params, line):  # feature = the Gaussian's x: order checkable
        os.makedirs(out_dir, exist_ok=True)
        if params.get("pca_out"):
            calls.append("pca")
            np.savez(params["pca_out"], pca_mean=np.zeros(8, "f4"), pca_basis=np.eye(8, 4, dtype="f4"),
                     explained=np.float32(0.9))
            return
        calls.append(out_dir)
        keep = arrays["keep"]
        f = np.zeros((len(keep), 4), "f2")
        f[:, 0] = arrays["means"][keep, 0]
        np.savez(os.path.join(out_dir, "features.npz"), feats=f, weight=np.ones(len(keep), "f4"))
        json.dump({"photos": len(arrays["paths"]), "width": 64, "height": 48},
                  open(os.path.join(out_dir, "stats.json"), "w"))
    monkeypatch.setattr(semantic, "_work", fake_work)
    monkeypatch.setattr(semantic, "available", lambda: None)
    meta = semantic.run(project, on_line=lambda m: None)
    assert calls[0] == "pca" and len(calls) == 5 and meta["tiles"] == 4
    merged = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    d = np.load(os.path.join(project, semantic.DIR, "features.npz"))
    assert len(d["feats"]) == len(merged) and np.allclose(d["feats"][:, 0], merged["x"], atol=0.2)  # float16
    assert semantic.status(project) == "ready"

    splat_tiles.run(project, "splatfacto", args, on_line=lambda m: None, cache="disk", photos=16, only=1)
    calls[:] = []
    semantic.run(project, on_line=lambda m: None)
    assert calls == [splat_tiles.tile_dir(project, 1) + "/semantic"]  # the PCA and the other tiles reused


REAL = os.environ.get("MODELFORGE_TEST_PROJECT", "data/berlin_3_1_frames")  # any refined project


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined")),
                    reason="needs a refined project (MODELFORGE_TEST_PROJECT)")
def test_covisibility_layout(tmp_path):
    """Tile layout covisibility: photos split by the points they share, regions on the ground (a label map), each
    position in exactly one region, the tiles' datasets cropped to their region, and the clean keeps every small
    Gaussian of the scene box in exactly one tile."""
    p = str(tmp_path / "ws")
    os.makedirs(p)
    for f in ("images", "sparse", "sparse_pc.ply", "poses.json"):
        os.symlink(os.path.abspath(os.path.join(REAL, f)), os.path.join(p, f))
    t = json.load(open(os.path.join(REAL, "transforms.json")))
    json.dump(t, open(os.path.join(p, "transforms.json"), "w"))
    n = len(t["frames"])
    state = splat_tiles.setup(p, max(10, n // 3), layout="covisibility", on_line=lambda m: None)
    assert state["layout"] == "covisibility" and len(state["tiles"]) >= 2
    reg = splat_tiles.regions(p)
    assert sorted(set(np.unique(reg.labels))) == list(range(len(state["tiles"])))
    cores = [tl["core"] for tl in state["tiles"]]
    assert sum(cores) == n and min(cores) > 0
    for k, tl in enumerate(state["tiles"]):
        assert tl["region"] == k and set(range(n)) >= set(tl["frames"]) and tl["points"] > 0
        assert len(json.load(open(os.path.join(splat_tiles.tile_dir(p, k), "transforms.json")))["frames"]) == \
            len(tl["frames"])
    box = state["box"]
    rng = np.random.default_rng(0)
    v = np.zeros(5000, [(c, "f4") for c in ("x", "y", "z", "scale_0", "scale_1", "scale_2")])
    v["x"], v["y"] = rng.uniform(box[0], box[2], 5000), rng.uniform(box[1], box[3], 5000)
    v["scale_0"] = v["scale_1"] = v["scale_2"] = np.log(0.01)
    keeps = np.array([splat_tiles.clean_tile(p, v, tl, box) for tl in state["tiles"]])
    assert (keeps.sum(0) == 1).all()
    v["scale_0"] = np.log(1000.0)  # a huge Gaussian reaching over every seam: kept by none
    assert not np.array([splat_tiles.clean_tile(p, v, tl, box) for tl in state["tiles"]]).any()
    cells, planned = splat_tiles.overlay(p)
    assert not planned and all(c["outlines"] for c in cells)


def test_skip_tiles(fake_ns, tmp_path):  # noqa: F811
    """Tiles left out (tiles/skip.json): one before the run, one clicked away while it runs; neither trains and
    neither is merged. Taking one back and merging again brings its splat back."""
    project, _ = fake_ns
    write_scene(project)
    v = _spread_source(tmp_path)
    started = []

    def on_tile(k, n, target, label):
        started.append(k)
        if k == 1:
            splat_tiles.set_skipped(project, 3)  # the View tab click during the run
    lines = []
    stats = splat_tiles.run(project, "splatfacto", dict(iterations=20, downscale=1, camera_opt="off"),
                            on_line=lines.append, cache="disk", photos=16, round_steps=0, on_tile=on_tile, skip=[0])
    assert started == [1, 2] and any("tiles 1 left out" in m for m in lines)
    state = json.load(open(os.path.join(project, splat_tiles.STATE)))
    assert [t["status"] for t in state["tiles"]] == ["pending", "done", "done", "pending"]
    assert state["merged"]["tiles"] == [1, 2] and splat_tiles.skipped(project) == {0, 3}
    merged = plyfile.PlyData.read(os.path.join(project, "splat", "splat_enu.ply"))["vertex"].data
    assert 0 < len(merged) == stats["num_gaussians"] < len(v)
    cells, planned = splat_tiles.overlay(project)
    assert [c["skipped"] for c in cells] == [True, False, False, True]
    splat_tiles.set_skipped(project, 3, False)
    assert splat_tiles.skipped(project) == {0}


def test_unrecorded_checkpoint_is_adopted(tmp_path):
    """A tile marked training at step 0 whose run saved a checkpoint and ended before recording it continues from
    that checkpoint on resume instead of starting over."""
    project = str(tmp_path)
    write_scene(project)
    state = splat_tiles.setup(project, 16, on_line=lambda m: None)
    tile = state["tiles"][1]
    tile.update(status="training", step=0, iterations=50, schedule={"iterations": 50})
    state.update(method="splatfacto", extra=[], cache="disk", lidar_depth=0.2, depth_cloud=None, depth_dense=False)
    run_dir = os.path.join(splat_tiles.tile_dir(project, 1), "nerfstudio", "tile_01", "splatfacto-lidar", "run",
                           "nerfstudio_models")
    os.makedirs(run_dir)
    for step in (20, 41):
        open(os.path.join(run_dir, f"step-{step:09d}.ckpt"), "w").write("x")
    assert splat_tiles.adopt_checkpoint(project, 1, tile, state, 1) == 41
    assert tile["step"] == 41 and tile["status"] == "round"
    info = splat.resume_info(splat_tiles.tile_dir(project, 1))
    assert info and info["load_dir"] == run_dir and info["step"] == 41 and info["cache"] == "disk"
    other = state["tiles"][2]
    other.update(status="training", step=0, iterations=50, schedule={})
    assert splat_tiles.adopt_checkpoint(project, 2, other, state, 1) is None and other["step"] == 0


def test_narrow_region_keeps_its_seeds():
    """A region narrower than 2 x BAND_M (a 30 m strip between two wide ones) still gets seeds for the seam
    watershed: before, it had no cell BAND_M from a seam and its neighbours flooded it (kladno_dira: 2 of 11 tiles
    without a region, so without points)."""
    from modelforge import tile_regions
    x, y = np.meshgrid(np.arange(0, 200, 0.5), np.arange(0, 60, 0.5))
    xyz = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    strip = np.digitize(xyz[:, 0], [85, 115])
    centers = np.array([[42.0, 30, 20], [100, 30, 20], [157, 30, 20]])
    obs_pt = np.arange(len(xyz))
    reg = tile_regions.regions_from_parts(xyz, strip, obs_pt, centers, np.arange(3), 3)
    area = [int((reg.labels == t).sum()) for t in range(3)]
    assert min(area) > 0.8 * 30 * 60, area
    assert (reg.label(np.array([[100.0, 30]])) == 1).all()

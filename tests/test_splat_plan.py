"""Splat settings from the flight (modelforge/splat_plan.py) and their ns-train flags."""

import json
import os

import numpy as np
import pytest

from modelforge import splat, splat_plan


def make_survey_project(path, nx=10, ny=10, step=20.0, height=50.0, w=1920, h=1080, fx=1000.0, ground=0.0):
    """Nadir cameras on a grid at z = `height` (transforms.json, OpenGL axes) over sparse points on the
    plane z = `ground` inside the grid's interior (sparse_pc.ply)."""
    import plyfile
    frames = []
    for i in range(nx):
        for j in range(ny):
            c2w = np.eye(4)  # the OpenGL camera looks along its -z = world down (nadir), image up = north
            c2w[:3, 3] = (i * step, j * step, height)
            frames.append({"file_path": f"images/{i}_{j}.jpg", "transform_matrix": c2w.tolist()})
    json.dump({"fl_x": fx, "fl_y": fx, "cx": w / 2, "cy": h / 2, "w": w, "h": h, "frames": frames},
              open(os.path.join(path, "transforms.json"), "w"))
    xy = np.random.default_rng(0).uniform(60, 120, (2000, 2))
    v = np.zeros(len(xy), [("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"], v["z"] = xy[:, 0], xy[:, 1], ground
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(os.path.join(path, "sparse_pc.ply"))
    return path


def test_survey_overlap_gsd(tmp_path):
    p = make_survey_project(str(tmp_path))
    s = splat_plan.survey(p)
    assert s["images"] == 100 and s["train_images"] == 90 and s["source"] == "refined cameras"
    assert s["initial_points"] == 2000
    assert 11 < s["overlap"] < 15  # footprint 96 x 54 m over a 20 m grid: ~4.8 x 2.7 cameras per point
    assert abs(s["gsd_m"] - 0.05) < 1e-6  # depth 50 m / 1000 px
    assert splat_plan.needed_gaussians(s) == int(100 * 1920 * 1080 / (s["overlap"] * 4))


def test_survey_assumes_no_ground_plane(tmp_path):
    """The scene 100 m below the cameras' start (the flight began in the air, the terrain drops): the
    GSD follows the sparse points, not a takeoff ground."""
    s = splat_plan.survey(make_survey_project(str(tmp_path), ground=-100.0))
    assert abs(s["gsd_m"] - 0.15) < 1e-6 and s["overlap"] > 30  # 3x the footprint per side


def test_survey_without_points(tmp_path):
    p = make_survey_project(str(tmp_path))
    os.remove(os.path.join(p, "sparse_pc.ply"))
    s = splat_plan.survey(p)
    assert s["gsd_m"] is None and splat_plan.needed_gaussians(s) is None
    plan = splat_plan.suggest(s, vram=24)
    assert plan["strategy"] == "default" and plan["downscale"] == 1
    assert any("no sparse points" in n for n in plan["notes"])


def test_suggest_small_and_large():
    small = {"images": 200, "train_images": 180, "source": "x", "width": 1920, "height": 1080,
             "initial_points": 50_000, "area_m2": 2_000.0, "overlap": 30.0, "gsd_m": 0.02}
    p = splat_plan.suggest(small, vram=24)
    assert p["iterations"] == 30_000 and p["downscale"] == 1 and p["strategy"] == "default"
    assert p["reset_alpha_every"] == 30 and p["stop_split_at"] == 15_000 and p["max_gaussians"] is None
    assert p["camera_opt"] == "SO3xR3"
    # splatfacto-big grows ~2x the Gaussians without a cap: it takes the capped mcmc at half the need
    mid = dict(small, images=350)  # ~6.0M needed, under the 7M a 24 GB GPU fits
    assert splat_plan.suggest(mid, vram=24)["strategy"] == "default"
    for m in ("splatfacto-big", "splatfacto-big-lidar"):
        p = splat_plan.suggest(mid, vram=24, method=m)
        assert p["strategy"] == "mcmc" and p["max_gaussians"] == splat_plan.needed_gaussians(mid)
        assert any("grows ~2×" in n for n in p["notes"])
    assert splat_plan.suggest(small, vram=24, method="splatfacto-big")["strategy"] == "default"  # 3.5M x 2 fits

    big = dict(small, images=8000, train_images=7200, width=3840, height=2160, area_m2=250_000.0, gsd_m=0.03)
    p = splat_plan.suggest(big, vram=24)
    assert p["iterations"] == 144_000 and p["strategy"] == "mcmc"  # 20 views per training image
    assert p["downscale"] == 1 and p["needed"] > p["cap"]  # full resolution; the budget is the GPU's
    assert p["max_gaussians"] == p["cap"] and p["stop_split_at"] == 120_000
    p = splat_plan.suggest(big, vram=24, downscale=8)  # fewer pixels: the need shrinks with the factor²
    assert p["needed"] == splat_plan.needed_gaussians(big, 8) == splat_plan.needed_gaussians(big) // 64
    # the default strategy: refinement needs step % reset period >= training images + 100
    p = splat_plan.suggest(big, vram=24, strategy="default")
    assert p["reset_alpha_every"] * 100 > 7200 + 100 + 1000 and p["max_gaussians"] is None
    assert any("opacity reset" in n for n in p["notes"])

    # explicit values win; the dependent ones follow
    p = splat_plan.suggest(big, vram=24, iterations=60_000, downscale="1", strategy="mcmc", max_gaussians=3e6,
                           camera_opt="off")
    assert (p["iterations"], p["downscale"], p["max_gaussians"], p["camera_opt"], p["stop_split_at"]) == \
        (60_000, 1, 3_000_000, "off", 50_000)


def test_schedule_flags():
    plan = splat_plan.suggest({"images": 8000, "train_images": 7200, "source": "x", "width": 3840,
                               "height": 2160, "initial_points": 1, "area_m2": 250_000.0, "overlap": 20.0,
                               "gsd_m": 0.03}, vram=24)
    cmd = splat.train_command("/p", iterations=plan["iterations"], downscale=plan["downscale"], schedule=plan)
    val = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    assert val("--max-num-iterations") == val("--optimizers.means.scheduler.max-steps") == "144000"
    assert val("--pipeline.model.strategy") == "mcmc"
    assert int(val("--pipeline.model.max-gs-num")) == plan["max_gaussians"]
    assert val("--pipeline.model.camera-optimizer.mode") == "SO3xR3"
    assert val("--steps-per-eval-all-images") == "0"
    assert cmd[-3:] == ["nerfstudio-data", "--downscale-factor", str(plan["downscale"])]
    assert "--pipeline.model.strategy" not in splat.train_command("/p")  # no plan: splatfacto's own
    with pytest.raises(ValueError):
        splat.schedule_args(strategy="greedy")

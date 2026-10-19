"""Splat settings from the flight (modelforge/splat_plan.py) and their ns-train flags."""

import json
import os

import numpy as np
import pytest

from modelforge import splat, splat_plan


def make_survey_project(path, nx=10, ny=10, step=20.0, height=50.0, w=1920, h=1080, fx=1000.0):
    """Nadir cameras on a grid at `height` above ground z = 0 (transforms.json, OpenGL axes)."""
    frames = []
    for i in range(nx):
        for j in range(ny):
            c2w = np.eye(4)
            # identity: the OpenGL camera looks along its -z = world down (nadir), image up = north
            c2w[:3, 3] = (i * step, j * step, height)
            frames.append({"file_path": f"images/{i}_{j}.jpg", "transform_matrix": c2w.tolist()})
    json.dump({"fl_x": fx, "fl_y": fx, "cx": w / 2, "cy": h / 2, "w": w, "h": h, "frames": frames},
              open(os.path.join(path, "transforms.json"), "w"))
    json.dump({"ground_z": 0.0, "images": []}, open(os.path.join(path, "poses.json"), "w"))
    return path


def test_survey_area_overlap_gsd(tmp_path):
    p = make_survey_project(str(tmp_path))
    s = splat_plan.survey(p)
    fw, fh = 50.0 * 1920 / 1000, 50.0 * 1080 / 1000  # footprint 96 x 54 m
    expect = (180 + fw) * (180 + fh)  # union of the grid's footprints
    assert s["images"] == 100 and s["train_images"] == 90 and s["source"] == "refined cameras"
    assert abs(s["area_m2"] - expect) / expect < 0.03
    assert abs(s["overlap"] - 100 * fw * fh / expect) < 0.5
    assert 0.050 <= s["gsd_m"] < 0.060  # 50 m / 1000 px, slant range: median over the footprint's sample rays


def test_suggest_small_and_large():
    small = {"images": 200, "train_images": 180, "source": "x", "width": 1920, "height": 1080,
             "initial_points": 50_000, "area_m2": 2_000.0, "overlap": 30.0, "gsd_m": 0.02}
    p = splat_plan.suggest(small, vram=24)
    assert p["iterations"] == 30_000 and p["downscale"] == 1 and p["strategy"] == "default"
    assert p["reset_alpha_every"] == 30 and p["stop_split_at"] == 15_000 and p["max_gaussians"] is None
    assert p["camera_opt"] == "SO3xR3"

    big = dict(small, images=8000, train_images=7200, width=3840, height=2160, area_m2=250_000.0, gsd_m=0.03)
    p = splat_plan.suggest(big, vram=24)
    assert p["iterations"] == 144_000 and p["strategy"] == "mcmc"  # 20 views per training image
    assert p["needed"][p["downscale"]] <= p["cap"] and p["downscale"] > 1  # the smallest factor that fits
    assert p["max_gaussians"] <= p["cap"] and p["stop_split_at"] == 120_000
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

"""Per-photo exposure from the djmd camera records (modelforge/exposure.py)."""
import json
import os
import shutil

import numpy as np
import pytest

from modelforge import exposure

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", "data/fullrun_test")


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="no test project")
def test_gains_from_the_metadata(tmp_path):
    p = str(tmp_path / "proj")
    os.makedirs(p)
    shutil.copy(os.path.join(REAL, "poses.json"), p)
    e = exposure.per_image(p)
    poses = json.load(open(os.path.join(p, "poses.json")))
    assert set(e) == {r["image"] for r in poses["images"]}
    g = np.array([x["gain"] for x in e.values()])
    assert abs(np.median(g) - 1) < 1e-3 and (g > 0).all()
    x = next(iter(e.values()))
    rel = x["shutter_s"] * x["iso"] / x["fnum"] ** 2
    ref = np.median([v["shutter_s"] * v["iso"] / v["fnum"] ** 2 for v in e.values()])
    assert abs(x["gain"] - rel / ref) < 1e-4
    assert os.path.exists(os.path.join(p, exposure.CACHE)) and exposure.per_image(p) == e  # cached


def test_no_poses(tmp_path):
    assert exposure.per_image(str(tmp_path)) == {}


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="no test project")
def test_gains_are_relative_to_each_cameras_median(tmp_path):
    """Exposures compare within one camera only (berlin_all: a Mavic 3 with an ND filter next to a NEO2 read 4.7 stops
    apart for equally bright photos): every camera's median gain is 1; photos outside the set in use follow their
    camera's reference."""
    p = str(tmp_path / "proj")
    os.makedirs(p)
    poses = json.load(open(os.path.join(REAL, "poses.json")))
    imgs = sorted(poses["images"], key=lambda r: r["time_s"])
    cam = {r["image"]: k % 2 for k, r in enumerate(imgs[1:])}  # the first photo kept out: "registered" below
    poses["images"] = [dict(r, camera=cam[r["image"]]) for r in imgs[1:]]
    json.dump(poses, open(os.path.join(p, "poses.json"), "w"))
    e = exposure.per_image(p)
    raw = {n: x["shutter_s"] * x["iso"] / x["fnum"] ** 2 for n, x in e.items()}
    for k in (0, 1):
        g = [x["gain"] for n, x in e.items() if cam[n] == k]
        assert abs(np.median(g) - 1) < 1e-3
        ref = np.median([raw[n] for n in e if cam[n] == k])
        assert all(abs(e[n]["gain"] - raw[n] / ref) < 1e-4 for n in e if cam[n] == k)
    first = dict(imgs[0], camera=1)
    got = exposure.gains_for(p, [first])
    full = exposure.per_image(REAL)  # the first photo's raw exposure from the unmodified project
    x = full[first["image"]]
    ref1 = np.median([raw[n] for n in e if cam[n] == 1])
    assert abs(got[first["image"]] - x["shutter_s"] * x["iso"] / x["fnum"] ** 2 / ref1) < 1e-4


def test_camera_stops_brighten_one_camera(tmp_path):
    """Splat tab → Camera exposure: a camera's photos brightened by N stops = factor 2^-N on their gains (also for
    photos of the extracted pool registered later); the training command passes the factors to the dataparser, with
    the metadata gains or without them."""
    from modelforge import splat
    p = str(tmp_path)
    cams = [{"key": "DJIMavic3 1920x1080"}, {"key": "DJI NEO2 1920x1080"}]
    json.dump({"cameras": cams, "images": [{"image": "m.jpg", "camera": 0}, {"image": "n.jpg", "camera": 1}]},
              open(os.path.join(p, "poses.json"), "w"))
    os.makedirs(os.path.join(p, "extracted"))
    json.dump({"cameras": cams, "images": [{"image": "n2.jpg", "camera": 1}, {"image": "m2.jpg"}]},
              open(os.path.join(p, "extracted", "poses.json"), "w"))
    assert exposure.cameras(p) == ["DJIMavic3 1920x1080", "DJI NEO2 1920x1080"]
    assert exposure.stops(p) == {} and exposure.factors(p, {}) == {}
    exposure.set_stops(p, "DJI NEO2 1920x1080", 0.75)
    assert exposure.stops(p) == {"DJI NEO2 1920x1080": 0.75}
    f = exposure.factors(p, exposure.stops(p))
    assert set(f) == {"n.jpg", "n2.jpg"} and abs(f["n.jpg"] - 2 ** -0.75) < 1e-12
    exposure.set_stops(p, "DJI NEO2 1920x1080", 0)
    assert exposure.stops(p) == {}

    reg = {"exposure": True, "exposure_stops": {"DJI NEO2 1920x1080": 0.75, "DJIMavic3 1920x1080": 0.0}}
    assert splat.camera_stops(reg) == {"DJI NEO2 1920x1080": 0.75} and splat.regularizing({"exposure_stops": reg["exposure_stops"]})
    cmd = splat.train_command("/p", lidar_depth=0.0, regularize=reg, exposure_factors="/p/splat/f.json")
    assert "--pipeline.model.use-exposure" in cmd and cmd[cmd.index("--exposure-factors") + 1] == "/p/splat/f.json"
    assert "--exposure-meta" not in cmd
    cmd = splat.train_command("/p", lidar_depth=0.0, regularize={"exposure_stops": reg["exposure_stops"]},
                              exposure_factors="/p/splat/f.json")
    assert "--pipeline.model.use-exposure" in cmd and cmd[cmd.index("--exposure-meta") + 1] == "False"
    assert not splat.regularizing({"exposure_stops": {"DJI NEO2 1920x1080": 0.0}})

"""Multi-camera DJI projects (docs/todo_multicamera.md): a camera per source, features per camera, intrinsics and
boresight per camera in refine."""
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from modelforge import pipeline
from modelforge.dji import ingest
from modelforge.sfm import priors as priors_mod
from modelforge.sfm import refine

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


def clip(zoom, model="DJI Mavic4 Pro L3A", size=(3840, 2160)):
    return SimpleNamespace(model=model, width=size[0], height=size[1], pts=np.arange(len(zoom)) / 30.0,
                           zoom=np.asarray(zoom, float))


def test_camera_key_and_zoom():
    c = clip([1.0] * 90)
    c.zoom_range = lambda s=0.0, e=None: ingest.Clip.zoom_range(c, s, e)
    assert ingest.Clip.camera(c)["key"] == "DJI Mavic4 Pro L3A 3840x2160"
    assert not ingest.zooms(c, ingest.Source("x.MP4"))
    z = clip([1.0] * 30 + list(np.linspace(1, 3, 30)) + [3.0] * 30)
    z.zoom_range = lambda s=0.0, e=None: ingest.Clip.zoom_range(z, s, e)
    assert ingest.zooms(z, ingest.Source("x.MP4")) == "x1 to x3"
    assert not ingest.zooms(z, ingest.Source("x.MP4", start=0.0, end=0.9))  # the range before the zoom
    assert ingest.Clip.camera(z, 2.1, None)["key"] == "DJI Mavic4 Pro L3A 3840x2160 zoom 3"


def test_calibration_subset_covers_every_camera():
    names = [f"a_{k:05d}.jpg" for k in range(1000)]
    pri = {n: SimpleNamespace(order=k, source=0, time=float(k)) for k, n in enumerate(names)}
    cams = {n: (2 if 600 <= k < 640 else 1) for k, n in enumerate(names)}  # a camera between the runs
    sub = refine.calibration_subset(pri, names, cameras=cams)
    assert {cams[n] for n in sub} == {1, 2} and sum(cams[n] == 2 for n in sub) == 40
    assert refine.calibration_subset(pri, names) == sub[:300]


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "images")), reason="needs a project with photos")
def test_two_cameras_get_their_own_intrinsics(tmp_path):
    """Every other photo resized to 2/3 as a second camera: refine gives f1 = f0 x 2/3 (fullrun_test: 1084.59 /
    723.10 px), both cameras in report.json."""
    from PIL import Image
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "images"))
    p = json.load(open(os.path.join(REAL, "poses.json")))
    p["images"] = p["images"][:40]
    for k, r in enumerate(p["images"]):
        src, dst = os.path.join(REAL, "images", r["image"]), os.path.join(ws, "images", r["image"])
        im = Image.open(src)
        if k % 2:
            im.resize((im.width * 2 // 3, im.height * 2 // 3), Image.LANCZOS).save(dst, quality=95)
        else:
            shutil.copyfile(src, dst)
        r["camera"] = k % 2
    json.dump(p, open(os.path.join(ws, "poses.json"), "w"))
    if os.path.exists(os.path.join(REAL, "ingest.json")):
        shutil.copy2(os.path.join(REAL, "ingest.json"), ws)
    rep = pipeline.Report(ws)
    pipeline.stage_features(ws, rep)
    assert rep.data["features"]["cameras"] == 2
    pri, meta = priors_mod.load(ws)
    pipeline.stage_pairs(ws, rep, pri, meta)
    pipeline.stage_match(ws, rep)
    pipeline.stage_refine(ws, rep, pri, ba_mode="global")
    f = rep.data["refine"]["final"]
    assert f["reg_images"] == 40 and f["point_error_px"]["median"] < 1.0
    (a, b) = sorted(f["cameras"].values(), key=lambda c: -c["size"][0])
    assert abs(b["params"][0] / a["params"][0] - 2 / 3) < 0.005, (a, b)
    # the export: intrinsics per frame; the readers take each frame's camera
    from modelforge import splat_plan
    from modelforge.export import frame_camera
    pipeline.stage_export(ws, rep, drop_isolated=False)
    t = json.load(open(os.path.join(ws, "transforms.json")))
    cams, index = frame_camera.cameras(t)
    assert t["mf_cameras"] == 2 and len(cams) == 2 and sorted(c["w"] for c in cams) == [1280, 1920]
    names = [r["image"] for r in p["images"]]
    for f, i in zip(t["frames"], index):
        assert cams[i]["w"] == (1280 if names.index(os.path.basename(f["file_path"])) % 2 else 1920)
    s = splat_plan.survey(ws)
    assert s["width"] == 1920 and s["overlap"] > 2 and s["gsd_m"] > 0
    assert frame_camera.largest(t)["w"] == 1920


def test_frame_camera():
    from modelforge.export import frame_camera
    t = {"fl_x": 100.0, "w": 200, "h": 100, "frames": [{"file_path": "a"}, {"file_path": "b", "fl_x": 50.0, "w": 100},
                                                        {"file_path": "c"}]}
    cams, index = frame_camera.cameras(t)
    assert index == [0, 1, 0] and cams[1] == {"fl_x": 50.0, "w": 100, "h": 100}
    assert frame_camera.largest(t)["w"] == 200

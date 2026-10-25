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
    # one common training resolution: the largest camera at 1 / N, the others at that long side, never upscaled
    t = {"camera_model": "OPENCV", "frames": [
        {"file_path": "images/a.jpg", "fl_x": 400.0, "fl_y": 400.0, "cx": 200.0, "cy": 100.0, "w": 400, "h": 200},
        {"file_path": "images/b.jpg", "fl_x": 150.0, "fl_y": 150.0, "cx": 100.0, "cy": 50.0, "w": 200, "h": 100}]}
    a, b = (frame_camera.camera(t, f) for f in t["frames"])
    assert frame_camera.long_side(t) == 400
    assert frame_camera.train_scale(t, a, 2) == 0.5 and frame_camera.train_scale(t, b, 2) == 1.0
    assert frame_camera.train_scale(t, b, 4) == 0.5 and frame_camera.train_scale(t, b, 1) == 1.0
    tb = frame_camera.train_camera(t, b, 4)
    assert (tb["w"], tb["h"], tb["fl_x"], tb["cx"]) == (100, 50, 75.0, 50.0)
    assert frame_camera.train_scale(dict(t, mf_long_side=800), a, 4) == 0.5  # a tile without the largest camera


def test_training_data_at_one_resolution(tmp_path):
    """Several cameras: transforms_x<N>.json with every frame's camera at its training size, images_<N>/ for the
    photos that shrink, the others read from images/; one camera: the project and nerfstudio's factor."""
    from PIL import Image
    from modelforge import splat
    from modelforge.export import frame_camera
    (tmp_path / "images").mkdir()
    frames = []
    for k, (w, h) in enumerate([(400, 200), (200, 100), (400, 200)]):
        Image.new("RGB", (w, h), (k * 50, 90, 30)).save(tmp_path / "images" / f"p{k}.jpg")
        frames.append({"file_path": f"images/p{k}.jpg", "transform_matrix": np.eye(4).tolist(), "fl_x": w * 1.0,
                       "fl_y": w * 1.0, "cx": w / 2, "cy": h / 2, "w": w, "h": h})
    json.dump({"camera_model": "OPENCV", "frames": frames, "mf_cameras": 2, "mf_long_side": 400},
              open(tmp_path / "transforms.json", "w"))
    p = str(tmp_path)
    splat.prepare_downscaled(p, 4, on_line=lambda m: None)
    assert sorted(os.listdir(tmp_path / "images_4")) == ["p0.jpg", "p1.jpg", "p2.jpg"]
    assert Image.open(tmp_path / "images_4" / "p0.jpg").size == (100, 50)
    assert Image.open(tmp_path / "images_4" / "p1.jpg").size == (100, 50)
    data, factor = splat.training_data(p, 4)
    t = json.load(open(data))
    assert factor == 1 and os.path.basename(data) == "transforms_x4.json"
    assert [(f["w"], f["h"], f["fl_x"]) for f in t["frames"]] == [(100, 50, 100.0)] * 3
    assert all(f["file_path"].startswith("images_4/") for f in t["frames"])
    shutil.rmtree(tmp_path / "images_2", ignore_errors=True)
    splat.prepare_downscaled(p, 2, on_line=lambda m: None)
    assert sorted(os.listdir(tmp_path / "images_2")) == ["p0.jpg", "p2.jpg"]  # p1 trains at full size
    t = json.load(open(splat.training_data(p, 2)[0]))
    assert [f["file_path"] for f in t["frames"]] == ["images_2/p0.jpg", "images/p1.jpg", "images_2/p2.jpg"]
    assert all((f["w"], f["h"]) == (200, 100) for f in t["frames"])
    cmd = splat.train_command(p, data=splat.training_data(p, 2)[0], downscale=1)
    assert cmd[cmd.index("--data") + 1].endswith("transforms_x2.json") and cmd[-1] == "1"
    one = dict(json.load(open(tmp_path / "transforms.json")), frames=[frames[0], frames[2]])
    json.dump(one, open(tmp_path / "transforms.json", "w"))
    assert splat.training_data(p, 2) == (p, 2)
    # the readers: semantic features take each photo's camera
    from modelforge import semantic
    json.dump(dict(one, frames=frames), open(tmp_path / "transforms.json", "w"))
    views, K, dist, w, h, cam, paths = semantic._cameras(p)
    assert list(w) == [400, 200] and list(cam) == [0, 1, 0] and K.shape == (2, 3, 3)
    assert frame_camera.long_side(json.load(open(tmp_path / "transforms.json"))) == 400


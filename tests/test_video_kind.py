"""Project kind "video" (no metadata): ingest by image motion, pose-free priors, sequential pairs,
incremental SfM through the GUI (the last part on the sample clip, skipped without it)."""

import json
import os
import socket
import subprocess
import time

import numpy as np
import pytest

from modelforge import pipeline, projects
from modelforge.dji import extract as ex
from modelforge.dji import ingest as dji
from modelforge.sfm import pairs, priors
from modelforge.video import ingest as vi
from tests.sample_clip import sample_video

VIDEO = sample_video()


def panning_clip(path, still=1.0, pan=2.0, rotate=None):
    """A 'camera' standing still for `still` s, then panning across a texture for `pan` s (10 fps)."""
    n_still, n_pan = int(still * 10), int(pan * 10)
    tex = path + ".png"  # one frozen frame of the test pattern (testsrc2 itself animates)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=480x120", "-frames:v", "1",
                    tex], check=True)
    expr = f"if(lt(n,{n_still}),0,(n-{n_still})*4)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "10", "-i", tex,
                    "-frames:v", str(n_still + n_pan), "-vf", f"crop=160:90:x='{expr}':y=10", "-pix_fmt", "yuv420p",
                    path], check=True)
    if rotate:
        tmp = path + ".rot.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotate), "-i", path, "-c", "copy",
                        tmp], check=True)
        os.replace(tmp, path)
    return n_still, n_pan


def test_motion_sampling_and_extraction(tmp_path):
    clip = str(tmp_path / "pan.mp4")
    n_still, n_pan = panning_clip(clip)
    c = vi.load_clip(clip)
    assert c.n == n_still + n_pan and (c.width, c.height) == (160, 90)
    m = c.motion
    assert np.median(m[n_still + 1:]) > 10 * max(m[1:n_still].max(), 1e-4)  # still, then moving
    plan = vi.make_plan([dji.Source(clip)], dji.Settings(sampling="motion", num_images=8))
    frames = [(a + b) // 2 for a, b in plan.sources[0].windows]
    assert plan.num_images == 8 and min(frames) >= n_still - 1  # nothing while standing still
    timed = vi.make_plan([dji.Source(clip)], dji.Settings(sampling="time", num_images=8))
    assert min((a + b) // 2 for a, b in timed.sources[0].windows) < n_still  # equal time steps do sample it
    proj = str(tmp_path / "p")
    projects.create(str(tmp_path), "p", kind="video")
    vi.run(proj, [dji.Source(clip)], dji.Settings(sampling="motion", num_images=8))
    assert dji.load_kind(proj) == "video" and len(os.listdir(os.path.join(proj, "images"))) == 8
    meta = json.load(open(os.path.join(proj, "poses.json")))
    assert meta["kind"] == "video" and "colmap_qvec" not in meta["images"][0]
    pri, _ = priors.load(proj)
    assert not priors.has_poses(pri) and len(pri) == 8
    report = pipeline.Report(proj)
    pipeline.stage_pairs(proj, report, pri, meta)
    assert report.data["pairs"] == {"num_pairs": 28, "method": "sequential"}  # 8 images: exhaustive
    assert pipeline.sfm_stages("video") == ("features", "pairs", "match", "reconstruct", "export")


def test_portrait_video(tmp_path):
    clip = str(tmp_path / "portrait.mp4")
    panning_clip(clip, still=0.0, pan=1.0, rotate=90)
    assert ex.video_size(clip) == (90, 160)  # as decoded (ffmpeg applies the rotation)
    proj = str(tmp_path / "q")
    vi.run(proj, [dji.Source(clip)], dji.Settings(sampling="time", num_images=3))
    from PIL import Image
    name = sorted(os.listdir(os.path.join(proj, "images")))[0]
    assert Image.open(os.path.join(proj, "images", name)).size == (90, 160)


def test_kind_in_projects(tmp_path):
    root = str(tmp_path)
    p = projects.create(root, "phone", kind="video")
    sources, settings = dji.load_plan(p)
    assert dji.load_kind(p) == "video" and settings.sampling == "motion"
    assert projects.status(p)["kind"] == "video"
    q = projects.create(root, "copy", copy_plan_from=p)  # a copy takes the kind along
    assert dji.load_kind(q) == "video"
    assert dji.load_kind(projects.create(root, "drone")) == "dji"
    with pytest.raises(ValueError):
        projects.create(root, "x", kind="lidar")


def test_sequential_pairs():
    s = [[f"a{i}" for i in range(200)], [f"b{i}" for i in range(100)]]
    p = set(pairs.sequential(s))
    assert ("a0", "a1") in p and ("a0", "a10") in p and ("a0", "a11") not in p  # +-10 neighbours
    assert ("a0", "a16") in p and ("a0", "a128") in p  # quadratic jumps
    assert ("a0", "b0") in p and ("a1", "b0") not in p  # coarse grid between sources


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="needs the sample clip berlin_3_1.MP4 in data/")
def test_video_project_through_the_gui(tmp_path):
    """The sample clip treated as a video without metadata: create, ingest 20 images, run SfM."""
    from modelforge.gui.app import App
    root = str(tmp_path / "data")
    os.makedirs(root)
    path = projects.create(root, "phone", kind="video")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    app = App(path, port=port, root=root)
    try:
        ing = app.ingest
        assert ing.kind == "video" and ing.g_sampling.options == ("motion", "time")
        assert not ing.g_rotw.visible and not ing.g_yaw.visible and ing.g_extract.label == "Extract images"
        assert app.g_buttons["reconstruct"].visible and not app.g_buttons["refine"].visible
        assert not app.g_ba_iters.visible
        ing.sources = [dji.Source(os.path.relpath(VIDEO), 105.0, 117.0)]
        ing.settings = dji.Settings(sampling="motion", num_images=20)
        ing._rebuild_sources()
        ing.request_plan()
        t0 = time.time()
        while (ing.plan is None or ing.plan.num_images != 20) and time.time() - t0 < 300:
            time.sleep(0.5)
        assert ing.plan.num_images == 20
        ing.extract()
        while ing.running and time.time() - t0 < 600:
            time.sleep(0.5)
        assert app.project is not None and len(app.project.names) == 20 and not app.project.has_priors
        app.run_stage("sfm")
        while app.running and time.time() - t0 < 900:
            time.sleep(0.5)
        assert "sfm finished" in app.g_state.content, app.g_state.content
        r = pipeline.Report(path).data["reconstruct"]
        assert r["reg_images"] == 20 and r["camera"]["model"] == "RADIAL"
        assert abs(r["camera"]["params"][0] - 1085) < 60  # the DJI refinement found f = 1085 px
        assert app.snap.name == "refined" and app.snap.cams.valid.sum() == 20
        assert os.path.exists(os.path.join(path, "transforms.json"))
    finally:
        app.server.stop()

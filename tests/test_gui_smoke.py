"""Headless smoke test of the GUI logic on a real project (skipped if absent)."""

import os
import socket

import numpy as np
import pytest

PROJECT = os.environ.get("MODELFORGE_TEST_PROJECT",  # any refined project
                         os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(PROJECT, "sparse", "refined", "images.bin")),
                                reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app():
    from modelforge.gui.app import App
    a = App(PROJECT, port=free_port())
    yield a
    a.server.stop()


def test_snapshots_and_color_modes(app):
    from modelforge.gui.app import COLOR_BY
    for name in app.project.snapshot_names():
        app.g_snapshot.value = name  # triggers load_snapshot via on_update
        app.load_snapshot(name)
        assert app.snap.name == name
        for mode in COLOR_BY:
            app.g_color.value = mode
            assert app.color_values() is None or len(app.color_values()) == len(app.project.names)
    app.load_snapshot("refined")
    assert "refined" in app.g_stats.content and "prior deviation" in app.g_stats.content
    assert len(app.cam_handles) == len(app.project.names)
    assert not app.g_cams.value and not any(h.visible for h in app.cam_handles.values())  # off by default
    app.g_cams.value = True
    assert any(h.visible for h in app.cam_handles.values())


def test_inspect(app):
    app.load_snapshot("refined")
    app.select(len(app.project.names) // 2)
    assert app.g_image.image.shape[1] == 960
    assert "keypoints observed" in app.g_inspect_md.content


def test_live_refine_events(app):
    """Feed the refine progress events the GUI receives, using a saved reconstruction."""
    import pycolmap
    rec = pycolmap.Reconstruction(os.path.join(PROJECT, "sparse", "round_0"))
    app._last_live = 0.0
    app._on_refine_event({"phase": "triangulate", "round": 0, "rounds": 3, "rec": rec})
    app._on_refine_event({"phase": "ba", "round": 0, "rounds": 3, "iteration": 5, "cost": 1.0, "rec": rec})
    cams = app.project.poses_from_reconstruction(rec)
    i = int(np.flatnonzero(cams.valid)[0])
    assert np.allclose(app.cam_handles[i].position, cams.centers[i])
    app._on_refine_event({"phase": "round_done", "round": 0, "rounds": 3, "rec": rec, "eval": {}})
    assert app.snap.name == "round_0" and len(app.snap.points) > 0


def test_read_snapshot_matches_reconstruction():
    """The numpy reader of the binary model gives what pycolmap.Reconstruction gives (points in id order)."""
    import pycolmap
    from modelforge.gui.data import Project
    p = Project(PROJECT)
    fast = p.read_snapshot("refined")
    rec = pycolmap.Reconstruction(os.path.join(PROJECT, "sparse", "refined"))
    slow = p.snapshot_from_reconstruction("refined", rec)
    v = slow.cams.valid
    assert (fast.cams.valid == v).all() and fast.rec is None
    assert np.allclose(fast.cams.centers[v], slow.cams.centers[v]) and np.allclose(fast.cams.R_wc[v], slow.cams.R_wc[v])
    order = np.argsort(np.array(list(rec.points3D.keys())))
    assert np.array_equal(fast.points, slow.points[order]) and np.array_equal(fast.colors, slow.colors[order])
    assert np.allclose(fast.point_error, slow.point_error[order], atol=1e-5)
    assert np.allclose(fast.reproj_error, slow.reproj_error, equal_nan=True, atol=1e-6)
    assert np.array_equal(fast.num_obs, slow.num_obs) and fast.intrinsics == slow.intrinsics
    i = int(np.flatnonzero(v)[0])
    a, b = p.inspect_image(slow, i)[1], p.inspect_image(fast, i)[1]
    assert a["observations"] == b["observations"] and a["keypoints"] == b["keypoints"]
    assert abs(a["median_err_px"] - b["median_err_px"]) < 1e-6

"""GUI settings: dark / light mode, remembered across restarts."""

import json
import os
import socket

from modelforge.dji import ingest


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_dark_mode_persists(tmp_path):
    from modelforge.gui.app import App
    root = tmp_path / "data"
    proj = root / "p"
    proj.mkdir(parents=True)
    ingest.save_plan(str(proj), [], ingest.Settings())
    a = App(str(proj), port=free_port(), root=str(root))
    try:
        assert a.dark is False and a.g_dark.value is False and a.plot_template() == "plotly_white"
        a.set_dark(True)
        assert json.load(open(os.path.join(root, ".modelforge_gui.json")))["dark_mode"] is True
        assert a.plot_template() == "plotly_dark"
    finally:
        a.server.stop()
    b = App(str(proj), port=free_port(), root=str(root))  # a new session starts dark
    try:
        assert b.dark is True and b.g_dark.value is True
    finally:
        b.server.stop()


def test_last_opened_project_is_reopened(tmp_path):
    """Startup opens the project opened last, not the newest one; a deleted one falls back to the newest."""
    import shutil
    import time
    from modelforge.gui.app import App, last_project
    root = tmp_path / "data"
    for name in ("old", "new"):
        (root / name).mkdir(parents=True)
        ingest.save_plan(str(root / name), [], ingest.Settings())
        time.sleep(0.05)
    assert last_project(str(root)) == str(root / "new")  # nothing remembered: the newest
    a = App(str(root / "new"), port=free_port(), root=str(root))
    try:
        a.open_project(str(root / "old"))
    finally:
        a.server.stop()
    assert last_project(str(root)) == str(root / "old")  # opened last, though not modified since
    a = App(str(root / "old"), port=free_port(), root=str(root))
    try:
        a.set_dark(True)  # one setting does not wipe another
    finally:
        a.server.stop()
    assert json.load(open(os.path.join(root, ".modelforge_gui.json"))) == {"last_project": "old", "dark_mode": True}
    shutil.rmtree(root / "old")
    assert last_project(str(root)) == str(root / "new")


def test_orbit_runs_in_the_browser(tmp_path):
    """Orbit is the browsers' own per-frame rotation (patched viser bundle): the server only publishes the speed
    (data-orbit on the hidden #mf-move-speed element) and never sets the camera, so wheel zoom keeps working.
    The speed slider applies while orbiting; Fly turns the orbit off."""
    import contextlib

    from modelforge.gui.app import App
    root = tmp_path / "data"
    proj = root / "p"
    proj.mkdir(parents=True)
    ingest.save_plan(str(proj), [], ingest.Settings())

    class Camera:
        position, look_at = (10.0, 0.0, 5.0), (0.0, 0.0, 0.0)

    class Client:
        client_id, camera = 7, Camera()

        def atomic(self):
            return contextlib.nullcontext()

    a = App(str(proj), port=free_port(), root=str(root))
    try:
        c = Client()
        assert 'data-orbit="0"' in a.g_speed_html.content
        a.g_orbit_speed.value = 30
        a.toggle_orbit(c)
        assert a.g_orbit.label == "Stop orbit" and 'data-orbit="30"' in a.g_speed_html.content
        a.g_orbit_speed.value = 45
        assert 'data-orbit="45"' in a.g_speed_html.content
        assert c.camera.position == (10.0, 0.0, 5.0)  # the server does not drive the camera
        a.toggle_orbit(c)
        assert a.g_orbit.label == "Start orbit" and 'data-orbit="0"' in a.g_speed_html.content
        a.toggle_orbit(c)
        a._set_orbit(False)  # what Fly / Center / a glide do
        assert not a.orbit_on and 'data-orbit="0"' in a.g_speed_html.content
    finally:
        a.server.stop()


def test_flight_path_interpolation():
    """Through every camera (position and orientation), smooth in between, skipping unregistered
    cameras; parametrized by distance, turns counted by angle, jumps at long pauses shortened."""
    import numpy as np

    from modelforge.gui import data as gdata
    n = 8
    cams = gdata.Cameras.empty(n)
    times = np.array([0, 1, 2, 3, 10, 11, 12, 13], float)  # a 7 s pause (e.g. a second video)
    for i in range(n):
        a = 0.3 * i
        cams.centers[i] = (10 * np.cos(a), 10 * np.sin(a), 5.0 + i)
        c, s = np.cos(a), np.sin(a)
        cams.R_wc[i] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])  # looking down
    cams.centers[2] = np.nan  # not registered
    p = gdata.FlightPath(cams, times)
    c = cams.centers[p.index]
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    seg[2] = min(seg[2], 2.0)  # the jump at the 7 s pause: at most 2 m
    assert list(p.index) == [0, 1, 3, 4, 5, 6, 7] and np.allclose(p.t, np.concatenate([[0], np.cumsum(seg)]))
    for k, i in enumerate(p.index):
        pos, R, at = p.pose(p.t[k])
        assert np.allclose(pos, cams.centers[i]) and np.allclose(R, cams.R_wc[i], atol=1e-9) and at == i
    ts = np.linspace(0, p.duration, 701)
    pts = np.array([p.pose(t)[0] for t in ts])
    assert np.abs(np.diff(pts, axis=0)).max() < 0.1  # continuous: no jumps between the samples
    _, R, _ = p.pose(0.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(p.pose(-5)[0], cams.centers[0]) and np.allclose(p.pose(99)[0], cams.centers[7])
    q = gdata.FlightPath(cams, np.full(n, np.nan))  # no times: no pause shortening
    assert np.isclose(q.duration, np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    spin = gdata.Cameras.empty(2)  # a turn on the spot: 90 deg = 1.8 m of path
    spin.centers[:] = 0.0
    spin.R_wc[0], spin.R_wc[1] = np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.isclose(gdata.FlightPath(spin, [0, 1]).duration, 90 * gdata.ROT_M_PER_DEG)

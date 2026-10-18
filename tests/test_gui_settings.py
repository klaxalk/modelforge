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


def test_orbit_keeps_center_distance_and_height(tmp_path):
    import contextlib
    import time

    import numpy as np

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
        a.g_orbit_speed.value = 90
        a.toggle_orbit(c)
        assert a.g_orbit.label == "Stop orbit"
        time.sleep(0.5)
        a.toggle_orbit(c)
        assert a.g_orbit.label == "Start orbit"
        p = np.array(c.camera.position)
        assert np.allclose(c.camera.look_at, 0) and abs(p[2] - 5) < 1e-9 and abs(np.hypot(p[0], p[1]) - 10) < 1e-6
        moved = np.degrees(np.arctan2(p[1], p[0]))
        assert 20 < moved < 70  # ~45 deg after 0.5 s at 90 deg/s
        time.sleep(0.2)
        assert np.allclose(c.camera.position, p)  # stopped

        # mouse adjustments during an orbit stick: it pauses, then continues from the new view
        a.toggle_orbit(c)
        time.sleep(0.2)
        c.camera.position, c.camera.look_at = (3.0, 4.0, 20.0), (3.0, -16.0, 1.0)  # zoom out, tilt, pan
        time.sleep(0.3)  # within the hold: not rotating
        assert np.allclose(c.camera.position, (3.0, 4.0, 20.0))
        time.sleep(0.8)  # hold over: orbiting again around the new look-at, new distance and height
        a.toggle_orbit(c)
        q, ctr = np.array(c.camera.position), np.array(c.camera.look_at)
        assert np.allclose(ctr, (3.0, -16.0, 1.0)) and abs(q[2] - 20.0) < 1e-9
        assert abs(np.hypot(*(q - ctr)[:2]) - 20.0) < 1e-6 and not np.allclose(q, (3.0, 4.0, 20.0))
    finally:
        a.server.stop()


def test_orbit_never_steps_back_with_lagging_echoes(tmp_path):
    """A browser reports the camera a few updates late; the orbit must not restart from those."""
    import contextlib
    import time

    import numpy as np

    from modelforge.gui.app import App
    root = tmp_path / "data"
    proj = root / "p"
    proj.mkdir(parents=True)
    ingest.save_plan(str(proj), [], ingest.Settings())

    class LaggingCamera:
        def __init__(self):
            self.history = [(10.0, 0.0, 5.0)]  # poses set by the server, oldest first
            self.look_at = (0.0, 0.0, 0.0)

        @property
        def position(self):  # the browser's report: 3 updates behind
            return self.history[max(0, len(self.history) - 4)]

        @position.setter
        def position(self, p):
            self.history.append(tuple(p))

    class Client:
        client_id, camera = 9, LaggingCamera()

        def atomic(self):
            return contextlib.nullcontext()

    a = App(str(proj), port=free_port(), root=str(root))
    try:
        c = Client()
        a.g_orbit_speed.value = 30
        a.toggle_orbit(c)
        time.sleep(0.6)
        a.toggle_orbit(c)
        ang = np.unwrap([np.arctan2(p[1], p[0]) for p in c.camera.history])
        assert len(ang) > 10 and np.all(np.diff(ang) > 0)  # every step forward
    finally:
        a.server.stop()


def test_flight_path_interpolation():
    """Through every camera at its time (position and orientation), smooth in between, skipping
    unregistered cameras, shortening long pauses, one image per second without usable times."""
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
    assert list(p.index) == [0, 1, 3, 4, 5, 6, 7] and np.allclose(p.t, [0, 1, 3, 4, 5, 6, 7])  # pause -> 1 s
    for k, i in enumerate(p.index):
        pos, R, at = p.pose(p.t[k])
        assert np.allclose(pos, cams.centers[i]) and np.allclose(R, cams.R_wc[i], atol=1e-9) and at == i
    ts = np.linspace(0, p.duration, 701)
    pts = np.array([p.pose(t)[0] for t in ts])
    assert np.abs(np.diff(pts, axis=0)).max() < 0.1  # continuous: no jumps between the samples
    _, R, _ = p.pose(0.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(p.pose(-5)[0], cams.centers[0]) and np.allclose(p.pose(99)[0], cams.centers[7])
    q = gdata.FlightPath(cams, np.full(n, np.nan))  # no times: one image per second
    assert np.allclose(q.t, np.arange(7))

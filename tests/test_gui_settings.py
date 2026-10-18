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

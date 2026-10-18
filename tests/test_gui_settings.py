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

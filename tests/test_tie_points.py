"""Manual tie points (sfm/tie_points.py, gui/tiepoints.py): storage, and the Tie points tab's flow on a real project:
a Ctrl + clicked spot lists the photos that see it, a click zooms in, a second click records the spot's pixel."""
import os
import shutil
import time

import numpy as np
import pytest

from modelforge.sfm import tie_points as tp

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


def test_storage(tmp_path):
    p = str(tmp_path)
    assert tp.load(p) == [] and tp.new_name([]) == "T1"
    tp.set_obs(p, "T1", "a.jpg", 10.0, 20.0)
    tp.set_obs(p, "T1", "b.jpg", 11.0, 21.0)
    tp.set_obs(p, "T1", "a.jpg", 12.0, 22.0)  # clicked again: moved
    pts = tp.load(p)
    assert [(o["image"], o["x"]) for o in pts[0]["obs"]] == [("b.jpg", 11.0), ("a.jpg", 12.0)]
    assert tp.new_name(pts) == "T2"
    tp.remove(p, "T1", "b.jpg")
    assert len(tp.load(p)[0]["obs"]) == 1
    tp.remove(p, "T1")
    assert tp.load(p) == []


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project")
def test_tab_records_a_spot(tmp_path):
    import pycolmap
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = tmp_path / "proj"
    p.mkdir()
    for n in ("poses.json", "images", "sparse"):
        os.symlink(os.path.abspath(os.path.join(REAL, n)), p / n)
    shutil.copy2(os.path.join(REAL, "report.json"), p / "report.json")
    rec = pycolmap.Reconstruction(os.path.join(REAL, "sparse", "refined"))
    pid = max(rec.points3D, key=lambda k: rec.points3D[k].track.length())
    X = rec.points3D[pid].xyz
    app = App(str(p), port=free_port())
    try:
        tpp = app.tiepoints
        tpp.on_pick(X)
        assert len(tpp.candidates) >= 2 and tpp.g_photos.value == tpp.candidates[0][0]
        tpp._photo_button("Open photo")
        name = tpp.view["name"]
        iid = next(i for i in rec.reg_image_ids() if rec.image(i).name == name)
        q = tp.project(rec, iid, X)
        W, H = tpp.view["size"]
        tpp._clicked(f"{q[0] / W},{q[1] / H},1")  # the first click zooms in around it
        assert tpp.view["zoomed"]
        x0, y0, x1, y1 = tpp.view["box"]
        assert x0 <= q[0] <= x1 and y0 <= q[1] <= y1
        tpp._clicked(f"{(q[0] - x0) / (x1 - x0)},{(q[1] - y0) / (y1 - y0)},2")  # the second records it
        o = tp.load(str(p))[0]["obs"][0]
        assert o["image"] == name and abs(o["x"] - q[0]) < 0.01 and abs(o["y"] - q[1]) < 0.01
        assert "1 clicks" in tpp.g_status.content and not tpp.view["zoomed"]
        assert 'data-mf-click="' in tpp.g_img.content
        tpp._photo_button("▶")
        assert tpp.view["name"] != name
    finally:
        app.server.stop()

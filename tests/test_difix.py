"""Difix refinement (modelforge/difix.py): saved views -> camera poses, the splat ply written back, Undo."""
import json
import os
import shutil
import time

import numpy as np
import pytest

from modelforge import difix, splat_mesh
from modelforge.gui import views


def test_view_to_camera():
    v = views.pose((1.0, 2.0, 3.0), (1.0, 12.0, 3.0))  # looking north, level
    c = difix.c2w(v)
    assert np.allclose(c[:3, 3], [1, 2, 3])
    assert np.allclose(c[:3, 2], [0, 1, 0]) and np.allclose(c[:3, 0], [1, 0, 0]) and np.allclose(c[:3, 1], [0, 0, -1])
    down = difix.c2w(views.pose((0, 0, 50.0), (0, 0, 0)))  # straight down: still a rotation
    assert np.allclose(down[:3, :3] @ down[:3, :3].T, np.eye(3)) and np.allclose(down[:3, 2], [0, 0, -1])


def test_write_splat_and_undo(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    g = {"means": rng.normal(size=(n, 3)), "quats": rng.normal(size=(n, 4)), "scales": rng.normal(size=(n, 3)),
         "opacities": rng.normal(size=n), "sh": rng.normal(size=(n, 16, 3)).astype(np.float32)}
    p = tmp_path / "proj"
    (p / "splat").mkdir(parents=True)
    sp = str(p / "splat" / "splat_enu.ply")
    difix.write_splat(None, sp, g)
    back = splat_mesh.load_splat(sp)
    assert all(np.allclose(back[k], g[k], atol=1e-5) for k in g)
    g2 = dict(g, means=g["means"] + 1.0)
    difix.write_splat(sp, sp, g2)  # the template's layout kept
    assert np.allclose(splat_mesh.load_splat(sp)["means"], g2["means"], atol=1e-5)
    with pytest.raises(ValueError):
        difix.write_splat(sp, sp, {k: v[:10] for k, v in g.items()})
    os.makedirs(os.path.dirname(p / difix.BACKUP), exist_ok=True)
    difix.write_splat(None, str(p / difix.BACKUP), g)
    json.dump({"difix": {"targets": 1}}, open(p / "report.json", "w"))
    assert difix.refined(str(p))["backup"]
    difix.undo(str(p))
    assert np.allclose(splat_mesh.load_splat(sp)["means"], g["means"], atol=1e-5)
    assert difix.refined(str(p)) == {"backup": False, "info": None}
    with pytest.raises(RuntimeError):
        difix.undo(str(p))


def _scene(p, hole=(6.0, 6.0, 4.0)):
    """A 30 x 30 m ground (SfM mesh), a splat of small Gaussians on it except a square hole, photos at 10 m."""
    from modelforge import sfm_mesh
    import plyfile
    os.makedirs(p / "sfmmesh")
    os.makedirs(p / "splat")
    xs = np.linspace(-15, 15, 61)
    X, Y = np.meshgrid(xs, xs)
    v = np.zeros(X.size, [("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"] = X.ravel(), Y.ravel()
    k = np.arange(61 * 61).reshape(61, 61)
    a, b, c, d = k[:-1, :-1].ravel(), k[:-1, 1:].ravel(), k[1:, :-1].ravel(), k[1:, 1:].ravel()
    sfm_mesh.write_mesh(str(p / "sfmmesh" / "mesh.ply"), v, np.r_[np.c_[a, b, d], np.c_[a, d, c]])
    json.dump({"point_spacing": 0.25}, open(p / "sfmmesh" / "stats.json", "w"))
    rng = np.random.default_rng(0)
    G = rng.uniform(-15, 15, (60000, 2))
    hx, hy, hs = hole
    G = G[~((np.abs(G[:, 0] - hx) < hs / 2) & (np.abs(G[:, 1] - hy) < hs / 2))]
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    g = np.zeros(len(G), [(n, "f4") for n in names])
    g["x"], g["y"], g["opacity"] = G[:, 0], G[:, 1], 3.0
    g["scale_0"] = g["scale_1"] = g["scale_2"] = np.log(0.05)
    g["rot_0"] = 1
    plyfile.PlyData([plyfile.PlyElement.describe(g, "vertex")]).write(str(p / "splat" / "splat_enu.ply"))
    frames = []
    for x in np.linspace(-12, 12, 7):
        for y in np.linspace(-12, 12, 7):
            m = np.eye(4)
            m[:3, :3] = np.diag([1.0, -1.0, -1.0])  # OpenGL camera looking straight down
            m[:3, 3] = [x, y, 10.0]
            frames.append({"file_path": f"images/{len(frames)}.jpg", "transform_matrix": m.tolist()})
    json.dump({"fl_x": 1000.0, "fl_y": 1000.0, "w": 1920, "h": 1080, "frames": frames}, open(p / "transforms.json", "w"))


def test_suggested_views_look_at_the_thin_spot(tmp_path):
    """suggest_views (the user's idea, 2026-10-26): a gap in the splat on the mesh gets a view looking at it from
    above, within the flown height; adding them replaces earlier suggestions and keeps views saved by hand."""
    p = tmp_path / "proj"
    _scene(p)
    assert difix.can_suggest(str(p)) is None
    out, st = difix.suggest_views(str(p), n=3, on_line=lambda m: None)
    assert out and 0 < st["thin_share"] < 0.1
    v = out[0]
    at, pos = np.array(v["look_at"]), np.array(v["position"])
    assert abs(at[0] - 6) < 2 and abs(at[1] - 6) < 2  # the hole
    assert 0 < pos[2] <= 10.0 + 1e-6  # above the ground, not above the photos
    down = np.degrees(np.arcsin((pos[2] - at[2]) / np.linalg.norm(pos - at)))
    assert difix.MIN_DOWN_DEG - 1e-6 <= down <= difix.MAX_DOWN_DEG + 1e-6
    views.add(str(p), "mine", views.pose([0, 0, 5], [0, 0, 0]), difix.VIEWS_FILE)
    difix.add_suggested(str(p), out)
    difix.add_suggested(str(p), out[:1])  # again: replaces the suggested ones only
    names = [x["name"] for x in views.load(str(p), difix.VIEWS_FILE)]
    assert names == ["mine", out[0]["name"]]
    assert difix.clear_views(str(p)) == 1  # the suggested one; the hand-saved stays
    assert [x["name"] for x in views.load(str(p), difix.VIEWS_FILE)] == ["mine"]
    assert difix.clear_views(str(p), suggested_only=False) == 1
    assert views.load(str(p), difix.VIEWS_FILE) == []


def test_suggested_views_keep_clear_of_the_splat(tmp_path):
    """A view keeps the clearance the photos there have, also from what only the splat holds (the user saw views
    almost touching things, 2026-10-27): opaque Gaussians where the first suggestion stood move it away."""
    import plyfile
    p = tmp_path / "proj"
    _scene(p)
    out, _ = difix.suggest_views(str(p), n=1, on_line=lambda m: None)
    first = np.array(out[0]["position"])
    sp = str(p / "splat" / "splat_enu.ply")
    g = plyfile.PlyData.read(sp)["vertex"].data
    blob = np.zeros(200, g.dtype)
    blob["x"], blob["y"], blob["z"] = (first[:, None] + np.random.default_rng(1).normal(0, 0.3, (3, 200)))
    blob["opacity"], blob["rot_0"] = 3.0, 1
    blob["scale_0"] = blob["scale_1"] = blob["scale_2"] = np.log(0.05)
    plyfile.PlyData([plyfile.PlyElement.describe(np.concatenate([g, blob]), "vertex")]).write(sp)
    out, st = difix.suggest_views(str(p), n=1, on_line=lambda m: None)
    pos = np.array(out[0]["position"])
    B = np.stack([blob["x"], blob["y"], blob["z"]], 1)
    assert st["too_close"] >= 1 and np.linalg.norm(B - pos, axis=1).min() > 1.5


def test_baseline_follows_a_new_training(tmp_path):
    """Difix runs build on each other and Compare / Undo use the splat from before the first one (the user, 2026-10-27);
    a new training replaces that baseline: the old one, its level of detail and the Difix record go."""
    from modelforge import splat_lod
    from modelforge.pipeline import Report
    p = tmp_path / "proj"
    _scene(p)
    sp = str(p / "splat" / "splat_enu.ply")
    shutil.copy2(sp, p / difix.BACKUP)  # as a Difix run before the record existed left it
    os.makedirs(splat_lod.lod_dir(str(p), difix.BACKUP))
    Report(str(p)).set("difix", {"targets": 1})
    assert difix.baseline(str(p)) == str(p / difix.BACKUP)  # adopted: the current splat is Difix's output
    assert difix.baseline(str(p)) and difix.refined(str(p))["backup"]  # unchanged: kept
    time.sleep(0.01)
    with open(sp, "ab") as f:  # a new training wrote the splat
        f.write(b"\0")
    assert difix.baseline(str(p)) is None
    assert not os.path.exists(p / difix.BACKUP) and not os.path.isdir(splat_lod.lod_dir(str(p), difix.BACKUP))
    assert not os.path.exists(p / difix.STATE) and Report(str(p)).data.get("difix") is None

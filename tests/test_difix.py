"""Difix refinement (modelforge/difix.py): saved views -> camera poses, the splat ply written back, Undo."""
import json
import os

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

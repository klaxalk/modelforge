"""modelforge/replace.py on a synthetic splat: textured flat ground with two blob trees (a big one, a small one)."""
import json
import os

import numpy as np
import plyfile
import pytest

from modelforge import difix, replace, splat_mesh


def _scene(tmp_path):
    rng = np.random.default_rng(0)
    xs = np.arange(-20, 20, 0.25)
    gx, gy = np.meshgrid(xs, xs)
    ground = np.stack([gx.ravel(), gy.ravel(), rng.normal(0, 0.02, gx.size)], 1)
    checker = ((np.floor(gx.ravel() / 2) + np.floor(gy.ravel() / 2)) % 2).astype(np.float32)
    gcol = np.stack([0.2 + 0.2 * checker, 0.5 - 0.1 * checker, 0.1 + 0 * checker], 1)

    def tree(cx, cy, h, r, n):
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        crown = np.array([cx, cy, 0.65 * h]) + u * rng.uniform(0.3, 1, (n, 1)) * np.array([r, r, 0.35 * h])
        trunk = np.stack([np.full(n // 10, cx), np.full(n // 10, cy), rng.uniform(0, 0.4 * h, n // 10)], 1)
        return np.concatenate([crown, trunk])
    big, small = tree(8, 8, 8.0, 3.0, 4000), tree(-8, -6, 6.0, 2.2, 1500)
    P = np.concatenate([ground, big, small]).astype(np.float32)
    n = len(P)
    col = np.concatenate([gcol, np.tile([0.1, 0.35, 0.1], (len(big), 1)), np.tile([0.15, 0.3, 0.12], (len(small), 1))])
    g = {"means": P, "quats": np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
         "scales": np.full((n, 3), np.log(0.12), np.float32), "opacities": np.full(n, 3.0, np.float32),
         "sh": ((col - 0.5) / 0.28209479177387814).astype(np.float32)[:, None, :]}
    g["scales"][: len(ground), 2] = np.log(0.02)  # flat ground
    os.makedirs(tmp_path / "splat")
    difix.write_splat(None, str(tmp_path / "splat" / "splat_enu.ply"), g)
    pts = np.concatenate([ground[::7], big[::5], small[::5]])
    v = np.zeros(len(pts), [(k, "f4") for k in "xyz"])
    for i, k in enumerate("xyz"):
        v[k] = pts[:, i]
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(str(tmp_path / "sparse_pc.ply"))
    frames = []
    for x in (-15, 0, 15):
        for y in (-15, 0, 15):
            c2w = np.eye(4)
            c2w[:3, 3] = [x, y, 20]
            frames.append({"file_path": "images/x.jpg", "transform_matrix": c2w.tolist()})
    json.dump({"fl_x": 1000.0, "fl_y": 1000.0, "cx": 480, "cy": 270, "w": 960, "h": 540, "frames": frames},
              open(tmp_path / "transforms.json", "w"))
    kind = np.zeros(n, bool)
    kind[len(ground):] = True
    return str(tmp_path), kind, len(ground), len(big)


def test_find_remove_fill_replace_undo(tmp_path):
    project, kind, n_ground, n_big = _scene(tmp_path)
    original = open(splat_mesh.splat_path(project), "rb").read()
    objs = replace.find(project, kind=kind, on_line=lambda m: None)
    assert len(objs) == 2
    big = min(objs, key=lambda o: np.hypot(o["x"] - 8, o["y"] - 8))
    small = min(objs, key=lambda o: np.hypot(o["x"] + 8, o["y"] + 6))
    assert big["id"] != small["id"]
    assert abs(big["ground_z"]) < 0.2 and abs(small["ground_z"]) < 0.2  # the ground from the SfM points
    assert 6.5 < big["height"] < 8.5 and 4.5 < small["height"] < 6.5
    assert replace.object_at(project, 8.3, 7.8) == big["id"] and replace.object_at(project, 0, 0) is None

    # remove the small tree, fill its ground
    replace.add(project, small["id"], on_line=lambda m: None)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    P = g["means"]
    col = np.hypot(P[:, 0] + 8, P[:, 1] + 6) < 2.0
    assert not (col & (P[:, 2] > 0.5)).any()  # nothing stands there any more
    assert (col & (np.abs(P[:, 2]) < 0.2)).sum() > 0.5 * 4 * np.pi / 0.25 ** 2 / 4  # ground filled in
    assert os.path.exists(os.path.join(project, replace.BACKUP))

    # replace the small tree by the big one instead: scaled to 6 m, standing on the ground
    replace.undo(project, on_line=lambda m: None)
    assert open(splat_mesh.splat_path(project), "rb").read() == original
    assert not os.path.exists(os.path.join(project, replace.BACKUP))
    replace.add(project, small["id"], donor=big["id"], on_line=lambda m: None)
    P = splat_mesh.load_splat(splat_mesh.splat_path(project))["means"]
    col = np.hypot(P[:, 0] + 8, P[:, 1] + 6) < 3.0
    top = P[col, 2].max()
    assert 4.5 < top < 6.8 and (col & (P[:, 2] > 0.5)).sum() > 0.5 * n_big
    assert np.hypot(P[:, 0] - 8, P[:, 1] - 8).min() < 1.0  # the donor stays where it was

    # the objects survive the edit (found on the base), undo all restores the original file
    assert replace.objects(project) is not None
    replace.undo(project, "all", on_line=lambda m: None)
    assert open(splat_mesh.splat_path(project), "rb").read() == original


def test_new_splat_drops_stale_edits(tmp_path):
    project, kind, *_ = _scene(tmp_path)
    objs = replace.find(project, kind=kind, on_line=lambda m: None)
    replace.add(project, objs[0]["id"], on_line=lambda m: None)
    g = splat_mesh.load_splat(os.path.join(project, replace.BACKUP))
    difix.write_splat(None, splat_mesh.splat_path(project), g)  # a "new training" writes the splat
    os.utime(splat_mesh.splat_path(project), ns=(1, 1))
    assert replace.base(project) == splat_mesh.splat_path(project)
    assert not os.path.exists(os.path.join(project, replace.BACKUP)) and replace.edits(project) == []
    assert replace.objects(project) is None


def test_patchfill_takes_matching_texture():
    """A hole in a striped field is filled from cells of the same stripe pattern (coherent offsets)."""
    from modelforge import patchfill
    H, W = 60, 80
    col = np.zeros((H, W, 3), np.float32)
    col[:, :, 1] = (np.arange(W) // 4 % 2)[None, :]  # vertical stripes, 4 cells wide
    R = {"H": H, "W": W, "cell": 0.5, "n": np.full((H, W), 5.0, np.float32), "obj": np.zeros((H, W), bool),
         "col": col, "feat": None, "rough": np.zeros((H, W), np.float32), "lo": np.zeros(2)}
    hole = np.zeros((H, W), bool)
    hole[20:30, 30:42] = True
    off, cost = patchfill.fill(R, hole, np.zeros_like(hole))
    hy, hx = np.nonzero(hole)
    got = col[hy + off[hy, hx, 0], hx + off[hy, hx, 1], 1]
    want = col[hy, hx, 1]  # the stripes continue through the hole
    assert (got == want).mean() > 0.9
    assert np.all(np.isfinite(cost[hole]))


def test_replace_panel_flow(tmp_path, monkeypatch):
    """The Replace tab with a stand-in App: find (kind mask), suggest, apply checked, pick + donor, undo all."""
    import viser
    from modelforge.gui import replace as gui_replace
    project, kind, *_ = _scene(tmp_path)
    original = open(splat_mesh.splat_path(project), "rb").read()
    monkeypatch.setattr(replace, "rate", lambda p, *a, **k: _rated(p))

    def _rated(p):  # the small tree smeared, the big one sharp
        objs = replace.objects(p)
        for o in objs:
            o.update(sharp=0.9 if o["height"] > 7 else 0.05, psnr=25.0, photos=3)
        replace._save_objects(p, objs)
        return objs
    real_find = replace.find
    monkeypatch.setattr(replace, "find", lambda p, qs, on_line: real_find(p, qs, on_line, kind=kind))

    class App:
        path, snap, fov, aspect = project, None, 1.0, 1.6

        def __init__(self, server):
            self.server = server

        def run_stage(self, stage):  # the job, at once
            assert stage == "replace"
            panel.run_job(project, lambda m: None, None, None)
            panel.refresh()

        def ray_hit(self, origin, direction):
            return np.array([-8.0, -6.0, 3.0])

        def glide_to(self, *a, **k):
            pass
    server = viser.ViserServer(port=0, verbose=False)
    try:
        app = App(server)
        panel = gui_replace.ReplacePanel(app, server.gui)
        panel._start(("find",))
        assert len(panel.objs) == 2 and "1 rendered much blurrier" in panel.g_status.content
        # automatic: the smeared small tree is proposed; no donor fits a 6 m tree from an 8 m one? (0.75: it does)
        panel.g_auto_action.value = "replace by a donor where one fits"
        panel.suggest()
        assert len(panel.proposals) == 1 and panel.proposals[0]["donor"] is not None
        panel.apply_checked()
        assert len(replace.edits(project)) == 1 and replace.edits(project)[0]["donor"] is not None
        panel._start(("undo", "all"))
        assert open(splat_mesh.splat_path(project), "rb").read() == original
        # manual: pick the small tree, remove + fill
        panel._arm()
        assert panel.on_pick(np.zeros(3), np.array([0, 0, -1.0]))
        small = panel.selected
        assert panel._objs_by_id()[small]["height"] < 7 and panel.donor_list
        panel.manual(False)
        assert [e["object"] for e in replace.edits(project)] == [small]
        assert not panel.on_pick(np.zeros(3), np.array([0, 0, -1.0]))  # not picking any more: the click is the pivot's
    finally:
        server.stop()


def test_ground_texture_repair(tmp_path):
    """A smeared disc of the checkered ground (a few large uniform Gaussians) gets the checker texture back from
    the ground around it; Gaussians outside the disc stay; undo restores the file."""
    project, kind, n_ground, _ = _scene(tmp_path)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    P = g["means"]
    disc = (np.hypot(P[:, 0] - 3, P[:, 1] + 10) < 2.0) & (np.arange(len(P)) < n_ground)
    keep = ~disc
    rng = np.random.default_rng(1)
    k = 12  # the smear
    blob = {"means": np.c_[3 + rng.uniform(-1.5, 1.5, (k, 2)) * [1, 1] + [0, -10] - [0, 0], np.zeros(k)],
            "quats": np.tile([1.0, 0, 0, 0], (k, 1)), "scales": np.full((k, 3), np.log(0.8)),
            "opacities": np.full(k, 3.0), "sh": np.tile(g["sh"][disc].mean(0), (k, 1, 1))}
    blob["means"][:, 1] = -10 + rng.uniform(-1.5, 1.5, k)
    blob["scales"][:, 2] = np.log(0.02)
    sm = {key: np.concatenate([v[keep], blob[key].astype(np.float32)]) for key, v in g.items()}
    difix.write_splat(None, splat_mesh.splat_path(project), sm)
    before = open(splat_mesh.splat_path(project), "rb").read()
    replace.find(project, kind=np.r_[kind[keep], np.zeros(k, bool)], on_line=lambda m: None)
    replace.add_many(project, [{"circle": [3.0, -10.0, 2.0]}], on_line=lambda m: None)
    out = splat_mesh.load_splat(splat_mesh.splat_path(project))
    Q, col = out["means"], out["sh"][:, 0, 0]
    inside = np.hypot(Q[:, 0] - 3, Q[:, 1] + 10) < 1.8
    assert inside.sum() > 0.6 * disc.sum()  # densely filled again
    assert np.all(np.abs(Q[inside, 2]) < 0.2)  # on the ground
    assert not (np.exp(out["scales"][inside].max(1)) > 0.5).any()  # the smear is gone
    assert col[inside].std() > 0.5 * g["sh"][disc, 0, 0].std()  # textured, not uniform
    far = np.hypot(P[:, 0] - 3, P[:, 1] + 10) > 4
    assert np.isin(P[far & keep][:, 0], Q[:, 0]).all()  # the rest untouched
    replace.undo(project, "all", on_line=lambda m: None)
    assert open(splat_mesh.splat_path(project), "rb").read() == before


def test_replace_panel_ground_texture(tmp_path, monkeypatch):
    """Ground texture in the Replace tab: Map weak ground (a stand-in detail map with one weak square), the area
    listed and outlined, Repair checked areas, Pick a spot + Repair circle, both edits listed, Undo all."""
    import viser
    from modelforge.gui import replace as gui_replace
    project, kind, *_ = _scene(tmp_path)
    original = open(splat_mesh.splat_path(project), "rb").read()
    real_find = replace.find
    monkeypatch.setattr(replace, "find", lambda p, qs, on_line: real_find(p, qs, on_line, kind=kind))
    monkeypatch.setattr(replace, "rate", lambda p, *a, **k: replace.objects(p))

    def fake_map(p, *a, **k):  # weak: a 6 x 6 m square at (0, 10); everything else sharp
        d = np.load(os.path.join(p, replace.DIR, "objects.npz"))
        lo = d["lo"]
        H, W = (np.array(d["crown"].shape) * float(d["cell"]) / replace.MAP_CELL).astype(int) + 1
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = lo[0] + (xx + 0.5) * replace.MAP_CELL, lo[1] + (yy + 0.5) * replace.MAP_CELL
        ratio = np.where((np.abs(cx) < 3) & (np.abs(cy - 10) < 3), 0.1, 0.9).astype(np.float32)
        np.savez(os.path.join(p, replace.DIR, "detail.npz"), ratio=ratio, weight=np.ones_like(ratio), lo=lo,
                 cell=replace.MAP_CELL, splat=np.array(replace._key(replace.base(p))))
        return replace.ground_areas(p)
    monkeypatch.setattr(replace, "detail_map", fake_map)

    class App:
        path, snap, fov, aspect = project, None, 1.0, 1.6

        def __init__(self, server):
            self.server = server

        def run_stage(self, stage):
            panel.run_job(project, lambda m: None, None, None)
            panel.refresh()

        def ray_hit(self, origin, direction):
            return np.array([-12.0, 12.0, 0.0])

        def glide_to(self, *a, **k):
            pass
    server = viser.ViserServer(port=0, verbose=False)
    try:
        app = App(server)
        panel = gui_replace.ReplacePanel(app, server.gui)
        panel._start(("find",))
        panel._start(("map",))
        assert len(panel.areas) == 1 and abs(panel.areas[0]["x"]) < 1 and abs(panel.areas[0]["y"] - 10) < 1
        assert len(panel._area_outlines()) >= 4 and "1 weak areas" in panel.g_area_md.content
        for _, cb in panel.area_props:
            cb.value = True
        panel.repair_checked()
        assert [e.get("area") for e in replace.edits(project)] == [panel.areas[0]["id"]]
        assert "1 repaired" in panel.g_area_md.content and not panel.area_props  # repaired: no longer offered
        panel._arm("circle")
        assert panel.on_pick(np.zeros(3), np.array([0, 0, -1.0])) and panel.circle == (-12.0, 12.0)
        panel.g_radius.value = 2.0
        panel.repair_circle()
        eds = replace.edits(project)
        assert len(eds) == 2 and eds[1]["circle"] == [-12.0, 12.0, 2.0] and panel.circle is None
        panel._start(("undo", "all"))
        assert open(splat_mesh.splat_path(project), "rb").read() == original
    finally:
        server.stop()

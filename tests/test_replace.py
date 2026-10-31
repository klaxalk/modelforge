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
            o.update(sharp=0.9 if o["height"] > 7 else 0.05, psnr=25.0, photos=3, side_photos=30, side_len=0.3)
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
        panel.g_turn.value = True
        panel.apply_checked()
        assert len(replace.edits(project)) == 1 and replace.edits(project)[0]["donor"] is not None
        assert replace.edits(project)[0]["turn"] is True
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

        def ray_hit(self, origin, direction):  # the ray's origin is the spot hit (the test clicks corners)
            return np.array([origin[0], origin[1], 0.0])

        def glide_to(self, *a, **k):
            pass
    server = viser.ViserServer(port=0, verbose=False)
    try:
        app = App(server)
        panel = gui_replace.ReplacePanel(app, server.gui)
        panel._start(("map",))  # without Find first: it finds the objects itself
        assert len(panel.areas) == 1 and abs(panel.areas[0]["x"]) < 1 and abs(panel.areas[0]["y"] - 10) < 1
        assert len(panel._area_outlines()) >= 4 and "1 weak areas" in panel.g_area_md.content
        for _, cb in panel.area_props:
            cb.value = True
        panel.repair_checked()
        assert [e.get("area") for e in replace.edits(project)] == [panel.areas[0]["id"]]
        assert "1 repaired" in panel.g_area_md.content and not panel.area_props  # repaired: no longer offered
        click = lambda x, y: panel.on_pick(np.array([x, y, 50.0]), np.array([0, 0, -1.0]))  # noqa: E731
        panel._arm("template")  # a template polygon: corners, one undone, Finish
        for x, y in ((8, -16), (16, -16), (16, -8), (30, 30)):
            assert click(x, y)
        panel._poly_action("template", "Undo corner")  # the stray (30, 30)
        panel._poly_action("template", "Finish")
        assert panel.closed["template"] and panel.g_copy_from.value == "my template"
        assert panel.poly["template"] == [[8.0, -16.0], [16.0, -16.0], [16.0, -8.0]]
        assert "32.0 m²" in panel.g_tmpl_md.content
        panel._arm("area")  # the area to repair
        for x, y in ((-14, 10), (-10, 10), (-10, 14), (-14, 14)):
            assert click(x, y)
        panel._poly_action("area", "Finish")
        assert panel.picking is None and not click(0, 0)  # done: Ctrl + click is the pivot's again
        panel.repair_drawn()
        eds = replace.edits(project)
        assert len(eds) == 2 and eds[1]["polygon"] == [[-14.0, 10.0], [-10.0, 10.0], [-10.0, 14.0], [-14.0, 14.0]]
        assert eds[1]["source"]["polygon"][0] == [8.0, -16.0] and not panel.poly["area"]
        panel._poly_action("template", "Clear")
        assert panel._source() == "auto"  # back to the most similar detailed patch
        panel._start(("undo", "all"))
        assert open(splat_mesh.splat_path(project), "rb").read() == original
    finally:
        server.stop()


def test_turn_rotates_offsets_orientation_and_view_colour():
    """_turn by 90 deg: an offset east goes north, a Gaussian long along x is long along y, and a colour seen
    brightest from the east (+x) is brightest from the north (+y)."""
    from modelforge.gui import data as gdata
    sh = np.zeros((1, 4, 3), np.float32)
    sh[0, 3, :] = 1.0  # degree 1: the third coefficient follows +x (gsplat's order: y, z, x)
    q = {"quats": np.array([[1.0, 0, 0, 0]], np.float32), "sh": sh.copy()}
    dx, dy = replace._turn(q, np.pi / 2, np.array([1.0]), np.array([0.0]))
    assert np.allclose([dx[0], dy[0]], [0, 1], atol=1e-6)
    w, x, y, z = q["quats"][0].astype(np.float64)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(R[:, 0], [0, 1, 0], atol=1e-5)  # the local x axis now points north
    cols = np.concatenate([np.full((1, 3), 0.5), q["sh"][:, 1:, :].reshape(1, -1)], 1)
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]])
    lum = [gdata.sh_view_colour(cols, d[None])[0].sum() for d in dirs]
    base = np.concatenate([np.full((1, 3), 0.5), sh[:, 1:, :].reshape(1, -1)], 1)
    lum0 = [gdata.sh_view_colour(base, d[None])[0].sum() for d in dirs]
    assert np.argmax(lum0) in (0, 2) and np.argmax(lum) in (1, 3)  # the bright side turned by a quarter
    assert lum[(np.argmax(lum0) + 1) % 4] == max(lum)


def test_ground_repair_from_a_template(tmp_path, monkeypatch):
    """A template circle decides where the fill copies from: a hole in the green west filled from a template in the
    blue east is blue; without the template it stays green (similar ground nearby)."""
    monkeypatch.setattr(replace, "BLEND", 0.0)  # where the fill copies from, by its colour: no blending / feather
    monkeypatch.setattr(replace, "FEATHER_M", 0.0)
    project, kind, n_ground, _ = _scene(tmp_path)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    east = (g["means"][:, 0] > 5) & (np.arange(len(g["means"])) < n_ground)
    g["sh"][east, 0] = (np.array([0.1, 0.2, 0.8]) - 0.5) / 0.28209479177387814  # blue
    difix.write_splat(None, splat_mesh.splat_path(project), g)
    replace.find(project, kind=kind, on_line=lambda m: None)

    def blue_share(source):
        replace.undo(project, "all", on_line=lambda m: None)
        replace.add_many(project, [{"circle": [-10.0, 10.0, 2.0], **({"source": source} if source else {})}],
                         on_line=lambda m: None)
        out = splat_mesh.load_splat(splat_mesh.splat_path(project))
        inside = np.hypot(out["means"][:, 0] + 10, out["means"][:, 1] - 10) < 1.8
        rgb = out["sh"][inside, 0] * 0.28209479177387814 + 0.5
        return float((rgb[:, 2] > rgb[:, 1]).mean())
    assert blue_share(None) < 0.1
    assert blue_share([12.0, -12.0, 4.0]) > 0.9
    assert replace.edits(project)[0]["source"] == [12.0, -12.0, 4.0]


def test_automatic_template_matches_the_area_not_its_rim(tmp_path, monkeypatch):
    """Copy from "the most similar detailed patch": a weak blue spot inside the green west (blurred: one uniform
    blue) gets its template in the blue east and comes out blue; a green spot gets a green template."""
    monkeypatch.setattr(replace, "BLEND", 0.0)  # where the fill copies from, by its colour: no blending / feather
    monkeypatch.setattr(replace, "FEATHER_M", 0.0)
    project, kind, n_ground, _ = _scene(tmp_path)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    ground = np.arange(len(g["means"])) < n_ground
    P = g["means"]
    blue = ground & ((P[:, 0] > 5) | (np.hypot(P[:, 0] + 10, P[:, 1] - 10) < 2.5))  # the east + a spot in the west
    g["sh"][blue, 0] = (np.array([0.1, 0.2, 0.8]) - 0.5) / 0.28209479177387814
    difix.write_splat(None, splat_mesh.splat_path(project), g)
    replace.find(project, kind=kind, on_line=lambda m: None)
    replace.add_many(project, [{"circle": [-10.0, 10.0, 2.0], "source": "auto"},
                               {"circle": [-10.0, -12.0, 2.0], "source": "auto"}], on_line=lambda m: None)
    st = json.load(open(os.path.join(project, replace.DIR, "state.json")))["edits"]
    t_blue, t_green = st[0]["fill"]["template_at"], st[1]["fill"]["template_at"]
    assert t_blue[0] > 5 and t_green[0] < 5 and 3.0 <= t_blue[2] <= 8.0
    out = splat_mesh.load_splat(splat_mesh.splat_path(project))
    rgb = out["sh"][:, 0] * 0.28209479177387814 + 0.5
    spot = np.hypot(out["means"][:, 0] + 10, out["means"][:, 1] - 10) < 1.8
    other = np.hypot(out["means"][:, 0] + 10, out["means"][:, 1] + 12) < 1.8
    assert (rgb[spot, 2] > rgb[spot, 1]).mean() > 0.9 and (rgb[other, 2] > rgb[other, 1]).mean() < 0.1


def test_weak_areas_only_over_the_mesh(tmp_path):
    """Weak ground is proposed only over the measured mesh (the SfM mesh's trimmed part here): of two weak squares,
    the one beyond the mesh's edge is not listed; without a mesh both are."""
    project, kind, *_ = _scene(tmp_path)
    replace.find(project, kind=kind, on_line=lambda m: None)
    d = np.load(os.path.join(project, replace.DIR, "objects.npz"))
    lo = d["lo"]
    H, W = (np.array(d["crown"].shape) * float(d["cell"]) / replace.MAP_CELL).astype(int) + 1
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = lo[0] + (xx + 0.5) * replace.MAP_CELL, lo[1] + (yy + 0.5) * replace.MAP_CELL
    weak = ((np.abs(cx + 12) < 3) | (np.abs(cx - 12) < 3)) & (np.abs(cy + 15) < 3)  # away from both trees
    ratio = np.where(weak, 0.05, 0.9).astype(np.float32)
    np.savez(os.path.join(project, replace.DIR, "detail.npz"), ratio=ratio, weight=np.ones_like(ratio), lo=lo,
             cell=replace.MAP_CELL, splat=np.array(replace._key(replace.base(project))))
    assert sorted(round(a["x"]) for a in replace.ground_areas(project)) == [-12, 12]
    # a mesh over the west half only (two triangles per 1 m square)
    gx, gy = np.meshgrid(np.arange(-20, 1.0), np.arange(-20, 21.0))
    V = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    nx = gx.shape[1]
    f = []
    for r in range(gx.shape[0] - 1):
        for c in range(nx - 1):
            a = r * nx + c
            f += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    os.makedirs(os.path.join(project, "sfmmesh"))
    v = np.zeros(len(V), [(k, "f4") for k in "xyz"])
    for i, k in enumerate("xyz"):
        v[k] = V[:, i]
    fa = np.zeros(len(f), [("vertex_indices", "i4", (3,))])
    fa["vertex_indices"] = np.array(f)
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex"), plyfile.PlyElement.describe(fa, "face")]).write(
        os.path.join(project, "sfmmesh", "mesh_trimmed.ply"))
    assert [round(a["x"]) for a in replace.ground_areas(project)] == [-12]


def test_polygon_repair_changes_only_inside(tmp_path, monkeypatch):
    """A ground repair drawn as a polygon replaces the ground layer inside it only (copying from a template polygon:
    the blue east), everything outside stays."""
    monkeypatch.setattr(replace, "BLEND", 0.0)  # where the fill copies from, by its colour: no blending / feather
    monkeypatch.setattr(replace, "FEATHER_M", 0.0)
    project, kind, n_ground, _ = _scene(tmp_path)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    east = (g["means"][:, 0] > 5) & (np.arange(len(g["means"])) < n_ground)
    g["sh"][east, 0] = (np.array([0.1, 0.2, 0.8]) - 0.5) / 0.28209479177387814
    difix.write_splat(None, splat_mesh.splat_path(project), g)
    replace.find(project, kind=kind, on_line=lambda m: None)
    tri = [[-16.0, 8.0], [-8.0, 8.0], [-12.0, 16.0]]
    replace.add_many(project, [{"polygon": tri, "source": {"polygon": [[8, -18], [18, -18], [18, -8], [8, -8]]}}],
                     on_line=lambda m: None)
    out = splat_mesh.load_splat(splat_mesh.splat_path(project))
    P = out["means"]
    rgb = out["sh"][:, 0] * 0.28209479177387814 + 0.5
    blue = rgb[:, 2] > rgb[:, 1]
    inside = (P[:, 1] > 8.6) & (P[:, 1] < 15) & (np.abs(P[:, 0] + 12) < (16 - P[:, 1]) / 2 - 0.6)  # well inside
    outside = (np.hypot(P[:, 0] + 12, P[:, 1] - 11) < 9) & ((P[:, 1] < 7) | (P[:, 1] > 17))
    assert blue[inside].mean() > 0.9 and blue[outside].mean() < 0.02
    st = json.load(open(os.path.join(project, replace.DIR, "state.json")))["edits"][0]["fill"]
    assert st["template_polygon"][0] == [8, -18]


def test_template_matches_photo_texture_not_distance(tmp_path):
    """The automatic template compares what the photos show (colour, texture at two scales) and the semantics, not the
    distance: a weak spot whose photos show coarse texture gets its template on the far coarse ground, although
    smooth ground of the same colour lies right next to it."""
    project, kind, *_ = _scene(tmp_path)
    replace.find(project, kind=kind, on_line=lambda m: None)
    d = np.load(os.path.join(project, replace.DIR, "objects.npz"))
    lo = d["lo"]
    H, W = (np.array(d["crown"].shape) * float(d["cell"]) / replace.MAP_CELL).astype(int) + 1
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = lo[0] + (xx + 0.5) * replace.MAP_CELL, lo[1] + (yy + 0.5) * replace.MAP_CELL
    coarse = (cx > 10) | (np.hypot(cx + 4, cy + 14) < 2.5)  # the far east + the weak spot (its photos)
    rgb = np.full((H, W, 3), 0.4, np.float32)
    tex = np.where(coarse[..., None], [2.0, 1.5], [-1.0, -2.0]).astype(np.float32)
    weak = np.hypot(cx + 4, cy + 14) < 2.5
    ratio = np.where(weak, 0.05, 0.9).astype(np.float32)
    np.savez(os.path.join(project, replace.DIR, "detail.npz"), ratio=ratio, weight=np.ones_like(ratio), lo=lo,
             cell=replace.MAP_CELL, splat=np.array(replace._key(replace.base(project))), photo_rgb=rgb, photo_tex=tex)
    sc = replace._Scene(project)
    t = sc.suggest_template(sc.disc(-4.0, -14.0, 2.0), np.zeros_like(sc.crown, bool))
    assert t is not None and t[0] > 10  # the coarse ground, ~20 m away, not the smooth ground beside the spot


def _two_tone(tmp_path, monkeypatch, n_bright=None):
    """The scene with the east ground (x > 5) a third brighter than the west, its checker texture kept."""
    project, kind, n_ground, _ = _scene(tmp_path)
    g = splat_mesh.load_splat(splat_mesh.splat_path(project))
    east = (g["means"][:, 0] > 5) & (np.arange(len(g["means"])) < n_ground)
    rgb = g["sh"][east, 0] * 0.28209479177387814 + 0.5
    g["sh"][east, 0] = (np.clip(rgb + 0.25, 0, 1) - 0.5) / 0.28209479177387814
    difix.write_splat(None, splat_mesh.splat_path(project), g)
    replace.find(project, kind=kind, on_line=lambda m: None)
    return project


def _lum_profile(project, centre, r):
    """Mean luminance of the ground layer in rings around centre (0.5 m wide) out to r + 3 m; and the texture
    (std of luminance) inside r - 1 m."""
    out = splat_mesh.load_splat(splat_mesh.splat_path(project))
    P = out["means"]
    op = 1 / (1 + np.exp(-out["opacities"].reshape(-1)))
    lum = (out["sh"][:, 0] * 0.28209479177387814 + 0.5).mean(1)
    d = np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1])
    low = np.abs(P[:, 2]) < 0.3
    rings = np.arange(0, r + 3, 0.5)
    prof = [float(np.average(lum[low & (d >= a) & (d < a + 0.5)], weights=op[low & (d >= a) & (d < a + 0.5)]))
            for a in rings]
    inner = low & (d < r - 1)
    return np.array(prof), float(lum[inner].std())


def test_blending_removes_the_step_at_the_edge_and_keeps_the_texture(tmp_path, monkeypatch):
    """A hole in the dark west filled from the bright east: without blending a step of ~0.25 in luminance at the edge;
    with gradient-domain blending (+ feathering) the patch meets its surroundings without a step and keeps the
    template's checker texture."""
    project = _two_tone(tmp_path, monkeypatch)
    centre, r = (-10.0, 12.0), 3.0
    template = {"polygon": [[8, -18], [18, -18], [18, -8], [8, -8]]}

    def run(blend, feather):
        monkeypatch.setattr(replace, "BLEND", blend)
        monkeypatch.setattr(replace, "FEATHER_M", feather)
        replace.undo(project, "all", on_line=lambda m: None)
        replace.add_many(project, [{"circle": [centre[0], centre[1], r], "source": template}], on_line=lambda m: None)
        return _lum_profile(project, centre, r)
    (hard, tex_hard), (soft, tex_soft) = run(0.0, 0.0), run(1.0, 1.0)
    inside, outside = slice(0, 4), slice(int(r / 0.5) + 2, None)  # rings well inside / beyond the edge (+ band)
    step_hard = hard[inside].mean() - hard[outside].mean()
    step_soft = soft[inside].mean() - soft[outside].mean()
    assert step_hard > 0.15 and abs(step_soft) < 0.04
    assert np.abs(np.diff(soft)).max() < 0.5 * np.abs(np.diff(hard)).max()  # no jump between neighbouring rings
    assert tex_soft > 0.6 * tex_hard  # the copied texture stays


def test_feather_crossfades_opacity_across_the_band(tmp_path, monkeypatch):
    """With a 1.5 m feather the copied Gaussians' opacity falls off across the band beyond the hole and the original
    ground layer's rises; beyond the band nothing changes."""
    project = _two_tone(tmp_path, monkeypatch)
    monkeypatch.setattr(replace, "FEATHER_M", 1.5)
    before = splat_mesh.load_splat(splat_mesh.splat_path(project))
    replace.add_many(project, [{"circle": [-10.0, 12.0, 3.0]}], on_line=lambda m: None)
    after = splat_mesh.load_splat(splat_mesh.splat_path(project))
    d0 = np.hypot(before["means"][:, 0] + 10, before["means"][:, 1] - 12)
    far = d0 > 5.5
    keep = {tuple(np.round(p, 4)) for p in after["means"]}
    assert all(tuple(np.round(p, 4)) in keep for p in before["means"][far][::50])  # unchanged beyond the band
    d1 = np.hypot(after["means"][:, 0] + 10, after["means"][:, 1] - 12)
    op = 1 / (1 + np.exp(-after["opacities"].reshape(-1)))
    ground = np.abs(after["means"][:, 2]) < 0.3
    near_in = ground & (d1 > 3.1) & (d1 < 3.6)
    near_out = ground & (d1 > 4.0) & (d1 < 4.5)
    assert 0.05 < op[near_in].min() and op[near_in].max() < op[ground & (d1 < 2)].max() + 1e-6
    assert op[near_in].mean() < 0.95 and op[near_out].mean() < 0.95  # both halves faded, none at full strength

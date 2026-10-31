"""Level of detail for large splats (modelforge/splat_lod.py, gui/lod.py): a synthetic splat of discs tiling a ground
and a box, built into cells and merged levels; the cut for a camera; the View tab drawing cells."""
import os
import shutil
import time

import numpy as np
import pytest

from modelforge import splat_lod, splat_mesh
from modelforge.gui import data as gdata
from tests.test_splat_mesh import scene

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


def test_levels_keep_the_moments(tmp_path):
    p = str(tmp_path)
    scene(p)  # 57k discs, 5 cm apart
    idx = splat_lod.build(p, cell_max=32000)
    c, cov, rgb, op = gdata.load_splat(splat_mesh.splat_path(p), None)
    n = len(c)
    assert idx["gaussians"] == n and len(idx["cells"]) == 2 and splat_lod.index(p) is not None
    lv = [idx["levels"][str(k)] for k in range(len(idx["levels"]))]
    assert lv[0] == n and len(lv) >= 3
    for a, b in zip(lv, lv[1:]):
        assert 2.5 < a / b < 6  # ~1/4 per level
    seen = np.zeros(n, int)
    for cell in idx["cells"]:
        src, _ = splat_lod.cell_sources(p, 0, cell)
        seen[src] += 1
        c0 = splat_lod.load_cell(p, 0, cell)
        assert np.allclose(c0[0], c[src], atol=1e-6) and np.allclose(c0[3], op[src], atol=1 / 255)
        for k in range(1, len(cell["levels"])):
            ck, covk, rgbk, opk = splat_lod.load_cell(p, k, cell)
            _, parent = splat_lod.cell_sources(p, k, cell)
            assert len(parent) == len(src) and parent.max() == len(ck) - 1 and len(np.unique(parent)) == len(ck)
            # each merged Gaussian: the weighted mean of its children, wider than any of them, the same colour
            j = np.bincount(parent).argmax()
            ch = src[parent == j]
            assert np.linalg.norm(ck[j] - c[ch].mean(0)) < 0.5 * cell["levels"][k]["voxel"]
            assert np.trace(covk[j]) >= np.trace(cov[ch]).max() * 0.99
            assert np.allclose(rgbk[j], rgb[ch].mean(0), atol=0.02)
            # the covered area stays: opaque ground stays opaque
            assert np.median(opk) > 0.9
    assert (seen == 1).all()
    os.utime(splat_mesh.splat_path(p), (time.time() + 5, time.time() + 5))
    assert splat_lod.index(p) is None  # a new splat: built again


def test_choose_fine_near_coarse_far(tmp_path):
    p = str(tmp_path)
    scene(p)
    idx = splat_lod.build(p, cell_max=16000)
    cells = idx["cells"]
    near = int(np.argmin([splat_lod._box_distance(np.array([-5.0, -5, 1]), c["box"]) for c in cells]))
    cut = splat_lod.choose(idx, [-5.0, -5, 1], [0.6, 0.6, -0.5], fov_y=0.9, height_px=1080, budget=30_000)
    total = sum(cells[i]["levels"][k]["n"] for i, k in cut.items())
    assert total <= 30_000 and cut[near] == 0
    far = int(np.argmax([splat_lod._box_distance(np.array([-5.0, -5, 1]), c["box"]) for c in cells]))
    assert cut[far] > 0
    big = splat_lod.choose(idx, [-5.0, -5, 1], [0.6, 0.6, -0.5], fov_y=0.9, height_px=1080, budget=10 ** 9)
    assert sum(cells[i]["levels"][k]["n"] for i, k in big.items()) <= idx["gaussians"]
    tiny = splat_lod.choose(idx, [-5.0, -5, 1], None, budget=0)  # nothing fits: all coarsest
    assert all(k == len(cells[i]["levels"]) - 1 for i, k in tiny.items())
    high = splat_lod.choose(idx, [0, 0, 5000.0], [0, 0, -1.0], fov_y=0.9, height_px=1080, budget=10 ** 9)
    assert all(k == len(cells[i]["levels"]) - 1 for i, k in high.items())  # far away: coarse is enough


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_view_draws_cells(tmp_path, monkeypatch):
    """A splat larger than the budget: the View builds its hierarchy in the background, then draws cells; a camera
    close to one corner refines the cells there; the semantic search decorates merged Gaussians."""
    from types import SimpleNamespace
    from modelforge.gui import lod as glod
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path / "proj")
    os.makedirs(p)
    for name in ("poses.json", "images", "sparse"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))
    scene(p)  # writes splat/splat_enu.ply + transforms.json
    monkeypatch.setattr(splat_lod, "CELL_MAX", 16000)
    monkeypatch.setattr(glod, "SETTLE", 0.05)
    real_build = splat_lod.build
    monkeypatch.setattr(splat_lod, "build_in", lambda folder, splat=None: real_build(folder, cell_max=16000,
                                                                                  splat=splat))
    monkeypatch.setenv("MODELFORGE_INPROCESS", "1")
    app = App(p, port=free_port())
    try:
        app.g_lod_budget.value = 30_000
        app.g_splat.value = True
        app._draw_splat()
        t0 = time.time()
        while app.lod.idx is None and time.time() - t0 < 120:
            time.sleep(0.2)
        assert app.lod.idx is not None, app.g_dense_md.content
        assert app.lod.total() <= 30_000 and "level of detail" in app.g_dense_md.content
        t0 = time.time()
        while len(app.lod.drawn) < len(app.lod.idx["cells"]) and time.time() - t0 < 10:  # the first draw done
            time.sleep(0.1)
        with app.lod.lock:
            coarse, handles = dict(app.lod.drawn), dict(app.lod.handles)
        rng = np.random.default_rng(0)  # pack() = viser's own packing
        c, rgb, op = rng.normal(size=(5, 3)), rng.random((5, 3)), rng.random((5, 1))
        cov = np.tile(np.eye(3) * 0.01, (5, 1, 1))
        h = app.server.scene.add_gaussian_splats("/test_pack", centers=c, covariances=cov, rgbs=rgb, opacities=op)
        assert np.array_equal(h.buffer, glod.pack(c, cov, rgb, op))
        h.remove()
        far = SimpleNamespace(position=(0.0, 0.0, 1e4), look_at=(0.0, 0.0, 0.0), fov=0.9, image_height=1080)
        app.lod.on_camera(SimpleNamespace(camera=far))  # far away: every cell at its coarsest level
        coarsest = {i: len(c["levels"]) - 1 for i, c in enumerate(app.lod.idx["cells"])}
        t0 = time.time()
        while app.lod.drawn != coarsest and time.time() - t0 < 10:
            time.sleep(0.1)
        assert app.lod.drawn == coarsest
        box = np.array(app.lod.idx["cells"][0]["box"], float)  # inside cell 0: it gets the finest level
        mid = box.mean(0)
        cam = SimpleNamespace(position=tuple(mid), look_at=tuple(mid + [1.0, 0, -0.2]), fov=0.9, image_height=1080)
        app.lod.on_camera(SimpleNamespace(camera=cam))
        t0 = time.time()
        while app.lod.drawn.get(0) != 0 and time.time() - t0 < 10:
            time.sleep(0.1)
        assert app.lod.drawn[0] == 0 and app.lod.total() <= 30_000
        assert all(app.lod.handles[k] is v for k, v in handles.items())  # the same objects, buffers replaced
        app.semantic.rel = np.linspace(0, 1, idx_n := app.lod.idx["gaussians"])  # a fake search result
        app.semantic.g_near.value = False
        app.redraw_splat_decorated()
        assert len(app.lod.drawn) == len(app.lod.idx["cells"]) and idx_n > 0
        # Difix tab → Compare: the splat from before Difix (here: every other Gaussian), with its own level of detail
        import plyfile
        from modelforge import difix
        sp = os.path.join(p, "splat", "splat_enu.ply")
        g = plyfile.PlyData.read(sp)["vertex"].data
        plyfile.PlyData([plyfile.PlyElement.describe(np.ascontiguousarray(g[::2]), "vertex")]).write(os.path.join(p, difix.BACKUP))
        app._refresh_difix()
        assert app.g_difix_compare.visible and "after Difix" in app.g_difix_compare_md.content
        app._difix_compare(True)
        t0 = time.time()
        while (app.lod.idx is None or app.lod.idx["gaussians"] != len(g[::2])) and time.time() - t0 < 120:
            time.sleep(0.2)
            app._draw_splat()
        assert app.lod.splat == difix.BACKUP and app.lod.idx["gaussians"] == len(g[::2])
        assert "before Difix" in app.g_dense_md.content and "before Difix" in app.g_difix_compare_md.content
        assert os.path.isdir(splat_lod.lod_dir(p, difix.BACKUP)) and splat_lod.index(p) is not None  # both kept
        app._difix_compare(False)
        assert app.lod.splat is None and app.lod.idx["gaussians"] == idx_n
        app._difix_compare(True)
        difix.undo(p)  # the splat from before is the project's again, and so is its level of detail
        assert splat_lod.index(p)["gaussians"] == len(g[::2]) and not os.path.isdir(splat_lod.lod_dir(p, difix.BACKUP))
        app._refresh_difix()
        assert app._splat_variant is None and not app.g_difix_compare.visible
        app.g_splat_lod.value = False  # off: the subset / all as before
        app._draw_splat()
        assert app.lod.idx is None and app._splat_shown is not None
    finally:
        app.lod.stop()
        app.server.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_live_preview_through_the_lod(tmp_path, monkeypatch):
    """Training snapshots larger than the budget: the first shows a random subset of the budget, then its level of
    detail; the next snapshot's is built in the other folder and swapped in place (the same cell objects); the end of
    the run removes both."""
    import plyfile
    from modelforge import splat_tiles
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path / "proj")
    os.makedirs(p)
    for name in ("poses.json", "images", "sparse"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))
    scene(p)
    v = plyfile.PlyData.read(os.path.join(p, "splat", "splat_enu.ply"))["vertex"].data
    os.remove(os.path.join(p, "splat", "splat_enu.ply"))  # training: no finished splat yet
    monkeypatch.setenv("MODELFORGE_INPROCESS", "1")
    app = App(p, port=free_port())
    try:
        app.g_lod_budget.value = 30_000
        app._splat_live_on = True
        app.g_splat.disabled, app.g_splat.value = False, True
        npz, ident = splat_tiles._write_live(p, [v], 500)
        app._on_splat_snapshot(500, npz, ident)
        assert app._live_splat is not None and len(app._live_splat[0]) == 30_000  # the subset meanwhile
        t0 = time.time()
        while getattr(app, "_live_lod_idx", None) is None and time.time() - t0 < 120:
            time.sleep(0.2)
        idx = app._live_lod_idx
        assert idx is not None and idx["step"] == 500 and idx["gaussians"] == len(v)
        assert app.lod.splat == "lod:" + splat_lod.LIVE_DIRS[0] and app.lod.idx is idx and app._live_splat is None
        t0 = time.time()
        while len(app.lod.drawn) < len(idx["cells"]) and time.time() - t0 < 10:
            time.sleep(0.1)
        assert 0 < app.lod.total() <= 30_000 and "live, step 500" in app.g_dense_md.content
        handles = dict(app.lod.handles)
        npz, ident = splat_tiles._write_live(p, [v[::2]], 1000)
        app._on_splat_snapshot(1000, npz, ident)
        assert app.lod.idx is idx  # the old one stays until the new one is built
        t0 = time.time()
        while (app._live_lod_idx or {}).get("step") != 1000 and time.time() - t0 < 120:
            time.sleep(0.2)
        assert app._live_lod_idx["gaussians"] == len(v[::2]) and app.lod.splat == "lod:" + splat_lod.LIVE_DIRS[1]
        assert all(app.lod.handles[k] is h for k, h in handles.items() if k in app.lod.handles)
        # the race the user hit: a snapshot's subset loading (seconds) while the level of detail becomes ready
        ready, real = app._live_lod_idx, gdata.load_live_splat

        def slow(*args, **kw):
            out = real(*args, **kw)
            app._live_lod_idx = ready  # ready meanwhile
            return out
        monkeypatch.setattr(gdata, "load_live_splat", slow)
        monkeypatch.setattr(app, "_adopt_live_lod", lambda: False)  # nothing on disk: the subset path
        app._live_lod_idx = None
        app._on_splat_snapshot(1500, npz, ident)
        assert app._live_splat is None and app.lod.idx is ready  # no subset drawn over it
        monkeypatch.setattr(gdata, "load_live_splat", real)
        monkeypatch.delattr(app, "_adopt_live_lod")
        t0 = time.time()
        while getattr(app, "_live_building", False) and time.time() - t0 < 120:
            time.sleep(0.2)
        # the GUI restarted while paused: nothing in memory, the paused run's level of detail on disk is shown at once
        newest = app._live_lod_idx
        app._live_lod_idx, app._live_splat = None, None
        app.lod.hide()
        app._on_splat_snapshot(2000, npz, ident)
        assert app._live_splat is None and app.lod.idx is not None and app.lod.idx["step"] == newest["step"]
        t0 = time.time()
        while getattr(app, "_live_building", False) and time.time() - t0 < 120:
            time.sleep(0.2)
        app._splat_live_on = False
        app._end_live_lod()
        t0 = time.time()
        while any(os.path.exists(os.path.join(p, d)) for d in splat_lod.LIVE_DIRS) and time.time() - t0 < 10:
            time.sleep(0.1)
        assert app.lod.idx is None and app.lod.splat is None
        assert not any(os.path.exists(os.path.join(p, d)) for d in splat_lod.LIVE_DIRS)
    finally:
        app.lod.stop()
        app.server.stop()

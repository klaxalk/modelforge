"""Semantic search: scoring of PCA-reduced per-Gaussian features against text embeddings, the sidecar's
staleness, and the View tab panel (synthetic features; the C-RADIOv3 run needs the GPU and the models)."""

import json
import os

import numpy as np

from modelforge import semantic


def write_features(project, n=500, D=32, dims=8, seed=0):
    """A synthetic features.npz: Gaussians 0-99 carry direction e0 ("car"), the rest e1 ("road")."""
    rng = np.random.default_rng(seed)
    full = np.tile(np.eye(D)[1], (n, 1)) + 0.05 * rng.normal(size=(n, D))
    full[:100] = np.eye(D)[0] + 0.05 * rng.normal(size=(100, D))
    mean = full.mean(0)
    _, _, vt = np.linalg.svd(full - mean, full_matrices=False)
    basis = vt[:dims].T
    feats = (full - mean) @ basis
    weight = np.ones(n, np.float32)
    weight[-10:] = 0  # never seen
    os.makedirs(os.path.join(project, semantic.DIR), exist_ok=True)
    np.savez(os.path.join(project, semantic.DIR, "features.npz"), feats=feats.astype(np.float16), weight=weight,
             pca_mean=mean.astype(np.float32), pca_basis=basis.astype(np.float32), explained=np.float32(0.99))
    return np.eye(D)


def test_scores_find_the_right_gaussians(tmp_path):
    E = write_features(str(tmp_path))
    f = semantic.Features(str(tmp_path))
    c = f.cosine(np.stack([E[0], E[1]]))
    assert c.shape == (500, 2) and np.all(c[:100, 0] > 0.9) and np.all(c[100:490, 1] > 0.9)
    assert np.all(c[-10:] == -1)  # never seen
    negs = np.stack([E[2], E[3]])  # generic words: unrelated directions here
    r = f.relevance(E[0], negs)
    assert np.all(r[:100] > 0.99) and np.median(r[100:490]) < 0.6 and r[100:490].max() < 0.9
    assert np.all(r[-10:] == 0)


def test_status_follows_the_splat(tmp_path):
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "splat"))
    ply = os.path.join(p, "splat", "splat_enu.ply")
    open(ply, "w").write("x")
    assert semantic.status(p) == "none"
    write_features(p)
    json.dump({"splat": semantic.fingerprint(p)}, open(os.path.join(p, semantic.DIR, "meta.json"), "w"))
    assert semantic.status(p) == "ready"
    open(ply, "w").write("a new splat")
    assert semantic.status(p) == "stale"


def test_panel_decorates_the_drawn_splat(tmp_path):
    from modelforge.gui.semantic import RED, SemanticPanel

    class App:
        path, dense = str(tmp_path), {"splat": None}
    panel = SemanticPanel.__new__(SemanticPanel)  # no viser server: only decorate()
    panel.app = App()

    class V:
        def __init__(self, v):
            self.value = v
    panel.g_thr, panel.g_mode, panel.g_near, panel.surface = V(0.6), V("highlight in red"), V(False), None
    n = 6
    centers, cov, rgb, op = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.full((n, 3), 0.5, np.float32), np.ones((n, 1))
    idx = np.arange(n)
    panel.rel = None
    assert panel.decorate(centers, cov, rgb, op, idx)[2] is rgb  # no search: unchanged
    panel.rel = np.array([0.95, 0.7, 0.55, 0.2, 0.61, 0.9])
    out = panel.decorate(centers, cov, rgb, op, idx)[2]
    assert np.allclose(out[[2, 3]], 0.5) and np.all(out[[0, 1, 4, 5], 0] > 0.6)  # matches tinted towards red
    assert np.allclose(out[0], RED, atol=0.05)  # the strongest is fully red
    panel.g_mode = V("matches only")
    c2 = panel.decorate(centers, cov, rgb, op, idx)[0]
    assert len(c2) == 4
    sub = np.array([1, 3, 5])  # a subset of the file's Gaussians is drawn: scores follow the indices
    panel.g_mode = V("highlight in red")
    assert len(panel.decorate(centers[:3], cov[:3], rgb[:3], op[:3], sub)[2]) == 3


def test_objects_are_separate_instances():
    """Two blobs of matching Gaussians far apart + scattered matches = two objects, ranked by strength."""
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(0, 100, (5000, 2)), np.zeros(5000)]
    car1 = rng.normal((20, 20, 0.8), (1.5, 0.8, 0.5), (400, 3))
    car2 = rng.normal((70, 60, 0.8), (1.5, 0.8, 0.5), (200, 3))
    means = np.vstack([ground, car1, car2])
    rel = np.r_[rng.uniform(0.3, 0.55, 5000), np.full(400, 0.9), np.full(200, 0.8)]
    rel[rng.choice(5000, 20, replace=False)] = 0.7  # scattered single matches: below min_gaussians
    obs = semantic.objects(means, rel, 0.6)
    assert len(obs) == 2
    assert np.allclose(obs[0]["center"][:2], (20, 20), atol=0.5) and obs[0]["count"] > 370  # a few far outliers detach
    assert np.allclose(obs[1]["center"][:2], (70, 60), atol=0.5)
    assert 1.5 < obs[0]["radius"] < 4.5 and obs[0]["z0"] < 0.8 < obs[0]["z1"]
    labels = semantic._components(np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 5, 5], [9, 0, 0]]))
    assert labels[0] == labels[1] and labels[2] == labels[3] and len(set(labels)) == 3


def test_cylinder_and_best_view():
    from modelforge.gui import data as gdata
    from modelforge.gui.semantic import best_view, cylinder_segments
    seg = cylinder_segments(2.0, 3.0, n=24, struts=6)
    assert seg.shape == (24 * 2 + 6, 2, 3)
    assert np.allclose(np.linalg.norm(seg[:48, :, :2], axis=2), 2.0, atol=1e-5)  # the rings
    assert set(np.round(seg[:, :, 2].ravel(), 5)) == {0.0, 3.0}

    class Snap:
        cams = gdata.Cameras.empty(3)
    down = np.diag([1.0, -1.0, -1.0])  # OpenCV camera looking straight down
    for i, c in enumerate([(0, 0, 30), (50, 0, 30), (3, 0, 10)]):
        Snap.cams.centers[i], Snap.cams.R_wc[i] = c, down
    target = np.array([2.0, 0.0, 0.0])
    pos = best_view(Snap, target, np.radians(60), 16 / 9, radius=1.0)
    assert np.allclose(pos, (3, 0, 10))  # the near camera that has it in view (not the one 50 m away)
    pos = best_view(Snap, target, np.radians(60), 16 / 9, radius=8.0)  # a big object: pulled back to 20 m
    assert np.isclose(np.linalg.norm(pos - target), 20.0)
    assert best_view(Snap, np.array([500.0, 500.0, 0.0]), np.radians(60), 16 / 9, 1.0) is None


def test_surface_distance_and_near_mesh_filter(tmp_path):
    """Gaussians off the mesh surface are measured (cached per splat + mesh) and dropped from the matches."""
    import plyfile
    from modelforge.gui.semantic import SemanticPanel
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "splat"))
    os.makedirs(os.path.join(p, "splatmesh"))
    g = np.linspace(0, 10, 41)  # a flat mesh grid z = 0, 0.25 m apart
    X, Y = np.meshgrid(g, g)
    verts = np.zeros(X.size, [("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"], verts["y"] = X.ravel(), Y.ravel()
    faces = np.zeros(2, [("vertex_indices", "i4", (3,))])
    faces["vertex_indices"] = [[0, 1, 41], [1, 42, 41]]
    plyfile.PlyData([plyfile.PlyElement.describe(verts, "vertex"), plyfile.PlyElement.describe(faces, "face")]) \
        .write(os.path.join(p, "splatmesh", "mesh.ply"))
    json.dump({"voxel_m": 0.25}, open(os.path.join(p, "splatmesh", "stats.json"), "w"))
    sv = np.zeros(4, [("x", "f4"), ("y", "f4"), ("z", "f4")])
    sv["x"], sv["y"], sv["z"] = [5, 5, 5, 5], [5, 5, 5, 5], [0.0, 0.5, 2.0, 30.0]  # on, near, off, a floater
    plyfile.PlyData([plyfile.PlyElement.describe(sv, "vertex")]).write(os.path.join(p, "splat", "splat_enu.ply"))
    assert semantic.search_mesh(p)[1] == 0.25
    d, voxel, _ = semantic.surface_distance(p)
    assert np.allclose(d, [0, 0.5, 2.0, 30.0], atol=0.05) and voxel == 0.25  # FLANN is approximate
    assert os.path.exists(os.path.join(p, semantic.DIR, "surface.npz"))
    assert np.allclose(semantic.surface_distance(p)[0], d, atol=0.01)  # from the cache (float16)

    class V:
        def __init__(self, v):
            self.value = v

    class App:
        path = p
    panel = SemanticPanel.__new__(SemanticPanel)
    panel.app, panel.surface = App(), None
    panel.rel = np.array([0.9, 0.9, 0.9, 0.9])
    panel.g_thr, panel.g_near, panel.g_tol = V(0.6), V(True), V(3.0)  # 3 voxels = 0.75 m
    rel, dropped = panel._rel()
    assert list(rel) == [0.9, 0.9, 0.0, 0.0] and dropped == 2
    panel.g_near = V(False)
    assert list(panel._rel()[0]) == [0.9] * 4


def test_auto_threshold():
    """Lowered from the best match: stops before noise specks appear around a single object, before two
    distinct objects merge, and reports an absent query."""
    rng = np.random.default_rng(1)
    ground = np.c_[rng.uniform(0, 100, (20000, 2)), np.zeros(20000)]
    obj = rng.normal((50, 50, 1.0), (1.5, 1.5, 0.6), (1500, 3))
    specks = np.vstack([rng.normal((x, y, 0.2), 0.3, (40, 3)) for x, y in rng.uniform(5, 95, (12, 2))])
    means = np.vstack([ground, obj, specks])  # 12 small noise clusters (specks) of 40 Gaussians at 0.62
    rel = np.r_[rng.uniform(0.3, 0.58, 20000), rng.uniform(0.66, 0.75, 1500), np.full(len(specks), 0.62)]
    thr, why = semantic.auto_threshold(means, rel)
    assert why == "specks" and 0.62 < thr <= 0.66, (thr, why)
    obs = semantic.objects(means, rel, thr)
    assert len(obs) == 1 and np.allclose(obs[0]["center"][:2], (50, 50), atol=0.5)
    assert semantic.auto_threshold(means, np.minimum(rel, 0.55)) == (0.55, "absent")


def test_text_model_unloads_for_jobs_and_loads_on_typing(monkeypatch):
    """A job stops the text worker; typing in Query starts it again (in the background)."""
    from modelforge.gui.semantic import SemanticPanel

    class V:
        def __init__(self, v):
            self.value, self.content = v, ""

    class Proc:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else 0
    started = []
    panel = SemanticPanel.__new__(SemanticPanel)
    panel.encoder = semantic.TextEncoder()
    panel.encoder.proc = Proc()
    panel.encoder.close = lambda: setattr(panel.encoder.proc, "alive", False)
    panel.g_text_md, panel.g_query = V(""), V("a ca")
    panel.preload = lambda: started.append(1)
    panel.search = lambda client=None: None
    panel.unload("splat")
    assert panel.encoder.proc is None and "unloaded for splat" in panel.g_text_md.content
    panel._query_typed(None)
    assert started == [1]

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
    panel.g_thr, panel.g_mode = V(0.6), V("highlight in red")
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

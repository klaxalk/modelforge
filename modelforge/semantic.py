"""Semantic search over a splat: C-RADIOv3 features per Gaussian (a sidecar of splat/splat_enu.ply) and text
queries scored against them (View tab → Semantic search).

compute (`run`): export/semantic_features.py in nerfstudio's venv back-projects RADIO's text-aligned patch
features from every photo onto the Gaussians (visibility-checked; option A, no training) and stores them
PCA-reduced in PROJECT/splat/semantic/features.npz with a fingerprint of the splat file. The splat itself
and its export are unchanged; a new or resumed splat makes the features stale (`status`).

query: TextEncoder keeps a worker (export/semantic_text.py) with the SigLIP2 text model loaded; `scores`
maps a text embedding onto every Gaussian: cosine in the full space (the reduced features are mapped back
through the PCA basis), and a relevance against generic phrases (LERF): for each negative n,
sigmoid(T (cos_q - cos_n)), the minimum over the negatives.

RADIO runs with extra packages (einops, timm >= 1.0, transformers) that nerfstudio's venv does not have; they
sit in their own folder put first on PYTHONPATH only for these subprocesses (`RADIO_PKGS`). Model weights are
cached in MODEL_CACHE (Hugging Face + torch hub; C-RADIOv3-L ~1 GB, SigLIP2-so400m ~4.3 GB).
"""

import json
import os
import subprocess
import threading

import numpy as np

from modelforge import jobs, splat, splat_mesh

DIR = os.path.join("splat", "semantic")
VERSION = "c-radio_v3-l"
ADAPTOR = "siglip2"
WIDTH = 2048  # RADIO input width: fine detail (128 x 96 patches for 4:3 photos)
DIMS = 64
NEGATIVES = ("object", "things", "stuff", "texture")
TEMPERATURE = 10.0
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADIO_PKGS = next((p for p in (os.environ.get("MODELFORGE_RADIO_PKGS"), "/opt/radio-pkgs",
                               os.path.join(_REPO, "build", "radio_pkgs")) if p and os.path.isdir(p)), None)
MODEL_CACHE = os.environ.get("MODELFORGE_MODEL_CACHE", os.path.join(_REPO, "build"))


def available():
    """None if semantic features can be computed here, else the reason."""
    why = splat.available()
    if why:
        return why
    if RADIO_PKGS is None:
        return ("the RADIO packages are missing (einops, timm >= 1.0, transformers in /opt/radio-pkgs or "
                "build/radio_pkgs; see modelforge/semantic.py)")
    return None


def _env():
    env = splat._env()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (RADIO_PKGS, env.get("PYTHONPATH")) if p)
    env["HF_HOME"] = os.path.join(MODEL_CACHE, "hf_cache")
    env["TORCH_HOME"] = os.path.join(MODEL_CACHE, "torch_hub")
    return env


def fingerprint(project):
    """Size + mtime of the splat file the features belong to (None without a splat)."""
    p = splat_mesh.splat_path(project)
    if not p or not os.path.exists(p):
        return None
    st = os.stat(p)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def status(project):
    """"none" (no features), "stale" (the splat changed since) or "ready"."""
    meta = os.path.join(project, DIR, "meta.json")
    if not os.path.exists(os.path.join(project, DIR, "features.npz")) or not os.path.exists(meta):
        return "none"
    return "ready" if json.load(open(meta)).get("splat") == fingerprint(project) else "stale"


def _cameras(project):
    """World -> OpenCV view matrices of the transforms.json frames, K, distortion, size, image paths."""
    t = json.load(open(os.path.join(project, "transforms.json")))
    if t.get("camera_model", "OPENCV") != "OPENCV":
        raise ValueError(f"semantic features support OPENCV cameras, not {t.get('camera_model')}")
    K = np.array([[t["fl_x"], 0, t["cx"]], [0, t["fl_y"], t["cy"]], [0, 0, 1]], float)
    dist = np.array([t.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")], float)
    views, paths = [], []
    for f in t["frames"]:
        c2w = np.array(f["transform_matrix"], float)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
        views.append(np.linalg.inv(c2w))
        paths.append(os.path.join(project, f["file_path"]))
    return np.array(views, np.float32), K, dist, int(t["w"]), int(t["h"]), paths


def run(project, width=WIDTH, dims=DIMS, version=VERSION, on_line=print, on_progress=None, job=None):
    """Compute the semantic features of the project's splat. on_progress(phase, done, total)."""
    import re
    import time
    why = available()
    if why:
        raise RuntimeError(why)
    sp = splat_mesh.splat_path(project)
    if not sp or not os.path.exists(sp):
        raise RuntimeError("no splat yet (Splat tab)")
    job = job or jobs.Token()
    t0 = time.time()
    out_dir = os.path.join(project, DIR)
    os.makedirs(out_dir, exist_ok=True)
    fp = fingerprint(project)
    views, K, dist, w, h, paths = _cameras(project)
    inp = os.path.join(out_dir, "input.npz")
    np.savez(inp, viewmats=views, K=K, dist=dist, width=w, height=h, paths=np.array(paths),
             **splat_mesh.load_splat(sp))
    params = os.path.join(out_dir, "params.json")
    json.dump({"version": version, "adaptor": ADAPTOR, "width": int(width), "dims": int(dims), "pca_images": 24,
               "depth_div": 4, "depth_tol": 0.03}, open(params, "w"))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "semantic_features.py")
    prog = re.compile(r"\[semantic\] (pca|image) (\d+) / (\d+)")

    def line(text):
        on_line(text)
        m = prog.search(text)
        if m and on_progress:
            on_progress(m.group(1), int(m.group(2)), int(m.group(3)))
    try:
        job.run([os.path.join(splat.NS_BIN, "python"), script, inp, out_dir, params], line, env=_env())
    finally:
        for f in (inp, params):
            if os.path.exists(f):
                os.remove(f)
    stats = json.load(open(os.path.join(out_dir, "stats.json")))
    meta = {**stats, "splat": fp, "negatives": list(NEGATIVES), "seconds_total": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    from modelforge.pipeline import Report
    Report(project).set("semantic", meta)
    return meta


class Features:
    """features.npz of a project, ready for scoring: reduced features, the PCA basis and per-Gaussian
    norms of the reconstructed (full-space) features."""

    def __init__(self, project):
        d = np.load(os.path.join(project, DIR, "features.npz"))
        self.f = d["feats"].astype(np.float32)  # (N, dims)
        self.seen = d["weight"] > 0
        self.mean = d["pca_mean"].astype(np.float32)  # (D,)
        self.basis = d["pca_basis"].astype(np.float32)  # (D, dims)
        rec_sq = (self.mean @ self.mean) + 2 * self.f @ (self.basis.T @ self.mean) + np.einsum(
            "nd,de,ne->n", self.f, self.basis.T @ self.basis, self.f)
        self.norm = np.sqrt(np.maximum(rec_sq, 1e-12)).astype(np.float32)

    def cosine(self, emb):
        """(N, Q) cosine of every Gaussian's (reconstructed) feature with unit text embeddings (Q, D)."""
        emb = np.atleast_2d(np.asarray(emb, np.float32))
        dots = self.mean @ emb.T + self.f @ (self.basis.T @ emb.T)  # (N, Q)
        c = dots / self.norm[:, None]
        c[~self.seen] = -1.0
        return c

    def relevance(self, q_emb, neg_embs, temperature=TEMPERATURE):
        """LERF-style relevance in [0, 1] per Gaussian: min over negatives of sigmoid(T (cos_q - cos_neg))."""
        c = self.cosine(np.vstack([q_emb, neg_embs]))
        r = 1.0 / (1.0 + np.exp(-temperature * (c[:, :1] - c[:, 1:])))
        out = r.min(1)
        out[~self.seen] = 0.0
        return out


class TextEncoder:
    """A long-lived worker with the text model (load once: ~1 min; then a query takes milliseconds)."""

    def __init__(self, version=VERSION):
        self.version, self.proc, self.lock = version, None, threading.Lock()

    def _start(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "semantic_text.py")
        self.proc = subprocess.Popen([os.path.join(splat.NS_BIN, "python"), script, self.version, ADAPTOR],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, env=_env())
        for line in self.proc.stdout:
            if line.strip() == "ready":
                return
        raise RuntimeError("the text encoder did not start (see modelforge/export/semantic_text.py)")

    def encode(self, texts):
        """(len(texts), D) unit embeddings."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps(list(texts)) + "\n")
            self.proc.stdin.flush()
            return np.array(json.loads(self.proc.stdout.readline()), np.float32)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

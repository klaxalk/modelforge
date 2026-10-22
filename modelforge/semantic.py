"""Semantic search over a splat: C-RADIOv3 features per Gaussian (a sidecar of splat/splat_enu.ply) and text
queries scored against them (the Search tab on the left).

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


def _work(job, arrays, out_dir, params, line):
    """One run of export/semantic_features.py on these input arrays (npz) and params."""
    os.makedirs(out_dir, exist_ok=True)
    inp, pfile = os.path.join(out_dir, "input.npz"), os.path.join(out_dir, "params.json")
    np.savez(inp, **arrays)
    json.dump(params, open(pfile, "w"))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "semantic_features.py")
    try:
        job.run([os.path.join(splat.NS_BIN, "python"), script, inp, out_dir, pfile], line, env=_env())
    finally:
        for f in (inp, pfile):
            if os.path.exists(f):
                os.remove(f)


def _liner(on_line, on_progress, label=None):
    """Log lines of the worker -> on_line, and its "[semantic] pca|image i / n" -> on_progress(phase, i, n);
    label replaces the phase "image" (a tile's photos)."""
    import re
    prog = re.compile(r"\[semantic\] (pca|image) (\d+) / (\d+)")

    def line(text):
        on_line(text)
        m = prog.search(text)
        if m and on_progress:
            ph = m.group(1)
            on_progress(label if label and ph == "image" else ph, int(m.group(2)), int(m.group(3)))
    return line


def _camera_arrays(project):
    views, K, dist, w, h, paths = _cameras(project)
    return dict(viewmats=views, K=K, dist=dist, width=w, height=h, paths=np.array(paths))


def run(project, width=WIDTH, dims=DIMS, version=VERSION, on_line=print, on_progress=None, job=None):
    """Compute the semantic features of the project's splat. on_progress(phase, done, total).
    A merged tiled splat (splat_tiles.merged_tiles) is done per tile, each from its own photos, and the tiles'
    features merged; a tile whose cleaned splat did not change keeps its features (e.g. after training one tile
    again)."""
    import time
    from modelforge import splat_tiles
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
    params = {"version": version, "adaptor": ADAPTOR, "width": int(width), "dims": int(dims), "pca_images": 24,
              "depth_div": 4, "depth_tol": 0.03}
    tiles = splat_tiles.merged_tiles(project)
    if tiles:
        stats = _run_tiles(project, tiles, params, on_line, on_progress, job)
    else:
        _work(job, dict(_camera_arrays(project), **splat_mesh.load_splat(sp)), out_dir, params,
              _liner(on_line, on_progress))
        stats = json.load(open(os.path.join(out_dir, "stats.json")))
    meta = {**stats, "splat": fp, "negatives": list(NEGATIVES), "seconds_total": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    from modelforge.pipeline import Report
    Report(project).set("semantic", meta)
    return meta


def _run_tiles(project, tiles, params, on_line, on_progress, job):
    """Features of a merged tiled splat: one PCA basis over the project's photos (splat/semantic/pca.npz), then per
    tile its whole splat rendered for the occlusion test from the tile's photos, features kept for its cleaned
    part (tiles/tile_KK/semantic/), concatenated in merge order -> splat/semantic/features.npz."""
    import time
    from modelforge import splat_plan, splat_tiles
    t0 = time.time()
    out_dir = os.path.join(project, DIR)
    key = {k: params[k] for k in ("version", "adaptor", "width", "dims", "pca_images")}
    pca, pca_key = os.path.join(out_dir, "pca.npz"), os.path.join(out_dir, "pca.json")
    if not os.path.exists(pca) or not os.path.exists(pca_key) or json.load(open(pca_key)) != key:
        on_line(f"semantic: PCA basis from {params['pca_images']} of the project's photos (shared by the tiles)")
        _work(job, _camera_arrays(project), out_dir, dict(params, pca_out=pca), _liner(on_line, on_progress))
        json.dump(key, open(pca_key, "w"))
    pca_mtime = os.stat(pca).st_mtime_ns
    state = splat_tiles.load(project)
    feats, weight, photos, size = [], [], 0, None
    for j, k in enumerate(tiles):
        tile, tdir = state["tiles"][k], splat_tiles.tile_dir(project, k)
        tout = os.path.join(tdir, "semantic")
        clean = os.path.join(tdir, "splat", "clean_enu.ply")
        st = os.stat(clean)
        tkey = dict(key, clean=[st.st_size, st.st_mtime_ns], pca=pca_mtime)
        kfile = os.path.join(tout, "key.json")
        if os.path.exists(os.path.join(tout, "features.npz")) and os.path.exists(kfile) and \
                json.load(open(kfile)) == tkey:
            on_line(f"semantic: tile {k + 1}: features unchanged")
        else:
            jobs.check(job)
            full = os.path.join(tdir, "splat", "splat_enu.ply")
            g, keep = None, None
            if os.path.exists(full):  # the whole tile occludes, the cleaned part gets features
                v = splat_tiles._read(full)
                m = splat_tiles.clean(v, splat_tiles._cell(tile["cell"]), tile["margin"], state["box"])
                del v
                if int(m.sum()) == splat_plan.ply_count(clean):
                    g, keep = splat_mesh.load_splat(full), np.flatnonzero(m)
            if g is None:  # the tile's splat changed since its clean: its cleaned part alone
                g = splat_mesh.load_splat(clean)
                keep = np.arange(len(g["means"]))
            on_line(f"semantic: tile {k + 1} ({j + 1} / {len(tiles)}): {len(tile['frames'])} photos, "
                    f"{len(keep):,} of {len(g['means']):,} gaussians")
            _work(job, dict(_camera_arrays(tdir), keep=keep, **g), tout, dict(params, pca_in=pca),
                  _liner(on_line, on_progress, f"tile {j + 1} / {len(tiles)}: photo"))
            del g
            json.dump(tkey, open(kfile, "w"))
        d = np.load(os.path.join(tout, "features.npz"))
        feats.append(d["feats"])
        weight.append(d["weight"])
        ts = json.load(open(os.path.join(tout, "stats.json")))
        photos += ts["photos"]
        size = (ts["width"], ts["height"])
    p = np.load(pca)
    feats, weight = np.concatenate(feats), np.concatenate(weight)
    tmp = os.path.join(out_dir, "features.part.npz")
    np.savez(tmp, feats=feats, weight=weight, pca_mean=p["pca_mean"], pca_basis=p["pca_basis"],
             explained=p["explained"])
    os.replace(tmp, os.path.join(out_dir, "features.npz"))
    stats = {"version": params["version"], "adaptor": params["adaptor"], "width": size[0], "height": size[1],
             "dims": params["dims"], "explained": round(float(p["explained"]), 4), "photos": photos,
             "gaussians": int(len(weight)), "seen": int((weight > 0).sum()), "tiles": len(tiles),
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    on_line(f"semantic: {len(tiles)} tiles merged: {stats['seen']:,} of {stats['gaussians']:,} gaussians seen")
    return stats


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


def search_mesh(project):
    """(path, voxel m) of the mesh that prunes search results: the mesh from the splat (its TSDF voxel), else
    COLMAP's Delaunay / Poisson mesh (voxel = median edge length); None without a mesh."""
    sm = os.path.join(project, "splatmesh", "mesh.ply")
    if os.path.exists(sm):
        try:
            voxel = float(json.load(open(os.path.join(project, "splatmesh", "stats.json")))["voxel_m"])
        except (OSError, ValueError, KeyError):
            voxel = None
        return sm, voxel or _edge_length(sm)
    for m in ("delaunay", "poisson"):
        p = os.path.join(project, "dense", f"meshed-{m}.ply")
        if os.path.exists(p):
            return p, _edge_length(p)
    return None


def _edge_length(path):
    from modelforge import sfm_mesh
    d, f = sfm_mesh.read_mesh(path)  # one block, not a Python object per face (24M faces on kladno_dira)
    v = np.stack([d[k] for k in ("x", "y", "z")], 1)
    f = f[:20000]
    return float(np.median(np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)))


class Surface:
    """Distances of Gaussian centres to the nearest vertex of the search mesh (`search_mesh`; the nearest vertex
    stands in for the surface, vertices are ~1 voxel apart), measured only for the Gaussians a search asks about
    (its matches) and remembered while the mesh and the splat stay the same. All Gaussians at once took minutes
    on kladno_dira (4.0M against 12.4M vertices: ~250 s on one core), a query's matches take seconds. The k-d tree
    (OpenCV FLANN, approximate) is built once (~15 s at 12M vertices); queries run on all cores."""

    def __init__(self, project, means):
        mesh = search_mesh(project)
        if mesh is None:
            raise FileNotFoundError("no mesh")
        self.path, self.voxel = mesh
        self.means = np.ascontiguousarray(means, np.float32)
        self.key = self._key(project)
        self.d = np.full(len(means), np.nan, np.float32)
        self._index, self._lock = None, threading.Lock()

    def _key(self, project):
        sp = splat_mesh.splat_path(project)
        return tuple((os.stat(f).st_size, os.stat(f).st_mtime_ns) for f in (self.path, sp) if f and os.path.exists(f))

    def current(self, project):
        mesh = search_mesh(project)
        return mesh is not None and mesh[0] == self.path and self._key(project) == self.key

    def index(self):
        import cv2
        if self._index is None:
            from modelforge import sfm_mesh
            mv = sfm_mesh.read_mesh(self.path)[0]  # the faces as one block: plyfile's per-face objects took minutes
            V = np.ascontiguousarray(np.stack([mv["x"], mv["y"], mv["z"]], 1), np.float32)
            self._index = cv2.flann_Index(V, dict(algorithm=1, trees=4))
        return self._index

    def distance(self, need):
        """The distances (N,), measured where `need` (bool per Gaussian) and not yet known; NaN elsewhere."""
        from concurrent.futures import ThreadPoolExecutor
        with self._lock:
            todo = np.flatnonzero(need & np.isnan(self.d))
            if len(todo):
                index = self.index()
                parts = np.array_split(todo, max(1, min(len(todo) // 20000, 4 * (os.cpu_count() or 1))))
                with ThreadPoolExecutor(os.cpu_count()) as ex:  # OpenCV releases the GIL in the search
                    for part, d2 in zip(parts, ex.map(
                            lambda k: index.knnSearch(self.means[k], 1, params=dict(checks=32))[1][:, 0], parts)):
                        self.d[part] = np.sqrt(d2)
        return self.d


def _components(keys):
    """Connected components (26-neighbourhood) of occupied voxels; keys (M, 3) int, unique. Labels (M,)."""
    M = len(keys)
    lab = np.arange(M)
    if M < 2:
        return lab
    lo = keys.min(0)
    dims = keys.max(0) - lo + 3
    lin = ((keys - lo + 1) * np.array([dims[1] * dims[2], dims[2], 1])).sum(1)
    order = np.argsort(lin)
    slin = lin[order]
    a_idx, b_idx = [], []
    for dx in (-1, 0, 1):  # half of the 26 neighbours: each edge once
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                nb = lin + dx * dims[1] * dims[2] + dy * dims[2] + dz
                pos = np.clip(np.searchsorted(slin, nb), 0, M - 1)
                hit = slin[pos] == nb
                a_idx.append(np.flatnonzero(hit))
                b_idx.append(order[pos[hit]])
    a_idx, b_idx = np.concatenate(a_idx), np.concatenate(b_idx)
    while True:  # min-label propagation along the edges until nothing changes
        m = np.minimum(lab[a_idx], lab[b_idx])
        new = lab.copy()
        np.minimum.at(new, a_idx, m)
        np.minimum.at(new, b_idx, m)
        new = new[new]  # pointer jumping
        if np.array_equal(new, lab):
            return lab
        lab = new


def objects(means, rel, thr, voxel=None, min_gaussians=30, max_objects=20):
    """Object instances of a query: the Gaussians with relevance >= thr, grouped by 26-connected voxels (a
    voxel of 1/300 of the scene's size unless given). Ranked by summed relevance above the threshold;
    groups under min_gaussians are dropped. Each: center (weighted by relevance), radius (95 % of the
    horizontal distances), z0 / z1 (2nd / 98th percentile height), count, score, mean relevance."""
    sel = np.flatnonzero(rel >= thr)
    if len(sel) < min_gaussians:
        return []
    if voxel is None:
        lo, hi = np.percentile(means, 2, axis=0), np.percentile(means, 98, axis=0)
        voxel = max(float(np.linalg.norm(hi - lo)) / 300.0, 1e-6)
    p = means[sel]
    keys, inv = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_inverse=True)
    comp = _components(keys)[inv.reshape(-1)]
    out = []
    for c in np.unique(comp):
        m = comp == c
        if m.sum() < min_gaussians:
            continue
        q, r = p[m], rel[sel][m]
        w = r - thr + 1e-6
        center = (q * w[:, None]).sum(0) / w.sum()
        radius = float(np.percentile(np.linalg.norm(q[:, :2] - center[:2], axis=1), 95))
        z0, z1 = np.percentile(q[:, 2], [2, 98])
        out.append({"center": center, "radius": max(radius, voxel), "z0": float(z0), "z1": float(max(z1, z0 + voxel)),
                    "count": int(m.sum()), "score": float(w.sum()), "relevance": float(r.mean())})
    out.sort(key=lambda o: -o["score"])
    return out[:max_objects]


ABSENT = 0.6  # best relevance below this: probably not in the scene
AUTO_FLOOR = 0.55  # the automatic threshold stays above this (0.5 = no better than the generic words)


def auto_threshold(means, rel, min_gaussians=30, step=0.01, floor=AUTO_FLOOR, specks=3):
    """Threshold for a query, from how its objects behave while it is lowered from the best match: the last
    value before (1) `specks` or more new groups barely above min_gaussians appear (noise sets in) or (2) the
    largest object at least 1.5x its size by absorbing another one (objects merge); else the floor. Specks count
    only while no new substantial object appears alongside (real instances emerge with some). Both only
    count once there is a substantial object (>= 10 x min_gaussians; a merge needs two): near the top an
    object still forms from fragments. Returns (threshold, reason); a best relevance below ABSENT returns
    (best, "absent"). kladno_dopravnik: "a machine" 0.66 (specks), "gravel pile" 0.62 (merge)."""
    top = float(rel.max()) if len(rel) else 0.0
    if top < ABSENT:
        return top, "absent"
    t = np.floor(top / step) * step
    prev, prev_t = None, None
    while t >= floor - 1e-9:
        obs = objects(means, rel, t, min_gaussians=min_gaussians, max_objects=100000)
        small = sum(1 for o in obs if o["count"] < 3 * min_gaussians)
        big = [o["count"] for o in obs]
        if prev is not None:
            p_small, p_big = prev
            solid = 10 * min_gaussians
            n_solid, p_solid = sum(1 for c in big if c >= solid), sum(1 for c in p_big if c >= solid)
            # noise: new specks while no new substantial object appears (new piles of gravel come with some)
            if p_solid and small - p_small >= specks and n_solid <= p_solid:
                return prev_t, "specks"
            # two distinct objects of comparable size join (a fragment joining its own object is not a merge)
            if len(p_big) > 1 and p_big[1] >= max(solid, 0.3 * p_big[0]) and big and big[0] >= 1.5 * p_big[0] and \
                    big[0] >= p_big[0] + 0.5 * p_big[1]:
                return prev_t, "merge"
        if obs:
            prev, prev_t = (small, big), t
        t = round(t - step, 6)
    return (prev_t if prev_t is not None else floor), "floor"


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
        """Stop the worker and wait for it to exit (its GPU and RAM are free afterwards)."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

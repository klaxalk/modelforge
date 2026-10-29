"""Learned features and matching for SfM, in place of COLMAP's SIFT (the user asked, 2026-10-28; docs/todo_matching.md).

On berlin_all's hard pairs (more than 60 s apart, overlapping by the model, 0 COLMAP SIFT inliers) GIM-LightGlue linked
41 of 64 and ALIKED + LightGlue 37, while RoMa, a dense matcher, linked 53, and was the only method that handled
large changes of scale. So:
1. features: a learned detector + descriptor per photo (METHODS) into features_learned/<method>/<photo>.npz; its
   keypoints go into features.db like SIFT's (images and cameras through pycolmap.import_images, so cameras, rigs
   and frames are COLMAP's own), its descriptors stay in the npz files. Those files are a cache per photo (valid
   while the photo's size + mtime are the ones stored): SfM on another choice of Images used and Register more
   photos (whose photos it also fills) extract only what is missing. A new choice of images keeps it (apply_subsample);
   a new extraction archives it.
2. match: LightGlue on every pair of pairs.txt, written into database.db's matches table, then COLMAP's geometric
   verification (pycolmap.verify_matches) as for SIFT.
3. dense (optional, DENSE): pairs between passes or flights (more than FAR_S apart, or two sources) that verification
   left unlinked (< min inliers), at most `per_photo` per photo, best covisibility first (pairs_covis.json), go
   through RoMa. Its matches (up to `num` per pair) snap onto a keypoint of the photo within SNAP_PX, else become
   new keypoints (on a 0.5 px grid, shared by all of that photo's pairs), appended to the photo's keypoints in
   database.db (features.db and the npz files keep the learned ones, at the same indices). The pairs keep their
   LightGlue matches too and are verified again.
   RoMa returns its matches for any two photos, and over flat ground they fit some fundamental matrix: on berlin_all
   verification linked all 60 pairs of photos > 400 m apart (median 527 inliers). So a RoMa match is kept only when
   its two ends, each lifted into the earlier model (Ground: the photo's pose and the depth of its nearest observed
   points), land within GROUND_M, and a pair only with GROUND_MIN such matches (those 60 pairs: none within 10 m;
   berlin_all's RoMa pairs: 81 % with >= 100). Without an earlier model (a first run) the dense pass is skipped.

The worker (modelforge/export/matchers.py) runs in build/matchers/venv (scripts/install_matchers), not in the image.
"""
import json
import os
import shutil
import sqlite3
import time

import numpy as np

from modelforge import jobs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOME = os.path.join(REPO, "build", "matchers")
WORKER = os.path.join(REPO, "modelforge", "export", "matchers.py")
DIR = "features_learned"
METHODS = {
    "gim-lightglue": "GIM's SuperPoint (2048 per photo) + LightGlue, retrained on internet video: the best sparse "
                     "method on berlin_all's hard pairs (41 of 64 vs COLMAP SIFT 0), 63 ms per photo, 15 ms per pair",
    "aliked-lightglue": "ALIKED (4096 per photo) + LightGlue: 37 of 64, 69 ms per photo, 15 ms per pair",
}
DENSE = {
    "off": "no dense matching",
    "roma": "RoMa on pairs between passes / flights that stay unlinked: 53 of 64 hard pairs, the only method for "
            "large changes of scale, ~0.6 s per pair",
}
DEFAULT_DENSE_PER_PHOTO = 3
DENSE_NUM = 2000  # RoMa matches per pair
FAR_S = 60.0
SNAP_PX = 1.5
GRID = 0.5
GROUND_M = 10.0  # a RoMa match is kept when its two ends, lifted into the earlier model, land this close (m)
GROUND_MIN = 100  # a RoMa pair needs this many such matches, else it is dropped
GROUND_MODEL = "refined"
PAIR_ID = 2147483647


def available():
    """None when the environment is installed, else why not."""
    if not os.path.exists(os.path.join(HOME, "venv", "bin", "python")):
        return "learned matchers not installed: run scripts/install_matchers in the dev container"
    return None


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env.update(PATH=os.path.join(HOME, "venv", "bin") + os.pathsep + env.get("PATH", ""),
               TORCH_HOME=os.path.join(REPO, "build", "torch_hub"), HF_HOME=os.path.join(REPO, "build", "hf_cache"),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MODELFORGE_MATCHERS=HOME)
    return env


def _run(req, ws, on_line=None, token=None):
    """Run the worker on request `req` (written to ws/<op>.json)."""
    why = available()
    if why:
        raise RuntimeError(why)
    os.makedirs(ws, exist_ok=True)
    path = os.path.join(ws, f"{req['op']}.json")
    json.dump(req, open(path, "w"))
    (token or jobs.Token()).run([os.path.join(HOME, "venv", "bin", "python"), WORKER, path], on_line=on_line,
                                env=_env())


def _progress_lines(op, progress):
    """on_line for the worker: "[matchers] op k / n ..." -> progress(k, n, rest)."""
    def on_line(line):
        s = line.split()
        if progress and len(s) >= 5 and s[0] == "[matchers]" and s[1] == op and s[3] == "/":
            try:
                progress(int(s[2]), int(s[4]), " ".join(s[5:]))
            except ValueError:
                pass
        elif not line.startswith("[matchers]"):
            print(line, flush=True)
    return on_line


def feature_dir(project, method):
    return os.path.join(project, DIR, method)


def _stamp(path):
    st = os.stat(path)
    return [st.st_size, round(st.st_mtime, 3)]


def method_of(project):
    """The learned method features.db was made with, None for SIFT (or a features.db made since)."""
    try:
        cur = json.load(open(os.path.join(project, DIR, "current.json")))
        return cur["method"] if cur.get("features_db") == _stamp(os.path.join(project, "features.db")) else None
    except (OSError, ValueError, KeyError):
        return None


def forget(project):
    """features.db is SIFT's now."""
    try:
        os.remove(os.path.join(project, DIR, "current.json"))
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------- features
def extract(project, method, groups, camera_model="OPENCV", camera_params=None, progress=None, token=None):
    """features_learned/<method>/*.npz for the photos (groups: names per camera, features.camera_groups) and
    features.db with their keypoints. progress(done, total, name)."""
    import pycolmap
    if method not in METHODS:
        raise ValueError(f"method must be one of {tuple(METHODS)}")
    t0 = time.time()
    names = [n for g in groups for n in g]
    images = os.path.join(project, "images")
    fdir = feature_dir(project, method)
    forget(project)
    hits = [0]
    lines = _progress_lines("extract", progress)

    def on_line(line):
        if line.startswith("[matchers] cached "):
            hits[0] = int(line.split()[2])
        lines(line)
    _run({"op": "extract", "method": method, "images_dir": images, "names": names, "out_dir": fdir},
         os.path.join(project, DIR), on_line, token)
    db_path = os.path.join(project, "features.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(db_path + ext):
            os.remove(db_path + ext)
    pycolmap.Database.open(db_path).close()  # import_images wants the database to exist
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if camera_params:
        reader.camera_params = ",".join(f"{x:.10g}" for x in camera_params)
    for group in groups:  # one camera per group, as for SIFT
        pycolmap.import_images(db_path, images, camera_mode=pycolmap.CameraMode.SINGLE, image_names=group,
                               options=reader)
    con = sqlite3.connect(db_path)
    ids = dict(con.execute("SELECT name, image_id FROM images"))
    total = 0
    rows = []
    for n in names:
        kp = np.load(os.path.join(fdir, n + ".npz"))["keypoints"].astype(np.float32)
        rows.append((ids[n], len(kp), 2, kp.tobytes()))
        total += len(kp)
    con.executemany("INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)", rows)
    con.commit()
    ncam = con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    con.close()
    json.dump({"method": method, "features_db": _stamp(db_path)}, open(os.path.join(project, DIR, "current.json"), "w"))
    return {"images": len(names), "keypoints": int(total), "cameras": int(ncam), "extractor": method,
            "cached": hits[0], "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------- matching
def _verify(db_path, pairs_path, min_inliers, token=None):
    import pycolmap
    from modelforge.sfm import reconstruct
    import threading
    opts = pycolmap.TwoViewGeometryOptions()
    opts.min_num_inliers = int(min_inliers)
    ct, done = pycolmap.CancellationToken(), threading.Event()
    if token is not None:
        threading.Thread(target=reconstruct._cancel_on, args=(token, ct, done), daemon=True).start()
    try:
        pycolmap.verify_matches(db_path, pairs_path, options=opts, cancellation_token=ct)
    except InterruptedError:
        raise jobs.Cancelled("stopped")
    finally:
        done.set()


def match(project, method, min_inliers=15, dense="off", per_photo=DEFAULT_DENSE_PER_PHOTO, num=DENSE_NUM,
          progress=None, token=None):
    """LightGlue on pairs.txt into database.db (a copy of features.db, made by the caller), verification, then
    the dense pass (module doc). progress(text, k, n). Returns stats."""
    db_path = os.path.join(project, "database.db")
    pairs_path = os.path.join(project, "pairs.txt")
    ws = os.path.join(project, DIR)
    t0 = time.time()
    _run({"op": "match", "method": method, "feat_dir": feature_dir(project, method), "pairs": pairs_path,
          "db": db_path}, ws, _progress_lines("match", progress and (lambda k, n, _: progress("matching: pair", k, n))),
         token)
    if progress:
        progress("geometric verification", 0, 1)
    _verify(db_path, pairs_path, min_inliers, token)
    stats = {"method": method, "sparse_seconds": round(time.time() - t0, 1)}
    if dense != "off":
        stats["dense"] = densify(project, dense, min_inliers, per_photo, num, progress, token)
    return stats


def verified(db_path):
    """{pair_id: verified inlier count} (two_view_geometries)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return dict(con.execute("SELECT pair_id, rows FROM two_view_geometries"))
    finally:
        con.close()


def dense_pairs(project, min_inliers, per_photo, far_s=None):
    """The unlinked pairs between passes / flights of pairs.txt, at most per_photo per photo, best covisibility
    first (pairs_covis.json: overlap from the earlier model; other pairs rank after them)."""
    far_s = FAR_S if far_s is None else far_s
    db_path = os.path.join(project, "database.db")
    poses = {r["image"]: r for r in json.load(open(os.path.join(project, "poses.json")))["images"]}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = dict(con.execute("SELECT name, image_id FROM images"))
    con.close()
    inl = verified(db_path)
    try:
        score = {frozenset(p[:2]): p[2] for p in json.load(open(os.path.join(project, "pairs_covis.json")))["pairs"]}
    except (OSError, ValueError, KeyError):
        score = {}
    cands = []
    for line in open(os.path.join(project, "pairs.txt")):
        s = line.split()
        if len(s) < 2 or s[0] not in ids or s[1] not in ids:
            continue
        a, b = s[0], s[1]
        ra, rb = poses.get(a, {}), poses.get(b, {})
        far = ra.get("source", 0) != rb.get("source", 0) or \
            abs((ra.get("time_s") or 0.0) - (rb.get("time_s") or 0.0)) > far_s
        i, j = sorted((ids[a], ids[b]))
        if far and inl.get(i * PAIR_ID + j, 0) < min_inliers:
            cands.append((score.get(frozenset((a, b)), 0.0), a, b))
    cands.sort(key=lambda c: -c[0])
    used, out = {}, []
    for _, a, b in cands:
        if used.get(a, 0) < per_photo and used.get(b, 0) < per_photo:
            out.append((a, b))
            used[a], used[b] = used.get(a, 0) + 1, used.get(b, 0) + 1
    return out, len(cands)


def snap(existing, new, snap_px=SNAP_PX, grid=GRID):
    """Keypoint indices for the points `new` (M, 2) of one photo whose keypoints are `existing` (N, 2): the nearest
    existing keypoint within snap_px, else a new keypoint at the point rounded to `grid`, shared by equal points.
    Returns (indices (M,), the new keypoints to append (K, 2)); new ones are numbered from N."""
    n = len(existing)
    idx = np.full(len(new), -1, np.int64)
    if n:
        cell = snap_px
        ek = np.floor(existing / cell).astype(np.int64)
        key = (ek[:, 0] << 32) + ek[:, 1]
        order = np.argsort(key)
        skey = key[order]
        nk = np.floor(new / cell).astype(np.int64)
        best = np.full(len(new), np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = ((nk[:, 0] + dx) << 32) + (nk[:, 1] + dy)
                lo, hi = np.searchsorted(skey, q, "left"), np.searchsorted(skey, q, "right")
                # cells hold few points: walk them in vectorized steps
                k = lo.copy()
                while True:
                    live = k < hi
                    if not live.any():
                        break
                    e = order[k[live]]
                    d = np.linalg.norm(existing[e] - new[live], axis=1)
                    rows = np.flatnonzero(live)
                    better = (d < best[rows]) & (d <= snap_px)
                    best[rows[better]] = d[better]
                    idx[rows[better]] = e[better]
                    k[live] += 1
    rest = np.flatnonzero(idx < 0)
    if not len(rest):
        return idx, np.zeros((0, 2), np.float32)
    q = np.round(new[rest] / grid) * grid
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    idx[rest] = n + inv.ravel()
    return idx, uq.astype(np.float32)


class Ground:
    """The project's earlier model (sparse/GROUND_MODEL) as a lift per photo: pixel -> 3D point on the ray, at the
    median depth of the photo's 8 observed points nearest in the image. `ok` False without a model; photos it does
    not hold (or that observe < 20 points) are not in `photos`."""

    def __init__(self, project, names=None, model=GROUND_MODEL):
        from modelforge.gui.data import read_cameras_bin, read_images_bin, read_keypoints, read_points3d_bin
        d = os.path.join(project, "sparse", model)
        path = os.path.join(d, "images.bin")
        self.ok, self.photos, self._flann = os.path.exists(path), {}, {}
        if not self.ok:
            return
        self.cams = read_cameras_bin(os.path.join(d, "cameras.bin"))
        keep, allp = [], []
        for im in read_images_bin(path):
            kp = read_keypoints(path, im[4], im[5])
            v = kp["pid"] >= 0
            allp.append(kp["pid"][v])  # read_points3d_bin wants every point of the file, with its track length
            if v.sum() >= 20 and (names is None or im[0] in names):
                keep.append((im, kp["xy"][v], kp["pid"][v]))
        ids, counts = np.unique(np.concatenate(allp) if allp else np.zeros(0, np.int64), return_counts=True)
        xyz, _ = read_points3d_bin(os.path.join(d, "points3D.bin"), ids, counts)
        for (name, cid, R, t, *_), xy, pid in keep:
            z = (xyz[np.searchsorted(ids, pid)] @ R.T + t)[:, 2]
            self.photos[name] = (xy.astype(np.float32), z, self.cams[cid], R, t)

    def lift(self, name, uv):
        """3D points (N, 3) of the pixels uv (N, 2) of photo `name` (in `photos`)."""
        import cv2
        xy, z, cam, R, t = self.photos[name]
        if name not in self._flann:
            self._flann[name] = cv2.flann_Index(xy, dict(algorithm=1, trees=4))
        idx, _ = self._flann[name].knnSearch(np.ascontiguousarray(uv, np.float32), min(8, len(xy)), params={})
        rays = np.c_[np.asarray(cam.cam_from_img(np.asarray(uv, np.float64))), np.ones(len(uv))]
        return (rays * np.median(z[idx], axis=1)[:, None] - t) @ R

    def agree(self, a, b, ka, kb, within=GROUND_M):
        """Which matches (ka in photo a, kb in photo b) land within `within` of each other."""
        return np.linalg.norm(self.lift(a, ka) - self.lift(b, kb), axis=1) < within


def _meta(out_dir):
    return os.path.join(out_dir, "meta.json")


def _dense_blocks(out_dir):
    """(a, b, offsets, ka, kb) of every finished block under out_dir (one folder per run)."""
    for root, _, files in sorted(os.walk(out_dir)):
        for f in sorted(files):
            if f.endswith(".npz") and not f.endswith(".part.npz"):
                try:
                    with np.load(os.path.join(root, f)) as d:  # each d[key] reads the array again: once per block
                        yield d["a"], d["b"], d["offsets"], d["ka"], d["kb"]
                except (OSError, ValueError, KeyError):
                    continue


def _dense_done(out_dir, num):
    """The pairs already matched under out_dir with `num` matches per pair; anything else there is removed."""
    try:
        ok = json.load(open(_meta(out_dir))).get("num") == int(num)
    except (OSError, ValueError, AttributeError):
        ok = False
    if not ok:
        shutil.rmtree(out_dir, ignore_errors=True)
        return set()
    return {(str(a), str(b)) for A, B, *_ in _dense_blocks(out_dir) for a, b in zip(A, B)}


def densify(project, dense, min_inliers, per_photo, num=DENSE_NUM, progress=None, token=None):
    """The dense pass (module doc); returns its stats."""
    if dense not in DENSE or dense == "off":
        raise ValueError(f"dense must be one of {tuple(k for k in DENSE if k != 'off')}")
    t0 = time.time()
    db_path = os.path.join(project, "database.db")
    pairs, candidates = dense_pairs(project, min_inliers, per_photo)
    stats = {"method": dense, "candidates": candidates, "pairs": len(pairs), "per_photo": per_photo}
    if not pairs:
        return stats | {"seconds": round(time.time() - t0, 1)}
    ground = Ground(project, names={n for p in pairs for n in p})
    if not ground.ok:  # RoMa's matches cannot be checked: refine once, then match again
        return stats | {"skipped": "no earlier model to check RoMa's matches", "seconds": round(time.time() - t0, 1)}
    pairs = [p for p in pairs if p[0] in ground.photos and p[1] in ground.photos]
    stats["checkable"] = len(pairs)
    if not pairs:
        return stats | {"seconds": round(time.time() - t0, 1)}
    out_dir = os.path.join(project, DIR, dense)
    want = set(pairs)
    done = _dense_done(out_dir, num)  # pairs an earlier run (maybe killed, maybe on other pairs) already matched
    todo = [p for p in pairs if p not in done]
    stats["reused"] = len(pairs) - len(todo)
    if todo:
        run = os.path.join(out_dir, time.strftime("run_%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"num": int(num)}, open(_meta(out_dir), "w"))
        _run({"op": "roma", "images_dir": os.path.join(project, "images"), "pairs": [list(p) for p in todo],
              "out_dir": run, "num": int(num)}, os.path.join(project, DIR),
             _progress_lines("roma", progress and (lambda k, n, _: progress(f"{dense}: pair", k, n))), token)
    if progress:
        progress(f"{dense}: merging matches into the keypoints", 0, 1)
    con = sqlite3.connect(db_path, timeout=60)
    ids = dict(con.execute("SELECT name, image_id FROM images"))
    # every dense point per photo, then one snap per photo
    per = {}  # image name -> list of (pair k, side, points)
    pair_list, dropped = [], 0
    for A, B, off, KA, KB in _dense_blocks(out_dir):
        for t in range(len(A)):
            a, b = str(A[t]), str(B[t])
            if (a, b) not in want:
                continue
            want.discard((a, b))  # once, should two runs hold it
            ka, kb = KA[off[t]:off[t + 1]], KB[off[t]:off[t + 1]]
            ok = ground.agree(a, b, ka, kb)
            if ok.sum() < GROUND_MIN:  # RoMa matches any two photos: these do not show the same ground
                dropped += 1
                continue
            ka, kb = ka[ok], kb[ok]  # copies, not views holding the block
            k = len(pair_list)
            pair_list.append((a, b, len(ka)))
            per.setdefault(a, []).append((k, 0, ka))
            per.setdefault(b, []).append((k, 1, kb))
    index = [[None, None] for _ in pair_list]
    added = 0
    for name, items in per.items():
        iid = ids[name]
        r, c, data = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (iid,)).fetchone()
        kp = np.frombuffer(data, np.float32).reshape(r, c)
        pts = np.concatenate([p for _, _, p in items]).astype(np.float64)
        idx, new = snap(kp[:, :2].astype(np.float64), pts)
        s = 0
        for k, side, p in items:
            index[k][side] = idx[s: s + len(p)]
            s += len(p)
        if len(new):
            extra = np.zeros((len(new), c), np.float32)
            extra[:, :2] = new
            if c == 4:  # SIFT's x, y, scale, orientation: unit scale
                extra[:, 2] = 1.0
            elif c == 6:  # SIFT's x, y and affine shape a11 a12 a21 a22: the identity
                extra[:, 2] = extra[:, 5] = 1.0
            allkp = np.concatenate([kp, extra])
            con.execute("UPDATE keypoints SET rows = ?, data = ? WHERE image_id = ?", (len(allkp), allkp.tobytes(),
                                                                                     iid))
            added += len(new)
    sub = os.path.join(project, DIR, "dense_pairs.txt")
    with open(sub, "w") as f:
        for (a, b, _), (ia, ib) in zip(pair_list, index):
            i, j = ids[a], ids[b]
            m = np.stack([ia, ib], 1)
            if i > j:
                i, j, m = j, i, m[:, ::-1]
            pid = i * PAIR_ID + j
            row = con.execute("SELECT rows, data FROM matches WHERE pair_id = ?", (pid,)).fetchone()
            if row and row[0]:
                m = np.concatenate([np.frombuffer(row[1], np.uint32).reshape(-1, 2).astype(np.int64), m])
            m = np.unique(m, axis=0).astype(np.uint32)
            con.execute("INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, 2, ?)",
                        (pid, len(m), np.ascontiguousarray(m).tobytes()))
            con.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pid,))
            f.write(f"{a} {b}\n")
    con.commit()
    con.close()
    if progress:
        progress(f"{dense}: geometric verification", 0, 1)
    _verify(db_path, sub, min_inliers, token)
    inl = verified(db_path)
    linked = sum(1 for a, b, _ in pair_list
                 if inl.get(min(ids[a], ids[b]) * PAIR_ID + max(ids[a], ids[b]), 0) >= min_inliers)
    return stats | {"linked": linked, "dropped": dropped, "matches": int(sum(n for _, _, n in pair_list)),
                    "keypoints_added": int(added),
                    "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------- locate
def query(project, method, images_dir, queries, ws, token=None, progress=None):
    """Match photos outside the model (register more photos, held-out evaluation) to model photos with the project's
    learned features: queries [{"image", "model": [names]}] -> [(query keypoints (N, 2), [(model name, matches
    (M, 2): query index, model keypoint index)])]."""
    out = os.path.join(ws, "query.npz")
    _run({"op": "query", "method": method, "feat_dir": feature_dir(project, method), "images_dir": images_dir,
          "queries": queries, "out": out}, ws,
         _progress_lines("query", progress and (lambda k, n, _: progress(k, n))), token)
    d = np.load(out)
    return [(d[f"q{i}_kp"], [(name, d[f"q{i}_m{j}"]) for j, name in enumerate(q["model"])])
            for i, q in enumerate(queries)]

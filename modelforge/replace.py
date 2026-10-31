"""Content-aware repair of a splat (the user asked, 2026-10-31): remove a weak object (a smeared tree or bush), fill the
ground under it from similar ground elsewhere (PatchMatch, modelforge/patchfill.py) and optionally place a sharper
object of the same kind (a donor) there. Prototype and its findings: build/treeswap/ (berlin).

Objects: the Gaussians the semantic features call vegetation (any of QUERIES at its threshold), small (< MAX_SIZE: sky
floaters labelled vegetation are large), ABOVE m above the ground (`_ground`: from the SfM points), give a canopy height model on
a CELL grid; smoothed, every cell climbs to its highest neighbour (watershed), and basins merge unless the dip between
them is at least max(PROMINENCE_M, PROMINENCE_REL x the lower peak) (persistence, from the highest saddle down).
Per object: footprint, height, Gaussians per m2, the share of vegetation among what stands over it (veg_share), and,
after `rate`, how sharp its render is against the photos that see it best (export/replace_rate.py).

Edits (PROJECT/replace/edits.json, in order) are applied to the base splat (splat/splat_enu.pre_replace.ply, the splat
before the first edit; objects are always found on it, so their ids stay), so undoing one = dropping it and applying
the rest again. A new training (the splat no longer the one the last apply wrote) makes the base, the objects and the
edits stale: they are dropped.

Remove: every Gaussian over the object's cells (grown by GROW cells) more than CUT m above the ground, and large ones
(> HAZE_M) reaching into its volume (the fog a broken tree is often drawn with). Fill: the hole's ground layer
(GROUND_TOP: grass included) from per-cell PatchMatch sources (open ground: nothing above GROUND_TOP; no edit's hole),
moved onto the hole's terrain, its colour pulled half way to the surrounding ring. Replace: the donor's Gaussians
(taken like a removal) scaled uniformly to the target's height about their foot, stood on the target's terrain point by
point, band 0 pulled COLOUR of the way to the old object's mean colour per third of the height; no turn about the
vertical (the sunlit side stays where the sun was)."""
import json
import os
import shutil
import time

import cv2
import numpy as np

from modelforge import jobs, rig, semantic, splat, splat_mesh

DIR = "replace"
BACKUP = os.path.join("splat", "splat_enu.pre_replace.ply")
QUERIES = {"vegetation": 0.55, "trees": 0.58}
CELL, SIGMA = 0.5, 1.5  # m, cells (canopy smoothing)
ABOVE, MIN_AREA, MIN_HEIGHT, MAX_HEIGHT = 1.5, 6.0, 3.0, 40.0  # m, m2, m, m
MAX_SIZE = 1.0  # m: the largest axis of a vegetation Gaussian
PROMINENCE_M, PROMINENCE_REL = 1.0, 0.25
CUT, GROW = 0.3, 2  # m above the ground that goes with an object; cells its footprint is grown by
GROUND_TOP = 1.5  # m: the ground layer (grass, low shrubs) the fill copies; a source cell must have nothing above it
# (a cell is copied whole: copying up to 1 m from cells with up to 1.5 m cut low bushes into dark needle fragments)
HAZE_M = 0.5
GROUND_CELL, GROUND_PCT = 0.5, 0.1  # m; the height of a cell: this quantile of the SfM points in it
SPIKE_M, SPIKE_WIN, OPEN_M = 2.0, 2.5, 10.0  # m: a cell this far off its neighbourhood's median is dropped; the
# opening's width (wider than the crowns it removes)
BANDS, COLOUR = 3, 0.8
TARGET_SHARP, TARGET_VEG = 0.2, 0.6  # automatic targets: render this much blurrier than the photos, mostly vegetation
DONOR_SHARP, CLEAN = 0.6, 0.8  # donors: as sharp as their photos, clean (nothing else standing over them)
MAP_CELL, MAP_PHOTOS = 2.0, 300  # m; photos the detail map renders (spread over the flight)
GOOD_SOURCE, MAX_COPY_M = 0.6, 1.0  # fill sources keep at least this share of the photos' detail (when the detail
# map exists: else similar but equally blurry ground was copied, no better); copied Gaussians are smaller than this
# (larger ones near the ground are glare and needles: bright blobs in the fill)
WEAK_GROUND, MIN_GROUND_M2, MIN_WEIGHT = 0.3, 8.0, 0.5  # a weak ground area: detail kept below this, at least this
# large, seen with at least this much photo detail per cell (flat road or water count little)


def _dir(project):
    return os.path.join(project, DIR)


def _key(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime_ns)]


def _json(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def base(project):
    """The splat edits apply to: the backup before the first edit, else the project's splat. A splat that is not the
    one the last apply wrote (a new training) drops the backup, the objects and the edits."""
    live, backup = splat_mesh.splat_path(project), os.path.join(project, BACKUP)
    state = _json(os.path.join(_dir(project), "state.json"), {})
    if os.path.exists(backup):
        if os.path.exists(live) and state.get("written") and state["written"] != _key(live):
            os.remove(backup)
            shutil.rmtree(_dir(project), ignore_errors=True)
            return live
        return backup
    return live


def edits(project):
    return _json(os.path.join(_dir(project), "edits.json"), [])


# ------------------------------------------------------------------------------------------------ finding objects
def _relevance(project, query):
    """Relevance of every Gaussian of the base splat to a query (cached per query and features file)."""
    feats = os.path.join(project, semantic.DIR, "features.npz")
    key = _key(feats)
    path = os.path.join(_dir(project), f"relevance_{query.replace(' ', '_')}.npy")
    meta = path + ".json"
    if os.path.exists(path) and _json(meta) == key:
        return np.load(path)
    enc = semantic.TextEncoder()
    try:
        emb = enc.encode([query] + list(semantic.NEGATIVES))
    finally:
        enc.close()
    rel = semantic.Features(project).relevance(emb[:1], emb[1:]).astype(np.float32)
    np.save(path, rel)
    json.dump(key, open(meta, "w"))
    return rel


def _climb(v):
    """Peak index (flat) every cell reaches by steepest ascent over its 3 x 3 neighbours."""
    H, W = v.shape
    pad = np.pad(v, 1, constant_values=-1e9)
    best = np.arange(H * W).reshape(H, W)
    bestv = v.copy()
    ii, jj = np.mgrid[0:H, 0:W]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            n = pad[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            better = n > bestv
            bestv = np.where(better, n, bestv)
            best = np.where(better, np.clip(ii + di, 0, H - 1) * W + np.clip(jj + dj, 0, W - 1), best)
    p = best.ravel()
    while True:
        q = p[p]
        if np.array_equal(q, p):
            return p
        p = q


def _merge(sm, peak, min_cells):
    """Basins merged where the dip between two is shallow (see the module doc), labelled by their highest peak."""
    H, W = sm.shape
    lab, v = peak.ravel(), sm.ravel()
    idx = np.arange(H * W).reshape(H, W)
    pairs = []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        m = (lab[a] >= 0) & (lab[b] >= 0) & (lab[a] != lab[b])
        la, lb = np.minimum(lab[a[m]], lab[b[m]]), np.maximum(lab[a[m]], lab[b[m]])
        pairs.append(np.stack([la, lb, np.minimum(v[a[m]], v[b[m]])], 1))
    pairs = np.concatenate(pairs)
    if not len(pairs):
        return peak
    key = pairs[:, 0] * (H * W) + pairs[:, 1]
    order = np.lexsort((-pairs[:, 2], key))
    key, pairs = key[order], pairs[order]
    pairs = pairs[np.r_[True, key[1:] != key[:-1]]]  # one row per pair: its highest saddle
    pairs = pairs[np.argsort(-pairs[:, 2])]
    parent = {}
    size = {int(k): int(c) for k, c in zip(*np.unique(lab[lab >= 0], return_counts=True))}
    top = {k: float(v[k]) for k in size}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for a, b, sad in pairs:
        ra, rb = root(int(a)), root(int(b))
        if ra == rb:
            continue
        low = min(top[ra], top[rb])
        if low - sad < max(PROMINENCE_M, PROMINENCE_REL * low) or min(size[ra], size[rb]) < min_cells:
            hi_, lo_ = (ra, rb) if top[ra] >= top[rb] else (rb, ra)
            parent[lo_] = hi_
            size[hi_] += size[lo_]
    out = lab.copy()
    for k in size:
        r = root(k)
        if r != k:
            out[lab == k] = r
    return out.reshape(H, W)


def _ground(project, fallback, lo, hi):
    """Ground height map over the box (lo, hi) on GROUND_CELL cells, from the SfM points (sparse_pc.ply; else the
    `fallback` points): per cell their GROUND_PCT height (>= 2 points), cells more than SPIKE_M off the median of
    their neighbourhood dropped (pits and spikes: floaters, stray points), the gaps filled smoothly (rig.fill_holes),
    then a grey opening of OPEN_M (min- then max-filter) takes away bushes and canopies standing on it. The splat's
    own Gaussians are not used when SfM points exist: floaters above and below the ground put cells tens of metres
    off (berlin: -351 ... +44 m on a site at about -13 m)."""
    path = os.path.join(project, "sparse_pc.ply")
    if os.path.exists(path):
        import plyfile
        v = plyfile.PlyData.read(path)["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    else:
        pts = fallback
    cell = GROUND_CELL
    W, H = int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    m = np.all((pts[:, :2] >= lo) & (pts[:, :2] < hi), 1)
    p = pts[m]
    c = ((p[:, 1] - lo[1]) // cell).astype(np.int64) * W + ((p[:, 0] - lo[0]) // cell).astype(np.int64)
    order = np.lexsort((p[:, 2], c))
    c, z = c[order], p[order, 2]
    cells, first, counts = np.unique(c, return_index=True, return_counts=True)
    ok = counts >= 2
    grid = np.zeros(H * W, np.float32)
    known = np.zeros(H * W, bool)
    grid[cells[ok]] = z[first[ok] + (counts[ok] * GROUND_PCT).astype(int)]
    known[cells[ok]] = True
    grid, known = grid.reshape(H, W), known.reshape(H, W)
    filled = rig.fill_holes(grid, known)
    k = int(SPIKE_WIN / cell) | 1
    med = cv2.medianBlur(filled, 5) if k <= 5 else cv2.medianBlur(cv2.medianBlur(filled, 5), 5)
    known &= np.abs(grid - med) <= SPIKE_M
    g = rig.fill_holes(grid, known)
    r = int(OPEN_M / cell) | 1
    g = cv2.dilate(cv2.erode(g, np.ones((r, r), np.uint8)), np.ones((r, r), np.uint8))
    return Ground(g, (lo[0], lo[1], cell))


class Ground:
    """The ground height map saved with the objects, read bilinearly."""

    def __init__(self, grid, origin):
        self.g, (self.x0, self.y0, self.cell) = grid, origin

    def __call__(self, x, y):
        H, W = self.g.shape
        u = np.clip((np.asarray(x, float) - self.x0) / self.cell - 0.5, 0, W - 1.001)
        v = np.clip((np.asarray(y, float) - self.y0) / self.cell - 0.5, 0, H - 1.001)
        j, i = u.astype(int), v.astype(int)
        fu, fv = u - j, v - i
        g = self.g
        return (1 - fv) * ((1 - fu) * g[i, j] + fu * g[i, j + 1]) + fv * ((1 - fu) * g[i + 1, j] + fu * g[i + 1, j + 1])


def find(project, queries=None, on_line=print, kind=None):
    """Find the objects of the base splat (see the module doc) -> replace/objects.json + objects.npz. Returns them.
    queries: {text: relevance threshold} (default QUERIES); kind: a mask of the Gaussians that may form objects
    instead (tests)."""
    t0 = time.time()
    queries = queries or QUERIES
    os.makedirs(_dir(project), exist_ok=True)
    src = base(project)
    g = splat_mesh.load_splat(src)
    P = g["means"].astype(np.float64)
    op = 1 / (1 + np.exp(-g["opacities"].reshape(-1)))
    if kind is None:
        kind = np.zeros(len(P), bool)
        for q, thr in queries.items():
            kind |= _relevance(project, q) >= thr
    lo, hi = np.percentile(P[:, :2], 0.5, 0), np.percentile(P[:, :2], 99.5, 0)
    ground = _ground(project, P[(op > 0.5) & ~kind], lo, hi)
    h = P[:, 2] - ground(P[:, 0], P[:, 1])
    inside = np.all((P[:, :2] >= lo) & (P[:, :2] < hi), 1)
    small = np.exp(g["scales"].max(1)) < MAX_SIZE
    sel = np.flatnonzero(kind & inside & small & (h > ABOVE) & (h < MAX_HEIGHT) & (op > 0.2))
    W, H = int(np.ceil((hi[0] - lo[0]) / CELL)), int(np.ceil((hi[1] - lo[1]) / CELL))

    def cells(idx):
        return (((P[idx, 1] - lo[1]) // CELL).astype(int).clip(0, H - 1) * W
                + ((P[idx, 0] - lo[0]) // CELL).astype(int).clip(0, W - 1))
    cell = cells(sel)
    chm = np.zeros(H * W, np.float32)
    np.maximum.at(chm, cell, h[sel].astype(np.float32))
    chm = chm.reshape(H, W)
    occupied = cv2.morphologyEx((chm > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    sm = cv2.GaussianBlur(chm, (0, 0), SIGMA)
    sm[~occupied] = 0
    peak = _climb(np.where(occupied, sm, -1.0)).reshape(H, W)
    peak[~occupied] = -1
    peak = _merge(sm, peak, MIN_AREA / CELL ** 2).ravel()
    lab_of = peak[cell]
    ids, inv = np.unique(lab_of, return_inverse=True)
    inv = inv.ravel()
    area_cells = np.bincount(np.unique(np.stack([inv, cell], 1), axis=0)[:, 0], minlength=len(ids))
    t = json.load(open(os.path.join(project, "transforms.json")))
    C = np.array([np.array(f["transform_matrix"], float)[:3, 3] for f in t["frames"]])
    fl = float(t["frames"][0].get("fl_x", t.get("fl_x")))
    size = np.exp(g["scales"][sel].max(1))
    objs, new_id = [], np.full(len(ids), -1)
    for k, (pid, n_cells) in enumerate(zip(ids, area_cells)):
        if pid < 0:
            continue
        m = inv == k
        area, top = n_cells * CELL * CELL, float(chm.ravel()[pid])
        if area < MIN_AREA or top < MIN_HEIGHT:
            continue
        q = P[sel[m]]
        py, px = divmod(int(pid), W)
        cx, cy = lo[0] + (px + 0.5) * CELL, lo[1] + (py + 0.5) * CELL
        gz = float(ground(cx, cy))
        d = float(np.linalg.norm(C - np.array([cx, cy, gz + top / 2]), axis=1).min())
        new_id[k] = len(objs)
        objs.append({"id": len(objs), "x": round(cx, 2), "y": round(cy, 2), "ground_z": round(gz, 2),
                     "height": round(top, 2), "area_m2": round(area, 1), "radius": round(float(np.sqrt(area / np.pi)), 2),
                     "gaussians": int(m.sum()), "per_m2": round(float(m.sum() / area), 1),
                     "median_size_cm": round(float(np.median(size[m]) * 100), 1), "camera_m": round(d, 1),
                     "cm_per_px": round(100 * d / fl, 2),
                     "box": [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]})
    lab = new_id[inv]
    crown = np.full(H * W, -1, np.int32)
    crown[cell[lab >= 0]] = lab[lab >= 0]
    over = np.flatnonzero(inside & (h > CUT) & (h < MAX_HEIGHT) & (op > 0.2))
    ol = crown[cells(over)]
    keep = ol >= 0
    tot = np.bincount(ol[keep], minlength=len(objs))
    veg = np.bincount(ol[keep], weights=(kind & small)[over[keep]], minlength=len(objs))
    for o in objs:
        o["veg_share"] = round(float(veg[o["id"]] / max(tot[o["id"]], 1)), 3)
    np.savez(os.path.join(_dir(project), "objects.npz"), crown=crown.reshape(H, W), lo=lo, cell=CELL,
             ground_grid=ground.g, ground_origin=np.array([ground.x0, ground.y0, ground.cell]))
    json.dump({"splat": _key(src), "queries": queries, "objects": objs}, open(os.path.join(_dir(project),
                                                                                           "objects.json"), "w"))
    on_line(f"replace: {len(objs)} objects ({', '.join(queries)}) in {time.time() - t0:.0f} s")
    return objs


def objects(project):
    """The objects found on the current base splat, or None (find them first)."""
    d = _json(os.path.join(_dir(project), "objects.json"))
    src = base(project)
    if not d or not os.path.exists(src) or d.get("splat") != _key(src):
        return None
    return d["objects"]


def _save_objects(project, objs):
    p = os.path.join(_dir(project), "objects.json")
    d = _json(p)
    d["objects"] = objs
    json.dump(d, open(p, "w"))


def rate(project, on_line=print, on_progress=None, job=None):
    """Rate every object against its photos (export/replace_rate.py): adds sharp, psnr, photos to objects.json."""
    objs = objects(project)
    if objs is None:
        raise RuntimeError("find the objects first")
    job = job or jobs.Token()
    work = _dir(project)
    npz = os.path.join(work, "rate_in.npz")
    np.savez(npz, **splat_mesh.load_splat(base(project)))
    spec = {"npz": npz, "transforms": os.path.join(project, "transforms.json"), "root": project,
            "objects": [{k: o[k] for k in ("id", "x", "y", "ground_z", "height", "box")} for o in objs],
            "out": os.path.join(work, "rated.json"), "crops": os.path.join(work, "crops")}
    json.dump(spec, open(os.path.join(work, "rate.json"), "w"))

    def line(text):
        if text.startswith("[rate] object") and on_progress:
            k, n = text.split()[2], text.split()[4]
            on_progress("rating: object", int(k), int(n))
        elif not text.startswith("[rate]"):
            on_line(text)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "replace_rate.py")
    try:
        job.run([os.path.join(splat.NS_BIN, "python"), script, os.path.join(work, "rate.json")], line, env=splat._env())
    finally:
        if os.path.exists(npz):
            os.remove(npz)
    got = json.load(open(spec["out"]))
    for o in objs:
        o.update(got.get(str(o["id"]), {}))
    _save_objects(project, objs)
    s = [o["sharp"] for o in objs if o.get("sharp") is not None]
    on_line(f"replace: rated {len(s)} of {len(objs)} objects (sharpness median {np.median(s):.2f})" if s else
            "replace: no object could be rated (no photo sees one near its centre)")
    return objs


def object_at(project, x, y):
    """The object whose footprint holds the ENU point (x, y), or the nearest within 2 cells; None if none."""
    d = np.load(os.path.join(_dir(project), "objects.npz"))
    crown, lo, cell = d["crown"], d["lo"], float(d["cell"])
    H, W = crown.shape
    j, i = int((x - lo[0]) // cell), int((y - lo[1]) // cell)
    for r in range(3):
        win = crown[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
        if (win >= 0).any():
            ids, counts = np.unique(win[win >= 0], return_counts=True)
            return int(ids[np.argmax(counts)])
    return None


def detail_map(project, on_line=print, on_progress=None, job=None, photos=MAP_PHOTOS):
    """How much of the photos' detail the base splat keeps, on a MAP_CELL grid over the objects' grid (render vs
    photo of `photos` photos, per window, placed by the rendered depth: export/replace_rate.py --map) ->
    replace/detail.npz (ratio, weight, lo, cell)."""
    job = job or jobs.Token()
    d = np.load(os.path.join(_dir(project), "objects.npz"))
    lo = d["lo"]
    H, W = d["crown"].shape
    shape = (int(np.ceil(H * float(d["cell"]) / MAP_CELL)), int(np.ceil(W * float(d["cell"]) / MAP_CELL)))
    work = _dir(project)
    npz = os.path.join(work, "map_in.npz")
    np.savez(npz, **splat_mesh.load_splat(base(project)))
    out = os.path.join(work, "detail.npz")
    spec = {"npz": npz, "transforms": os.path.join(project, "transforms.json"), "root": project, "photos": int(photos),
            "lo": [float(lo[0]), float(lo[1])], "cell": MAP_CELL, "shape": list(shape), "out": out + ".part.npz"}
    json.dump(spec, open(os.path.join(work, "map.json"), "w"))

    def line(text):
        if text.startswith("[rate] photo") and on_progress:
            on_progress("detail map: photo", int(text.split()[2]), int(text.split()[4]))
        elif not text.startswith("[rate]"):
            on_line(text)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "replace_rate.py")
    try:
        job.run([os.path.join(splat.NS_BIN, "python"), script, "--map", os.path.join(work, "map.json")], line,
                env=splat._env())
    finally:
        if os.path.exists(npz):
            os.remove(npz)
    m = np.load(spec["out"])
    np.savez(out, ratio=m["ratio"], weight=m["weight"], lo=lo, cell=MAP_CELL, splat=np.array(_key(base(project))))
    os.remove(spec["out"])
    areas = ground_areas(project)
    on_line(f"replace: detail map of {np.isfinite(m['ratio']).sum():,} cells; {len(areas)} weak ground areas")
    return areas


def ground_areas(project):
    """Weak ground areas from the detail map: connected cells keeping less than WEAK_GROUND of the photos' detail,
    seen well enough (MIN_WEIGHT), with no object on them (the objects' footprints grown by a cell), at least
    MIN_GROUND_M2 large -> replace/ground.json [{id, x, y, radius, area_m2, detail, box}] + ground_labels.npy."""
    path = os.path.join(_dir(project), "detail.npz")
    if not os.path.exists(path):
        return []
    cached = os.path.join(_dir(project), "ground.json")
    if os.path.exists(cached) and os.path.getmtime(cached) >= max(os.path.getmtime(path), os.path.getmtime(
            os.path.join(_dir(project), "objects.npz"))):
        return _json(cached, [])
    m = np.load(path)
    if list(m["splat"]) != _key(base(project)):
        return []
    ratio, weight, lo, cell = m["ratio"], m["weight"], m["lo"], float(m["cell"])
    d = np.load(os.path.join(_dir(project), "objects.npz"))
    crown, fine = d["crown"], float(d["cell"])
    k = int(round(cell / fine))
    H, W = ratio.shape
    occ = np.zeros((H * k, W * k), bool)
    occ[:crown.shape[0], :crown.shape[1]] = crown >= 0
    occ = occ.reshape(H, k, W, k).any((1, 3))
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    weak = np.nan_to_num(ratio, nan=1.0) < WEAK_GROUND
    weak &= (weight >= MIN_WEIGHT) & ~occ
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    areas, labels = [], np.full((H, W), -1, np.int32)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA] * cell * cell
        if a < MIN_GROUND_M2:
            continue
        mk = lab == i
        ys, xs = np.nonzero(mk)
        labels[mk] = len(areas)
        areas.append({"id": len(areas), "x": round(float(lo[0] + (cent[i][0] + 0.5) * cell), 2),
                      "y": round(float(lo[1] + (cent[i][1] + 0.5) * cell), 2), "area_m2": round(float(a), 1),
                      "radius": round(float(np.sqrt(a / np.pi)), 2),
                      "detail": round(float(np.nanmean(ratio[mk])), 3),
                      "box": [float(lo[0] + xs.min() * cell), float(lo[1] + ys.min() * cell),
                              float(lo[0] + (xs.max() + 1) * cell), float(lo[1] + (ys.max() + 1) * cell)]})
    areas.sort(key=lambda x: x["detail"])
    np.save(os.path.join(_dir(project), "ground_labels.npy"), labels)
    json.dump(areas, open(os.path.join(_dir(project), "ground.json"), "w"))
    return areas


def suggest_targets(objs):
    """Objects to replace automatically: rated, rendered much blurrier than their photos, mostly vegetation."""
    return [o for o in objs if o.get("sharp") is not None and o.get("photos", 0) >= 2 and o["sharp"] < TARGET_SHARP
            and o["veg_share"] >= TARGET_VEG]


def donors(objs, target, used=(), exclude=()):
    """Donor candidates for target, best first: as sharp as their photos, clean, denser than the median, of a similar
    shape (height and radius within 0.6-1.67 x, similar height / radius); ranked by absolute detail (small Gaussians,
    photos taken close, shrunk rather than enlarged); used ones after the others."""
    dens = np.median([o["per_m2"] for o in objs])
    ok = [o for o in objs if o["id"] != target["id"] and o["id"] not in exclude
          and (o.get("sharp") or 0) >= DONOR_SHARP and o["per_m2"] >= dens and o["veg_share"] >= CLEAN
          and 0.6 < o["height"] / target["height"] < 1.67 and 0.6 < o["radius"] / target["radius"] < 1.67
          and abs(np.log(o["height"] / o["radius"]) - np.log(target["height"] / target["radius"])) < 0.35]
    ok.sort(key=lambda o: (o["id"] in used, o["median_size_cm"] * max(1.0, target["height"] / o["height"]) ** 2
                           * (1 + o["cm_per_px"])))
    return ok


# ------------------------------------------------------------------------------------------------ applying edits
class _Scene:
    """The base splat with what the edits need: heights above the ground, the crown raster, ground rasters."""

    def __init__(self, project):
        self.g = splat_mesh.load_splat(base(project))
        d = np.load(os.path.join(_dir(project), "objects.npz"))
        self.crown, self.lo, self.cell = d["crown"], d["lo"], float(d["cell"])
        self.ground = Ground(d["ground_grid"], d["ground_origin"])
        self.objs = {o["id"]: o for o in _json(os.path.join(_dir(project), "objects.json"))["objects"]}
        self.P = self.g["means"].astype(np.float64)
        self.gz = self.P[:, 2] - self.ground(self.P[:, 0], self.P[:, 1])
        self.op = 1 / (1 + np.exp(-self.g["opacities"].reshape(-1)))
        H, W = self.crown.shape
        self.cij = ((self.P[:, :2] - self.lo) // self.cell).astype(np.int64)
        self.inside = (self.cij[:, 0] >= 0) & (self.cij[:, 0] < W) & (self.cij[:, 1] >= 0) & (self.cij[:, 1] < H)
        self.R = None
        self.project = project

    def mask(self, oid):
        """(H, W) bool: the object's cells grown by GROW."""
        m = (self.crown == oid).astype(np.uint8)
        return cv2.dilate(m, np.ones((2 * GROW + 1, 2 * GROW + 1), np.uint8)) > 0

    def taken(self, oid):
        """Indices of the object's Gaussians: over its grown cells above CUT, plus the large ones reaching into it."""
        m = self.mask(oid)
        idx = np.flatnonzero(self.inside)
        idx = idx[m[self.cij[idx, 1], self.cij[idx, 0]] & (self.gz[idx] > CUT)]
        o = self.objs[oid]
        smax = np.exp(self.g["scales"].max(1))
        x0, y0, x1, y1 = o["box"]
        r = 2 * smax
        P, gz = self.P, self.gz
        haze = (smax > HAZE_M) & (P[:, 0] > x0 - r) & (P[:, 0] < x1 + r) & (P[:, 1] > y0 - r) & (P[:, 1] < y1 + r) \
            & (gz + r > CUT) & (gz - r < o["height"] + 1.0)
        return np.union1d(idx, np.flatnonzero(haze))

    def rasters(self):
        """Per cell of the crown grid: anything standing (> GROUND_TOP), ground Gaussians, their mean colour, semantic
        feature (reduced) and roughness: patchfill's descriptors."""
        if self.R is not None:
            return self.R
        H, W = self.crown.shape
        c = self.cij[:, 1] * W + self.cij[:, 0]
        ok = self.inside & (self.op > 0.2)
        gz = self.gz
        obj = np.bincount(c[ok & (gz > GROUND_TOP)], minlength=W * H) > 0
        gm = ok & (gz >= -GROUND_TOP) & (gz <= GROUND_TOP)
        n = np.bincount(c[gm], minlength=W * H).astype(np.float32)
        nn = np.maximum(n, 1)
        col = np.stack([np.bincount(c[gm], weights=self.g["sh"][gm, 0, k], minlength=W * H) for k in range(3)], 1)
        zz = np.bincount(c[gm], weights=gz[gm] ** 2, minlength=W * H) / nn
        zm = np.bincount(c[gm], weights=gz[gm], minlength=W * H) / nn
        feat = None
        fp = os.path.join(self.project, semantic.DIR, "features.npz")
        if os.path.exists(fp):
            f = semantic.Features(self.project).f
            feat = np.stack([np.bincount(c[gm], weights=f[gm, k], minlength=W * H) for k in range(f.shape[1])], 1)
            feat = (feat / nn[:, None]).reshape(H, W, -1)
        self.R = {"W": W, "H": H, "lo": self.lo, "cell": self.cell, "obj": obj.reshape(H, W), "n": n.reshape(H, W),
                  "col": (col / nn[:, None]).reshape(H, W, 3), "feat": feat,
                  "rough": np.sqrt(np.maximum(zz - zm ** 2, 0)).astype(np.float32).reshape(H, W)}
        return self.R

    def blurry(self):
        """(H, W) bool: cells the detail map rates below GOOD_SOURCE (or did not see); none without a current map."""
        H, W = self.crown.shape
        path = os.path.join(_dir(self.project), "detail.npz")
        if not os.path.exists(path):
            return np.zeros((H, W), bool)
        m = np.load(path)
        if list(m["splat"]) != _key(base(self.project)):
            return np.zeros((H, W), bool)
        k = int(round(float(m["cell"]) / self.cell))
        big = np.repeat(np.repeat(~(np.nan_to_num(m["ratio"], nan=0.0) >= GOOD_SOURCE), k, 0), k, 1)
        out = np.ones((H, W), bool)
        h, w = min(H, big.shape[0]), min(W, big.shape[1])
        out[:h, :w] = big[:h, :w]
        return out

    def ground_hole(self, e):
        """(H, W) bool: a ground edit's cells: its weak area (ground_labels at MAP_CELL) or its circle."""
        H, W = self.crown.shape
        if e.get("circle"):
            x, y, r = e["circle"]
            yy, xx = np.mgrid[0:H, 0:W]
            cx, cy = self.lo[0] + (xx + 0.5) * self.cell, self.lo[1] + (yy + 0.5) * self.cell
            return np.hypot(cx - x, cy - y) <= r
        lab = np.load(os.path.join(_dir(self.project), "ground_labels.npy"))
        k = int(round(MAP_CELL / self.cell))
        big = np.repeat(np.repeat(lab == e["area"], k, 0), k, 1)
        out = np.zeros((H, W), bool)
        h, w = min(H, big.shape[0]), min(W, big.shape[1])
        out[:h, :w] = big[:h, :w]
        return out

    def fill(self, hole, avoid, old_top=CUT):
        """(copied arrays, old ground indices, stats): the hole's ground from PatchMatch sources; the hole's own
        Gaussians up to old_top above the ground go."""
        from modelforge import patchfill
        R = self.rasters()
        W, H, cell = R["W"], R["H"], self.cell
        off, cost = patchfill.fill(R, hole, avoid | self.blurry())
        gm = (self.gz >= -GROUND_TOP) & (self.gz <= GROUND_TOP) & self.inside \
            & (np.exp(self.g["scales"].max(1)) < MAX_COPY_M)
        gi = np.flatnonzero(gm)
        gc = self.cij[gi, 1] * W + self.cij[gi, 0]
        order = np.argsort(gc)
        gi, gc = gi[order], gc[order]
        hy, hx = np.nonzero(hole & np.isfinite(cost))
        sc = (hy + off[hy, hx, 0]) * W + (hx + off[hy, hx, 1])
        a, b = np.searchsorted(gc, sc), np.searchsorted(gc, sc, "right")
        src = np.concatenate([gi[i:j] for i, j in zip(a, b)]) if len(a) else np.zeros(0, np.int64)
        shift = np.repeat(np.stack([off[hy, hx, 1], off[hy, hx, 0]], 1) * cell, b - a, axis=0)
        q = {k: v[src].copy() for k, v in self.g.items()}
        x, y = self.P[src, 0] - shift[:, 0], self.P[src, 1] - shift[:, 1]
        q["means"] = np.stack([x, y, self.ground(x, y) + self.gz[src]], 1).astype(np.float32)
        ring = (cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~hole & (R["n"] > 0) & ~R["obj"]
        if len(src) and ring.any():
            q["sh"][:, 0] += 0.5 * (R["col"][ring].mean(0) - q["sh"][:, 0].mean(0))
        old = np.flatnonzero(gm & (self.gz <= old_top))
        old = old[hole[self.cij[old, 1], self.cij[old, 0]]]
        d = np.hypot(off[hy, hx, 0], off[hy, hx, 1]) * cell
        return q, old, {"cells": int(hole.sum()), "gaussians": int(len(src)),
                        "source_distance_m": round(float(np.median(d)), 1) if len(d) else None}

    def donor(self, target, donor, taken_target):
        """The donor's Gaussians fitted to the target (see the module doc)."""
        t, d = self.objs[target], self.objs[donor]
        di = self.taken(donor)
        di = di[self.gz[di] > CUT]
        s = t["height"] / d["height"]
        q = {k: v[di].copy() for k, v in self.g.items()}
        x = (self.P[di, 0] - d["x"]) * s + t["x"]
        y = (self.P[di, 1] - d["y"]) * s + t["y"]
        q["means"] = np.stack([x, y, self.ground(x, y) + self.gz[di] * s], 1).astype(np.float32)
        q["scales"] = q["scales"] + np.float32(np.log(s))
        rt = np.clip(self.gz[taken_target] / t["height"], 0, 0.999)
        rd = np.clip(self.gz[di] / d["height"], 0, 0.999)
        for b in range(BANDS):
            mt, md = (rt * BANDS).astype(int) == b, (rd * BANDS).astype(int) == b
            if mt.sum() >= 10 and md.sum() >= 10:
                q["sh"][md, 0] += COLOUR * (self.g["sh"][taken_target[mt], 0].mean(0) - q["sh"][md, 0].mean(0))
        return q, s


def _build(project, eds, on_line=print):
    """The base splat with the edits applied: arrays (load_splat layout) and per-edit stats."""
    sc = _Scene(project)
    holes = {e["id"]: (sc.ground_hole(e) if e.get("kind") == "ground" else sc.mask(e["object"]))
             for e in eds if e.get("fill", True)}
    avoid = np.zeros_like(sc.crown, bool)
    for h in holes.values():
        avoid |= h
    drop, add, stats = [], [], []
    for e in eds:
        if e.get("kind") == "ground":  # texture repair: the area's ground layer is replaced
            q, old, fs = sc.fill(holes[e["id"]], avoid, old_top=GROUND_TOP)
            drop.append(old)
            add.append(q)
            stats.append({"id": e["id"], "removed": int(len(old)), "fill": fs})
            on_line(f"replace: edit {e['id']} (ground): {stats[-1]}")
            continue
        taken = sc.taken(e["object"])
        drop.append(taken)
        st = {"id": e["id"], "removed": int(len(taken))}
        if e.get("fill", True):
            q, old, fs = sc.fill(holes[e["id"]], avoid)
            drop.append(old)
            add.append(q)
            st["fill"] = fs
        if e.get("donor") is not None:
            q, s = sc.donor(e["object"], e["donor"], taken)
            add.append(q)
            st.update(donor_gaussians=int(len(q["means"])), scale=round(float(s), 3))
        stats.append(st)
        on_line(f"replace: edit {e['id']} (object {e['object']}"
                + (f" -> {e['donor']}" if e.get("donor") is not None else "") + f"): {st}")
    keep = np.ones(len(sc.P), bool)
    if drop:
        keep[np.concatenate(drop)] = False
    return {k: np.concatenate([v[keep]] + [q[k] for q in add]) for k, v in sc.g.items()}, stats


def apply(project, on_line=print):
    """Write the project's splat = the base with all edits applied (no edits: the base itself, the backup dropped)."""
    from modelforge import difix
    live, backup = splat_mesh.splat_path(project), os.path.join(project, BACKUP)
    eds = edits(project)
    if not eds:
        if os.path.exists(backup):
            os.replace(backup, live)
        st = os.path.join(_dir(project), "state.json")
        if os.path.exists(st):
            os.remove(st)
        return []
    if not os.path.exists(backup):
        shutil.copy2(live, backup)  # keeps size + mtime: the objects and the semantic features still match it
    arrays, stats = _build(project, eds, on_line)
    tmp = live + ".replace.part"
    difix.write_splat(None, tmp, arrays)
    json.dump({"written": _key(tmp), "edits": stats}, open(os.path.join(_dir(project), "state.json"), "w"))
    os.replace(tmp, live)
    return stats


def add(project, obj, donor=None, fill=True, on_line=print):
    """Append an edit (remove obj, fill its ground, place donor) and apply all. Returns the edit."""
    eds = edits(project)
    if any(e.get("object") == obj for e in eds):
        raise RuntimeError(f"object {obj} is already edited (undo that edit first)")
    e = {"id": max([x["id"] for x in eds], default=-1) + 1, "object": int(obj),
         "donor": None if donor is None else int(donor), "fill": bool(fill), "time": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(eds + [e], open(os.path.join(_dir(project), "edits.json"), "w"), indent=1)
    apply(project, on_line)
    return e


def add_many(project, items, on_line=print):
    """Append several edits and apply once: (obj, donor or None, fill) for objects (already edited ones skipped), or
    dicts {"kind": "ground", "area": k} / {"kind": "ground", "circle": [x, y, r]} for texture repair."""
    eds = edits(project)
    done = {e["object"] for e in eds if "object" in e}
    areas = {e.get("area") for e in eds if e.get("kind") == "ground" and "area" in e}
    nxt = max([x["id"] for x in eds], default=-1) + 1
    for it in items:
        now = time.strftime("%Y-%m-%d %H:%M")
        if isinstance(it, dict):
            if it.get("area") is not None and it["area"] in areas:
                continue
            eds.append({"id": nxt, "kind": "ground", "fill": True, "time": now,
                        **{k: it[k] for k in ("area", "circle") if it.get(k) is not None}})
        else:
            obj, donor, fill = it
            if obj in done:
                continue
            eds.append({"id": nxt, "object": int(obj), "donor": None if donor is None else int(donor),
                        "fill": bool(fill), "time": now})
            done.add(obj)
        nxt += 1
    json.dump(eds, open(os.path.join(_dir(project), "edits.json"), "w"), indent=1)
    apply(project, on_line)
    return eds


def undo(project, edit_id=None, on_line=print):
    """Drop one edit (default: the last; "all": every one) and apply the rest."""
    eds = edits(project)
    if edit_id == "all":
        eds = []
    elif eds:
        eds = [e for e in eds if e["id"] != (eds[-1]["id"] if edit_id is None else edit_id)]
    json.dump(eds, open(os.path.join(_dir(project), "edits.json"), "w"), indent=1)
    apply(project, on_line)
    return eds

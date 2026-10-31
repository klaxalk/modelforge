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
DONOR_SHARP, CLEAN = 0.45, 0.7  # donors: nearly as sharp as their photos, mostly vegetation over their footprint
DONOR_SCALE, DONOR_SHAPE, DONOR_MAX_M = (0.5, 2.0), 0.6, 2.0
DONOR_PHOTOS = 10  # a donor is seen by at least this many photos (berlin: donors rated from 2-5 photos of one side
# were fog from anywhere else); seen from all round ranks higher (`sides`)
SIDE_CENTRE, SIDE_MAX_M = 0.5, 80.0  # an object's photos: within this share of the half-frame of the centre, this near  # a donor is scaled to the target's height within these; proportions
MAP_CELL, MAP_PHOTOS = 2.0, 300  # m; photos the detail map renders (spread over the flight)
TEMPLATE_R, TEMPLATE_VALID = (3.0, 8.0), 0.9  # automatic templates: m radius, the share of open detailed cells
TEMPLATE_W = {"colour": 0.5, "texture": 0.5, "semantic": 2.0}  # their say (per group): kladno_dira's road (g35)
# got a dirt yard 185 m away when photo colour + texture counted as much as the semantics; with the semantics leading
# the same road 12 m on (the semantics carry what the ground is; colour and texture refine)
AREA_SMOOTH_M, HAZE_TOP_M = 1.0, 6.0  # weak areas' outlines rounded at this scale; fog cleared up to this height
FEATHER_M, BLEND = 1.0, 0.7  # the fill's transition: crossfade band beyond the hole (m), colour blending strength
BLEND_WIN, BLEND_MAX = 2, 1.0  # cells around a border cell its colours are taken over; the largest correction (SH 0)
GOOD_SOURCE, MAX_COPY_M = 0.6, 1.0  # fill sources keep at least this share of the photos' detail (when the detail
# map exists: else similar but equally blurry ground was copied, no better); copied Gaussians are smaller than this
# (larger ones near the ground are glare and needles: bright blobs in the fill)
WEAK_GROUND, MIN_GROUND_M2, MIN_WEIGHT = 0.3, 8.0, 0.5
WEAK_PCT, GOOD_PCT, AREA_TILE_M, AREAS_VERSION = 20, 60, 20.0, 3  # (detail_thresholds); weak areas are cut into pieces of at most
# this square (a road and the grass beside it get their own templates; the fill's work stays bounded)  # a weak ground area: detail kept below this, at least this
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
    sides(project, objs)
    np.savez(os.path.join(_dir(project), "objects.npz"), crown=crown.reshape(H, W), lo=lo, cell=CELL,
             ground_grid=ground.g, ground_origin=np.array([ground.x0, ground.y0, ground.cell]))
    json.dump({"splat": _key(src), "queries": queries, "objects": objs}, open(os.path.join(_dir(project),
                                                                                           "objects.json"), "w"))
    on_line(f"replace: {len(objs)} objects ({', '.join(queries)}) in {time.time() - t0:.0f} s")
    return objs


def sides(project, objs):
    """Per object its photographed side (in place): "side_az" = the azimuth (rad, ENU) of the mean horizontal
    direction from it to the photos that have it within SIDE_CENTRE of their frame's centre (weighted by 1 /
    distance, at most SIDE_MAX_M away), "side_len" = how one-sided (0: seen from all round, 1: from one side only),
    "side_photos". A tree in a splat is good only from where it was photographed: a donor turned so its side faces the
    target's keeps its good look there (berlin, 2026-10-31: donors seen from the targets' cameras' angles were fog)."""
    t = json.load(open(os.path.join(project, "transforms.json")))
    M = np.array([np.array(f["transform_matrix"], float) for f in t["frames"]])
    C = M[:, :3, 3]
    R = M[:, :3, :3] @ np.diag([1, -1, -1])  # camera-to-world, OpenCV axes
    f0 = t["frames"][0]
    tx = 0.5 * float(f0.get("w", t.get("w"))) / float(f0.get("fl_x", t.get("fl_x")))
    ty = 0.5 * float(f0.get("h", t.get("h"))) / float(f0.get("fl_y", t.get("fl_y")))
    for o in objs:
        at = np.array([o["x"], o["y"], o["ground_z"] + 0.5 * o["height"]])
        v = at - C
        rel = np.einsum("nji,nj->ni", R, v)
        z = rel[:, 2]
        d = np.linalg.norm(v, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = (z > o["height"]) & (np.abs(rel[:, 0] / z) < SIDE_CENTRE * tx) & (np.abs(rel[:, 1] / z) < SIDE_CENTRE * ty)
        ok &= d < SIDE_MAX_M
        h = -v[ok, :2]  # from the object towards the camera
        n = np.linalg.norm(h, axis=1)
        keep = n > 1e-6
        if not keep.any():
            o.update(side_az=0.0, side_len=0.0, side_photos=0)
            continue
        w = 1.0 / d[ok][keep]
        u = (h[keep] / n[keep, None] * w[:, None]).sum(0) / w.sum()
        o.update(side_az=round(float(np.arctan2(u[1], u[0])), 4), side_len=round(float(np.linalg.norm(u)), 3),
                 side_photos=int(keep.sum()))
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
    extra = {k: m[k] for k in ("photo_rgb", "photo_tex") if k in m.files}
    np.savez(out, ratio=m["ratio"], weight=m["weight"], lo=lo, cell=MAP_CELL, splat=np.array(_key(base(project))),
             **extra)
    os.remove(spec["out"])
    areas = ground_areas(project)
    on_line(f"replace: detail map of {np.isfinite(m['ratio']).sum():,} cells; {len(areas)} weak ground areas")
    return areas


def cover_mesh(project):
    """The mesh that says where measured ground is: the LiDAR mesh, else the SfM mesh's measured part (mesh_trimmed:
    without its patched holes, which are interpolated), else its mesh; None without one."""
    for sub, name in (("lidarmesh", "mesh.ply"), ("sfmmesh", "mesh_trimmed.ply"), ("sfmmesh", "mesh.ply")):
        p = os.path.join(project, sub, name)
        if os.path.exists(p):
            return p
    return None


def mesh_cover(project, lo, cell, shape):
    """(H, W) bool on the grid (lo, cell, shape): cells with mesh triangles over them (their centres), small gaps
    closed; all True without a mesh. Weak ground is proposed, and templates taken, only there (the user asked,
    2026-10-31: over the LiDAR mesh, else the SfM mesh). Cached in replace/mesh_cover.npz per mesh file and grid."""
    path = cover_mesh(project)
    H, W = shape
    if path is None:
        return np.ones((H, W), bool)
    key = json.dumps([os.path.relpath(path, project), _key(path), [float(lo[0]), float(lo[1])], float(cell), [H, W]])
    cache = os.path.join(_dir(project), "mesh_cover.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if str(c["key"]) == key:
            return c["cover"]
    from modelforge import sfm_mesh
    v, faces = sfm_mesh.read_mesh(path)
    V = np.stack([v["x"], v["y"]], 1).astype(np.float64)
    cxy = V[faces].mean(1)
    j = ((cxy[:, 0] - lo[0]) // cell).astype(np.int64)
    i = ((cxy[:, 1] - lo[1]) // cell).astype(np.int64)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    cover = np.zeros((H, W), bool)
    cover[i[ok], j[ok]] = True
    cover = cv2.morphologyEx(cover.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    np.savez(cache, cover=cover, key=np.array(key))
    return cover


def detail_thresholds(ratio, weight):
    """(weak, good) detail thresholds of a project's map: the lower of WEAK_GROUND and its WEAK_PCT percentile, the
    lower of GOOD_SOURCE and its GOOD_PCT percentile (cells seen with MIN_WEIGHT). Relative: kladno_dira's map has a
    median of 0.33 (berlin 0.62): with the fixed values half of it was "weak" (one 10 ha area) and 19 % "good"."""
    ok = np.isfinite(ratio) & (weight >= MIN_WEIGHT)
    if not ok.any():
        return WEAK_GROUND, GOOD_SOURCE
    r = ratio[ok]
    return min(WEAK_GROUND, float(np.percentile(r, WEAK_PCT))), min(GOOD_SOURCE, float(np.percentile(r, GOOD_PCT)))


def ground_areas(project):
    """Weak ground areas from the detail map, over the measured mesh only (`mesh_cover`): connected cells keeping
    less than the weak threshold (`detail_thresholds`) of the photos' detail, cut into pieces of AREA_TILE_M,
    seen well enough (MIN_WEIGHT), with no object on them (the objects' footprints grown by a cell), at least
    MIN_GROUND_M2 large -> replace/ground.json [{id, x, y, radius, area_m2, detail, box}] + ground_labels.npy."""
    path = os.path.join(_dir(project), "detail.npz")
    if not os.path.exists(path):
        return []
    cached = os.path.join(_dir(project), "ground.json")
    mesh = cover_mesh(project)
    if os.path.exists(cached) and os.path.getmtime(cached) >= max(
            os.path.getmtime(path), os.path.getmtime(os.path.join(_dir(project), "objects.npz")),
            os.path.getmtime(mesh) if mesh else 0):
        got = _json(cached, {})
        if isinstance(got, dict) and got.get("version") == AREAS_VERSION:
            return got["areas"]
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
    weak_thr, _ = detail_thresholds(ratio, weight)
    weak = (np.nan_to_num(ratio, nan=1.0) < weak_thr) & mesh_cover(project, lo, cell, ratio.shape)
    weak &= (weight >= MIN_WEIGHT) & ~occ
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    tile = max(1, int(round(AREA_TILE_M / cell)))  # large areas in pieces of at most AREA_TILE_M square
    yy, xx = np.mgrid[0:H, 0:W]
    tid = (yy // tile) * (W // tile + 1) + xx // tile
    n0, lab0 = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    big_comp = np.bincount(lab0.ravel(), minlength=n0) * cell * cell > AREA_TILE_M ** 2  # only those are cut
    key = np.where(lab0 > 0, lab0.astype(np.int64) * (tid.max() + 1) + np.where(big_comp[lab0], tid, 0), 0)
    uniq, lab = np.unique(key, return_inverse=True)
    lab = lab.reshape(H, W).astype(np.int32)  # 0 = not weak (key 0 sorts first)
    n = len(uniq)
    stats = np.zeros((n, 5), np.int64)
    stats[:, cv2.CC_STAT_AREA] = np.bincount(lab.ravel(), minlength=n)
    cent = np.stack([np.bincount(lab.ravel(), weights=xx.ravel(), minlength=n),
                     np.bincount(lab.ravel(), weights=yy.ravel(), minlength=n)], 1) / np.maximum(
        stats[:, cv2.CC_STAT_AREA], 1)[:, None]
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
    json.dump({"version": AREAS_VERSION, "areas": areas}, open(os.path.join(_dir(project), "ground.json"), "w"))
    return areas


def suggest_targets(objs):
    """Objects to replace automatically: rated, rendered much blurrier than their photos, mostly vegetation."""
    return [o for o in objs if o.get("sharp") is not None and o.get("photos", 0) >= 2 and o["sharp"] < TARGET_SHARP
            and o["veg_share"] >= TARGET_VEG]


def donors(objs, target, used=(), exclude=()):
    """Donor candidates for target, best first: rated as sharp as DONOR_SHARP of their photos, mostly vegetation
    (CLEAN), any size from DONOR_SCALE[0] to DONOR_SCALE[1] times the target's height, roughly its proportions
    (height / width within DONOR_SHAPE in log); ranked by sharpness (capped at 1), less for enlarging (every Gaussian
    grows with it) and for unlike proportions; ones used already after the others. (Until 2026-10-31: within
    0.6-1.67 x, closely alike, denser than the median, ranked by Gaussian size: it hid berlin's good trees, big
    healthy ones use fewer, larger Gaussians, and chose hedge fragments.) At least DONOR_PHOTOS photos; seen from
    all round (`sides`: side_len low) ranks higher: a splat's tree is good only where it was photographed from."""
    def score(o):
        s = target["height"] / o["height"]
        shape = abs(np.log(o["height"] / o["radius"]) - np.log(target["height"] / target["radius"]))
        return min(o["sharp"], 1.0) - 0.5 * max(0.0, np.log(s)) - 0.3 * shape + 0.5 * (1 - o.get("side_len", 1.0))
    ok = [o for o in objs if o["id"] != target["id"] and o["id"] not in exclude and o.get("photos", 0) >= 2
          and o.get("side_photos", DONOR_PHOTOS) >= DONOR_PHOTOS
          and (o.get("sharp") or 0) >= DONOR_SHARP and o["veg_share"] >= CLEAN
          and DONOR_SCALE[0] <= target["height"] / o["height"] <= DONOR_SCALE[1]
          and abs(np.log(o["height"] / o["radius"]) - np.log(target["height"] / target["radius"])) < DONOR_SHAPE]
    ok.sort(key=lambda o: (o["id"] in used, -score(o)))
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
        if any("side_az" not in o for o in self.objs.values()):  # found before sides existed
            sides(project, list(self.objs.values()))
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

    def covered(self):
        """(H, W) bool: cells over the measured mesh (`mesh_cover`)."""
        if getattr(self, "_covered", None) is None:
            self._covered = mesh_cover(self.project, self.lo, self.cell, self.crown.shape)
        return self._covered

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
        _, good = detail_thresholds(m["ratio"], m["weight"])
        big = np.repeat(np.repeat(~(np.nan_to_num(m["ratio"], nan=0.0) >= good), k, 0), k, 1)
        out = np.ones((H, W), bool)
        h, w = min(H, big.shape[0]), min(W, big.shape[1])
        out[:h, :w] = big[:h, :w]
        return out

    def disc(self, x, y, r):
        """(H, W) bool: the cells within r of (x, y)."""
        H, W = self.crown.shape
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = self.lo[0] + (xx + 0.5) * self.cell, self.lo[1] + (yy + 0.5) * self.cell
        return np.hypot(cx - x, cy - y) <= r

    def polygon(self, pts):
        """(H, W) bool: the cells whose centres lie inside the polygon pts ([[x, y], ...], ENU)."""
        H, W = self.crown.shape
        q = (np.asarray(pts, float) - self.lo) / self.cell - 0.5  # cell centre coordinates (column, row)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(q * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        return m > 0

    def region(self, shape):
        """(H, W) bool of a shape: [x, y, r] (a disc) or {"polygon": [[x, y], ...]}."""
        if isinstance(shape, dict):
            return self.polygon(shape["polygon"])
        return self.disc(*shape)

    def ground_hole(self, e):
        """(H, W) bool: a ground edit's cells: its weak area (ground_labels at MAP_CELL) or its circle."""
        H, W = self.crown.shape
        if e.get("polygon"):
            return self.polygon(e["polygon"])
        if e.get("circle"):
            return self.disc(*e["circle"])
        lab = np.load(os.path.join(_dir(self.project), "ground_labels.npy"))
        k = int(round(MAP_CELL / self.cell))
        big = np.repeat(np.repeat(lab == e["area"], k, 0), k, 1).astype(np.float32)
        # rounded: the 2 m map cells gave staircase outlines and square corners (the most visible seam on kladno_dira)
        big = cv2.GaussianBlur(big, (0, 0), AREA_SMOOTH_M / self.cell) > 0.5
        out = np.zeros((H, W), bool)
        h, w = min(H, big.shape[0]), min(W, big.shape[1])
        out[:h, :w] = big[:h, :w]
        return out

    def suggest_template(self, hole, avoid):
        """The most similar detailed patch for a ground hole: (x, y, r) of a disc of the hole's size (r within
        TEMPLATE_R) whose cells are open, detailed (>= the good threshold where the detail map exists) and over the
        measured mesh (at least TEMPLATE_VALID of them), and whose mean description is nearest the hole's own: what the
        photos show (the detail map's photo colour and texture energy at two scales: blur-free, a smeared road's photos
        still show a road) and the semantic feature, each normalized over the site; without the photo arrays (maps from
        before 2026-10-31) the splat's colour instead and no texture. Distance does not count (the user: similarity in
        colour, texture and semantics is what matters). Judged by the hole's own content, not by its rim: on
        kladno_dira a road's rim of grass made the fill copy another ground. None if no place qualifies."""
        from modelforge import patchfill
        R = self.rasters()
        D = patchfill.descriptors(R)
        have = R["n"] > 0
        H, W = hole.shape
        groups = []  # (name, (H, W, n)): normalized per number, then weighted per group (weighting first is undone)
        photo = self.photo_maps()
        if photo is not None:
            groups += [("colour", photo[0]), ("texture", photo[1])]
        else:
            groups.append(("colour", D[..., :3]))
        if R.get("feat") is not None:
            groups.append(("semantic", D[..., 3:11]))
        A = np.concatenate([g for _, g in groups], -1).astype(np.float32)
        known = np.isfinite(A).all(-1) & have
        A[~known] = 0
        mu, sd = A[known].mean(0), A[known].std(0) + 1e-6
        A = np.where(known[..., None], (A - mu) / sd, 0).astype(np.float32)
        w = np.concatenate([np.full(g.shape[-1], TEMPLATE_W.get(n, 1.0) / np.sqrt(g.shape[-1])) for n, g in groups])
        A *= w.astype(np.float32)  # each group the same say, whatever its number of dimensions
        k = A.shape[-1]
        inner = hole & known
        if not inner.any():
            return None
        want = A[inner].mean(0)
        rad = float(np.clip(np.sqrt(hole.sum() / np.pi) * self.cell, *TEMPLATE_R))
        rc = max(1, int(round(rad / self.cell)))
        yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
        K = (np.hypot(yy, xx) <= rc).astype(np.float32)
        near_hole = cv2.dilate(hole.astype(np.uint8), np.ones((2 * rc + 1, 2 * rc + 1), np.uint8)) > 0
        ok = (known & ~R["obj"] & ~avoid & ~self.blurry() & ~near_hole & self.covered()).astype(np.float32)
        cnt = cv2.filter2D(ok, -1, K, borderType=cv2.BORDER_CONSTANT)
        frac = cnt / K.sum()
        mean = np.stack([cv2.filter2D(A[..., j] * ok, -1, K, borderType=cv2.BORDER_CONSTANT) for j in range(k)], -1)
        mean /= np.maximum(cnt, 1e-6)[..., None]
        cost = np.linalg.norm(mean - want, axis=-1)
        cost[frac < TEMPLATE_VALID] = np.inf
        if not np.isfinite(cost).any():
            return None
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        return [float(self.lo[0] + (j + 0.5) * self.cell), float(self.lo[1] + (i + 0.5) * self.cell), rad]

    def photo_maps(self):
        """The detail map's photo colour (H, W, 3) and texture (H, W, 2) on the fine grid, NaN where no photo saw
        it; None without them (a map from before they were recorded) or without a current map."""
        path = os.path.join(_dir(self.project), "detail.npz")
        if not os.path.exists(path):
            return None
        m = np.load(path)
        if "photo_rgb" not in m.files or list(m["splat"]) != _key(base(self.project)):
            return None
        H, W = self.crown.shape
        k = int(round(float(m["cell"]) / self.cell))
        out = []
        for a in (m["photo_rgb"], m["photo_tex"]):
            big = np.repeat(np.repeat(a, k, 0), k, 1)
            full = np.full((H, W, a.shape[-1]), np.nan, np.float32)
            h, w = min(H, big.shape[0]), min(W, big.shape[1])
            full[:h, :w] = big[:h, :w]
            out.append(full)
        return out

    def fill(self, hole, avoid, old_top=CUT, template=None, feather=None, blend=None):
        """(copied arrays, old indices to drop, stats): the hole's ground from PatchMatch sources; the hole's own
        Gaussians up to old_top above the ground go (any size: the smear is often large ones). template (x, y, r) or
        {"polygon"}: sources only there (the user's choice of good ground; kladno_dira, 2026-10-31: a road was filled
        with a different ground's texture).

        The transition (the user asked to make it seamless, 2026-10-31):
        - feather (m, FEATHER_M): the copy reaches that far beyond the hole, crossfading by opacity: the copied
          Gaussians fade out and the original ground layer fades in across the band (no hard edge);
        - blend (BLEND): gradient-domain colour blending (as Poisson image editing / a healing brush): the colour
          difference between the surroundings and the copy along the edge is spread smoothly over the patch (a
          harmonic fill) and added to the copy, so the edge matches and the copied texture stays."""
        from modelforge import patchfill
        feather = FEATHER_M if feather is None else feather
        blend = BLEND if blend is None else blend
        R = self.rasters()
        W, H, cell = R["W"], R["H"], self.cell
        have = R["n"] > 0
        # the band beyond the hole the copy crossfades over: ground cells (nothing standing) within `feather`
        dist_out = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 3) * cell
        band = ~hole & (dist_out <= feather) & have & ~R["obj"] if feather > 0 else np.zeros_like(hole)
        region = hole | band
        off, cost = patchfill.fill(R, region, avoid | (~self.region(template) if template else self.blurry()))
        gm = (self.gz >= -GROUND_TOP) & (self.gz <= GROUND_TOP) & self.inside
        small = np.exp(self.g["scales"].max(1)) < MAX_COPY_M
        gi = np.flatnonzero(gm & small)
        gc = self.cij[gi, 1] * W + self.cij[gi, 0]
        order = np.argsort(gc)
        gi, gc = gi[order], gc[order]
        hy, hx = np.nonzero(region & np.isfinite(cost))
        sc = (hy + off[hy, hx, 0]) * W + (hx + off[hy, hx, 1])
        a, b = np.searchsorted(gc, sc), np.searchsorted(gc, sc, "right")
        src = np.concatenate([gi[i:j] for i, j in zip(a, b)]) if len(a) else np.zeros(0, np.int64)
        tcell = np.repeat(np.arange(len(hy)), b - a)  # each copied Gaussian's target cell (index into hy, hx)
        shift = np.stack([off[hy, hx, 1], off[hy, hx, 0]], 1)[tcell] * cell
        q = {k: v[src].copy() for k, v in self.g.items()}
        x, y = self.P[src, 0] - shift[:, 0], self.P[src, 1] - shift[:, 1]
        q["means"] = np.stack([x, y, self.ground(x, y) + self.gz[src]], 1).astype(np.float32)
        # weights: 1 in the hole, falling to 0 across the band
        wcell = np.where(hole[hy, hx], 1.0, np.clip(1.0 - dist_out[hy, hx] / max(feather, 1e-6), 0.0, 1.0))
        stats = {"cells": int(hole.sum()), "band_cells": int(band.sum()), "gaussians": int(len(src)),
                 "template": bool(template)}
        if blend > 0 and len(src):
            corr = self._blend_field(R, region, hy, hx, tcell, q["sh"][:, 0])
            if corr is not None:
                q["sh"][:, 0] += blend * corr[tcell]
                stats["blend_mean"] = round(float(np.abs(corr).mean()), 3)
        elif len(src) and not template:  # the former global pull (blend 0)
            ring = (cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~hole & have & ~R["obj"]
            if ring.any():
                q["sh"][:, 0] += 0.5 * (R["col"][ring].mean(0) - q["sh"][:, 0].mean(0))
        old = np.flatnonzero(gm & (self.gz <= old_top))
        old = old[hole[self.cij[old, 1], self.cij[old, 0]]]
        if band.any():  # crossfade: copied Gaussians in the band weighted w, the original ground layer there 1 - w
            w = wcell[tcell].astype(np.float32)
            op = 1 / (1 + np.exp(-q["opacities"].reshape(-1)))
            q["opacities"] = _logit(op * w).reshape(q["opacities"].shape)
            keep = w > 0.02
            q = {k: v[keep] for k, v in q.items()}
            ob = np.flatnonzero(gm)
            ob = ob[band[self.cij[ob, 1], self.cij[ob, 0]]]
            wo = 1.0 - np.clip(1.0 - dist_out[self.cij[ob, 1], self.cij[ob, 0]] / feather, 0.0, 1.0)
            faded = {k: v[ob].copy() for k, v in self.g.items()}
            op = 1 / (1 + np.exp(-faded["opacities"].reshape(-1)))
            faded["opacities"] = _logit(op * wo).reshape(faded["opacities"].shape)
            q = {k: np.concatenate([q[k], faded[k]]) for k in q}
            old = np.concatenate([old, ob])
        d = np.hypot(off[hy, hx, 0], off[hy, hx, 1]) * cell
        stats["source_distance_m"] = round(float(np.median(d)), 1) if len(d) else None
        return q, old, stats

    def haze_over(self, hole):
        """Indices of the large Gaussians (> HAZE_M) hanging over a ground hole up to HAZE_TOP_M above the ground: the
        fog that makes weak ground weak (a fill under it looked like a carpet laid under fog). Weak areas exclude the
        objects' footprints, so trees and bushes are not touched."""
        idx = np.flatnonzero(self.inside)
        idx = idx[hole[self.cij[idx, 1], self.cij[idx, 0]]]
        big = np.exp(self.g["scales"][idx].max(1)) > HAZE_M
        return idx[big & (self.gz[idx] > GROUND_TOP) & (self.gz[idx] < HAZE_TOP_M)]

    def _blend_field(self, R, region, hy, hx, tcell, sh0):
        """(len(hy), 3) colour correction per region cell: on the region's border cells the difference between the
        surroundings (the mean colour of their outside neighbours' ground) and the copy there, harmonically
        interpolated over the region (Laplace with those boundary values: Poisson blending of a constant-gradient
        source). None if the region has no measurable border."""
        H, W = region.shape
        n = len(hy)
        cnt = np.bincount(tcell, minlength=n).astype(np.float64)
        F = np.stack([np.bincount(tcell, weights=sh0[:, c], minlength=n) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        outside = ~region & (R["n"] > 0) & ~R["obj"]
        col = np.where(outside[..., None], R["col"], 0).astype(np.float32)
        k = np.ones((2 * BLEND_WIN + 1, 2 * BLEND_WIN + 1), np.float32)  # wide: one dark neighbour must not decide
        s_out = cv2.filter2D(col, -1, k, borderType=cv2.BORDER_CONSTANT)
        n_out = cv2.filter2D(outside.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        # the copy's colour near each border cell, over the same window
        Fg = np.zeros((H, W, 3), np.float32)
        Fn = np.zeros((H, W), np.float32)
        Fg[hy, hx], Fn[hy, hx] = F * (cnt > 0)[:, None], (cnt > 0)
        s_in = cv2.filter2D(Fg, -1, k, borderType=cv2.BORDER_CONSTANT)
        n_in = cv2.filter2D(Fn, -1, k, borderType=cv2.BORDER_CONSTANT)
        one = np.ones((3, 3), np.float32)
        border = (cv2.filter2D(outside.astype(np.float32), -1, one, borderType=cv2.BORDER_CONSTANT)[hy, hx] > 0) \
            & (n_out[hy, hx] >= 3) & (n_in[hy, hx] >= 3)
        if not border.any():
            return None
        y0, y1, x0, x1 = hy.min(), hy.max() + 1, hx.min(), hx.max() + 1
        grid = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        known = np.zeros((y1 - y0, x1 - x0), bool)
        d = s_out[hy, hx] / np.maximum(n_out[hy, hx], 1)[:, None] - s_in[hy, hx] / np.maximum(n_in[hy, hx], 1)[:, None]
        grid[hy[border] - y0, hx[border] - x0] = np.clip(d[border], -BLEND_MAX, BLEND_MAX)
        known[hy[border] - y0, hx[border] - x0] = True
        # smoothed along the border (normalized convolution): no spot of the rim dominates
        sm = cv2.GaussianBlur(grid * known[..., None], (0, 0), BLEND_WIN)
        wn = cv2.GaussianBlur(known.astype(np.float32), (0, 0), BLEND_WIN)
        grid = np.where(known[..., None], sm / np.maximum(wn, 1e-6)[..., None], 0).astype(np.float32)
        field = np.stack([rig.fill_holes(grid[..., c], known) for c in range(3)], -1)
        return field[hy - y0, hx - x0]

    def donor(self, target, donor, taken_target, turn=False):
        """The donor's Gaussians fitted to the target (see the module doc); turn: turned about the vertical so the
        donor's photographed side faces the target's (`sides`; the sunlit side moves with it)."""
        t, d = self.objs[target], self.objs[donor]
        # its own Gaussians: centred over its footprint up to 0.5 m above its top, at most DONOR_MAX_M large; not the
        # haze around it nor floaters above (removal takes those, copying must not: white clouds), nor sky-coloured
        # ones (sky seen through the crown in its own photos: white specks once moved); large ones stay (a healthy
        # crown is mostly Gaussians > 0.5 m: without them the copy was hollow twigs)
        m = self.mask(donor)
        di = np.flatnonzero(self.inside)
        di = di[m[self.cij[di, 1], self.cij[di, 0]]]
        smax = np.exp(self.g["scales"][di].max(1))
        rgb = np.clip(self.g["sh"][di, 0] * 0.28209479177387814 + 0.5, 0, 1)
        sky = (rgb.min(1) > 0.6) & (rgb.max(1) - rgb.min(1) < 0.12)
        di = di[(self.gz[di] > CUT) & (self.gz[di] < d["height"] + 0.5) & (smax < DONOR_MAX_M) & ~sky]
        s = t["height"] / d["height"]
        q = {k: v[di].copy() for k, v in self.g.items()}
        dx, dy = (self.P[di, 0] - d["x"]) * s, (self.P[di, 1] - d["y"]) * s
        if turn:
            a = float(t["side_az"] - d["side_az"])
            dx, dy = _turn(q, a, dx, dy)
        x, y = dx + t["x"], dy + t["y"]
        q["means"] = np.stack([x, y, self.ground(x, y) + self.gz[di] * s], 1).astype(np.float32)
        q["scales"] = q["scales"] + np.float32(np.log(s))
        rt = np.clip(self.gz[taken_target] / t["height"], 0, 0.999)
        rd = np.clip(self.gz[di] / d["height"], 0, 0.999)
        for b in range(BANDS):
            mt, md = (rt * BANDS).astype(int) == b, (rd * BANDS).astype(int) == b
            if mt.sum() >= 10 and md.sum() >= 10:
                q["sh"][md, 0] += COLOUR * (self.g["sh"][taken_target[mt], 0].mean(0) - q["sh"][md, 0].mean(0))
        return q, s


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def _turn(q, a, dx, dy):
    """Turn arrays q (load_splat layout; in place: orientations and SH bands) and their horizontal offsets (dx, dy)
    by a radians about the vertical; returns the turned offsets."""
    from modelforge.export import splat_to_enu
    ca, sa = np.cos(a), np.sin(a)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    q["quats"] = splat_to_enu.quat_mul(splat_to_enu.quat_from_rot(Rz), q["quats"].astype(np.float64)).astype(np.float32)
    K = q["sh"].shape[1] - 1
    if K:  # c'(d) = c(Rz^T d): the view-dependent colour turns with it
        X = splat_to_enu.sh_rotation(Rz.T, int(round(np.sqrt(K + 1))) - 1)
        q["sh"][:, 1:, :] = (q["sh"][:, 1:, :].transpose(0, 2, 1) @ X.T).transpose(0, 2, 1).astype(np.float32)
    return ca * dx - sa * dy, sa * dx + ca * dy


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
            tmpl = e.get("source")
            if tmpl == "auto":  # the most similar detailed patch (recorded in the stats)
                tmpl = sc.suggest_template(holes[e["id"]], avoid)
            q, old, fs = sc.fill(holes[e["id"]], avoid, old_top=GROUND_TOP, template=tmpl)
            haze = sc.haze_over(holes[e["id"]])  # the fog above weak ground goes too
            old = np.union1d(old, haze)
            fs["haze_removed"] = int(len(haze))
            if isinstance(tmpl, dict):
                fs["template_polygon"] = tmpl["polygon"]
            elif tmpl is not None:
                fs["template_at"] = [round(v, 2) for v in tmpl]
            drop.append(old)
            add.append(q)
            stats.append({"id": e["id"], "removed": int(len(old)), "fill": fs})
            on_line(f"replace: edit {e['id']} (ground): {stats[-1]}")
            continue
        taken = sc.taken(e["object"])
        drop.append(taken)
        st = {"id": e["id"], "removed": int(len(taken))}
        if e.get("fill", True):
            src = e.get("source")  # under an object the hole's content is the object: "auto" judges by the rim then
            q, old, fs = sc.fill(holes[e["id"]], avoid, template=None if src == "auto" else src)
            drop.append(old)
            add.append(q)
            st["fill"] = fs
        if e.get("donor") is not None:
            q, s = sc.donor(e["object"], e["donor"], taken, turn=e.get("turn", False))
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


def add(project, obj, donor=None, fill=True, on_line=print, turn=False):
    """Append an edit (remove obj, fill its ground, place donor) and apply all. Returns the edit."""
    eds = edits(project)
    if any(e.get("object") == obj for e in eds):
        raise RuntimeError(f"object {obj} is already edited (undo that edit first)")
    e = {"id": max([x["id"] for x in eds], default=-1) + 1, "object": int(obj),
         "donor": None if donor is None else int(donor), "fill": bool(fill), "turn": bool(turn),
         "time": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(eds + [e], open(os.path.join(_dir(project), "edits.json"), "w"), indent=1)
    apply(project, on_line)
    return e


def add_many(project, items, on_line=print):
    """Append several edits and apply once: (obj, donor or None, fill[, turn[, source]]) for objects (already edited
    ones skipped), or dicts {"area": k} / {"circle": [x, y, r]} / {"polygon": [[x, y], ...]} for texture repair;
    "source": [x, y, r] or {"polygon": [[x, y], ...]} (either kind)
    takes the ground fill only from that circle (a template of good ground), "auto" (ground repairs) from the most
    similar detailed patch (`_Scene.suggest_template`)."""
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
                        **{k: it[k] for k in ("area", "circle", "polygon", "source") if it.get(k) is not None}})
        else:
            obj, donor, fill, *rest = it  # (obj, donor, fill[, turn[, source (x, y, r) of the ground fill]])
            if obj in done:
                continue
            eds.append({"id": nxt, "object": int(obj), "donor": None if donor is None else int(donor),
                        "fill": bool(fill), "turn": bool(rest[0]) if rest else False, "time": now,
                        **({"source": rest[1] if rest[1] == "auto" or isinstance(rest[1], dict) else list(rest[1])}
                           if len(rest) > 1 and rest[1] is not None else {})})
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

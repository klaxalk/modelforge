"""Gaussian splat of a large flight in tiles: one splatfacto run per part of the scene, cropped and merged.

Why: one splat over thousands of photos trains badly. On dira2k (2000 photos over ~14 ha) it stayed blurry at
~17 dB held-out, while worlds of up to ~800 photos (karlak 471: 23 dB) and a 150-photo patch of dira2k train
well: the run sees each photo only a few times, and one densification schedule serves the whole area. All tiles
are in the project's ENU frame (one global SfM), so they join by concatenation (as VastGaussian / CityGS).

  partition  cameras and sparse points (transforms.json, sparse_pc.ply) -> cells on the ground plane (ENU x/y):
             recursive splits of the cameras' look-at points (the median of the sparse points each camera sees)
             along the longer axis into ceil(n / photos_per_tile) parts, each cut placed where it splits the least
             (sparse points near the line weighted by the photos seeing them, in its worst stretch) within
             SIZE_SLACK of the balanced position. An orbited object is many points seen by many photos: a plain
             median cut went through one on kladno_dira, and its far-side views missed one of the halves. A cell's
             photos: those looking into it plus every photo with >= MIN_VISIBLE of its seen points inside it,
             or >= MIN_IN_CELL inside and its look-at point within half the margin of the cell (far-side
             views of objects near the edge; on kladno_dira +12 % photos, a cell + margin share rule +37 %).
             Outer cells extend to infinity.
  train      PROJECT/tiles/tile_KK/: transforms.json with the tile's frames, sparse_pc.ply cropped to the cell
             + margin, images/ and images_N/ as links to the project's; then splat.run with its own plan
             (splat_plan.suggest on the tile: steps from its photo count), one tile after the other. With an
             rounds (ROUND_STEPS, default 4000): every tile advances that many steps per round (it ends with a
             checkpoint, as on Stop, and continues from it next round), and the tiles are merged after each
             round, so the whole scene improves evenly and shows early.
  clean      in ENU, only at the sides shared with another tile: keep the Gaussians whose centre lies inside the
             cell (the margin is the neighbour's; it is trained by fewer views) and whose 3 sigma reach crosses
             no shared side by more than the margin. Outer tiles keep their view away from the scene (distant
             terrain, horizon); FAR x the scene's size beyond it drops absurd outliers. No height cut.
  merge      the cleaned tiles concatenated -> PROJECT/splat/splat_enu.ply, as an untiled run writes it.
  setup      partition + the tiles' datasets alone (tiles.json, all pending), before any training; a run reuses
             them while the photos per tile and the cameras are the same. `only`: one tile trains again from
             scratch with new settings, the others keep their splats, all are merged (semantic features then
             recompute that tile only: modelforge/semantic.py).

Progress, Stop / Pause / Resume: per tile, through splat.run; PROJECT/tiles/tiles.json records the cells and the
finished tiles, PROJECT/splat/resume.json ({"tiles": true, ...}) continues with the stopped tile, then the rest.
The live preview shows the finished tiles (cleaned) plus the training one (cropped to its cell).

Layout "covisibility" (modelforge/tile_regions.py; Splat tab "Tile layout", CLI --tile-layout): the photos are
partitioned by the points they see together and each tile's region is a label map (tiles/regions.npz) whose seams
avoid what sticks out of the ground; cropping, the clean (distance to the nearest seam instead of the shared sides)
and the photos joining a tile use the region instead of the rectangle. Default "rectangles".
"""

import json
import math
import os
import time

import numpy as np

from modelforge import jobs, splat, splat_plan

TILE_ABOVE = 800  # training images; more -> tiles (worlds up to ~800 photos train well in one piece)
PHOTOS_PER_TILE = 500  # cameras looking into a cell (its photos also include oblique views from neighbours)
MIN_VISIBLE = 0.25  # a photo joins a tile when this share of its seen sparse points lies in the cell, or ...
MIN_IN_CELL = 0.10  # ... this share in the cell when looking at most half the margin beyond its edge
SIZE_SLACK = 0.5  # a cut may move this share of the smaller side's photos off the balanced position (tiles of
#                   up to ~2x the photos still train well, the user found)
BALANCE = 0.3  # weight of being off balance (1 at the slack's end) against the cut's cost in median costs
GAP_WIDTH = 0.02  # of the look-ats' extent: half the band of sparse points a candidate cut is charged for
ROUND_STEPS = 4000  # steps every tile advances per round, merged after each (0: one tile after the other)
FAR = 2.0  # of the scene box's longer side beyond it: the clean's outlier limit (outer tiles keep their background)
MARGIN = 0.25  # of the cell's longer side: initial points and the clean's reach beyond the cell
MAX_POINTS = 50_000  # sparse points sampled for the partition
DIR = "tiles"
STATE = os.path.join(DIR, "tiles.json")
REGIONS = os.path.join(DIR, "regions.npz")  # layout "covisibility": the tiles' label map
SKIP = os.path.join(DIR, "skip.json")  # tiles left out of training and the merge (apart from tiles.json, which a
#                                        running job rewrites: a click in the View tab must survive that)
LAYOUTS = ("rectangles", "covisibility")
LIVE = os.path.join("splat", "live_tiles.npz")
IDENTITY = os.path.join(DIR, "identity_transforms.json")  # the live preview's arrays are already ENU
BAND0 = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
         "f_dc_0", "f_dc_1", "f_dc_2", "opacity")


def wanted(train_images, tiles="auto"):
    """Train in tiles? tiles: auto (above TILE_ABOVE training images) / on / off."""
    return tiles == "on" or (tiles in (None, "", "auto") and train_images > TILE_ABOVE)


def _seen(C, R, fx, w, h, xyz):
    """Per camera the indices of the sampled points it sees (in front, in frame, nearer than
    splat_plan.FAR_X_MEDIAN x its median depth), as in splat_plan.survey."""
    out = []
    for c, r in zip(C, R):
        p = (xyz - c) @ r
        z = p[:, 2]
        front = np.flatnonzero(z > 1e-9)
        u = fx * p[front, 0] / z[front] + w / 2
        v = fx * p[front, 1] / z[front] + h / 2
        idx = front[(u >= 0) & (u < w) & (v >= 0) & (v < h)]
        if len(idx) >= 5:
            idx = idx[z[idx] < splat_plan.FAR_X_MEDIAN * np.median(z[idx])]
        out.append(idx.astype(np.int32))
    return out


def _cut(looks, axis, frac, pts, views):
    """Where to cut along axis so that about frac of the cameras' look-at points lie below. Candidates are midpoints
    between neighbouring look-ats leaving frac -/+ SIZE_SLACK x the smaller side below. A candidate's cost is what
    the cut would split: the sparse points (pts, n x 2) within GAP_WIDTH of the line, each weighted by the photos
    seeing it (views), in the worst stretch of 4 x GAP_WIDTH along the line (an orbited object is local: summed
    over the whole line it would drown in the ground). Score: cost / median cost + distance from balance as a
    share of the slack, so ground noise keeps cuts near balance and an object moves them."""
    x = np.sort(looks[:, axis])
    n = len(x)
    slack = SIZE_SLACK * min(frac, 1 - frac) * n
    k = np.arange(max(1, int(np.ceil(frac * n - slack))), min(n - 1, int(frac * n + slack)) + 1)
    if not len(k) or not len(pts):
        return float(np.median(x))
    c = (x[k - 1] + x[k]) / 2
    h = GAP_WIDTH * max(x[int(0.98 * (n - 1))] - x[int(0.02 * (n - 1))], 1e-9)
    order = np.argsort(pts[:, axis])
    px, py, pw = pts[order, axis], pts[order, 1 - axis], views[order]
    lo, hi = np.searchsorted(px, c - h), np.searchsorted(px, c + h)
    cost = np.zeros(len(c))
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            cost[i] = max(np.bincount(((py[a:b] - py.min()) / (4 * h) + off).astype(int), pw[a:b]).max()
                          for off in (0, 0.5))
    off = np.abs(k - frac * n) / max(slack, 1.0)  # 1 at the slack's end
    best = np.argmin(cost / max(np.median(cost), 1e-9) + BALANCE * off)  # an object costs many medians
    return float(c[best])


def _split(ids, looks, cell, parts, out, pts, views):
    """Recursive split of the cameras `ids` (looking at `looks`) inside `cell` (x0, y0, x1, y1) into `parts`
    cells of about equal photo counts, cut where little is seen (_cut; pts, views: the sampled sparse points)."""
    if parts <= 1:
        out.append((cell, ids))
        return
    lo, hi = np.percentile(looks[ids], 2, axis=0), np.percentile(looks[ids], 98, axis=0)
    axis = int(np.argmax(hi - lo))
    here = _inside(pts, cell)
    m = _cut(looks[ids], axis, (parts // 2) / parts, pts[here], views[here])
    left = looks[ids, axis] < m
    if left.all() or not left.any():  # all looking at one spot: cannot split further
        out.append((cell, ids))
        return
    a, b = list(cell), list(cell)
    a[axis + 2], b[axis] = m, m
    _split(ids[left], looks, tuple(a), parts // 2, out, pts, views)
    _split(ids[~left], looks, tuple(b), parts - parts // 2, out, pts, views)


def _inside(xy, cell, pad=0.0):
    x0, y0, x1, y1 = cell
    return (xy[:, 0] >= x0 - pad) & (xy[:, 0] < x1 + pad) & (xy[:, 1] >= y0 - pad) & (xy[:, 1] < y1 + pad)


def partition(project, photos=PHOTOS_PER_TILE, max_points=MAX_POINTS, layout="rectangles"):
    """Cells and their photos: [{"cell": [x0, y0, x1, y1] (±inf outside), "margin": m, "frames": [frame indices],
    "core": cameras looking into it}], plus the scene box [x0, y0, x1, y1] (sampled points 1-99 % + margin) and,
    for layout "covisibility", the Regions (each tile then has "region": its label; "cell" is the region's box)."""
    C, R, fx, w, h, _ = splat_plan._cameras(project)
    xyz = splat_plan._points(project, max_points)
    if xyz is None or len(xyz) < 10:
        raise RuntimeError("sparse_pc.ply has too few points to divide the scene into tiles")
    seen = _seen(C, R, fx, w, h, xyz)
    looks = np.array([np.median(xyz[s, :2], axis=0) if len(s) else c[:2] for s, c in zip(seen, C)])
    views = np.bincount(np.concatenate(seen), minlength=len(xyz)).astype(float) if seen else np.zeros(len(xyz))
    leaves, regions = [], None
    parts = math.ceil(len(C) / max(1, int(photos)))
    if layout == "covisibility":
        from modelforge import tile_regions
        names = [os.path.basename(f["file_path"]) for f in
                 json.load(open(os.path.join(project, "transforms.json")))["frames"]]
        part, k, regions = tile_regions.partition(project, names, C, parts)
        leaves = [(regions.bbox(t), np.flatnonzero(part == t)) for t in range(k)]
        point_cell = regions.label(xyz[:, :2])
    else:
        _split(np.arange(len(C)), looks, (-np.inf, -np.inf, np.inf, np.inf), parts, leaves, xyz[:, :2], views)
        point_cell = np.full(len(xyz), -1)
        for k, (cell, _) in enumerate(leaves):
            point_cell[_inside(xyz[:, :2], cell)] = k
    lo, hi = np.percentile(xyz[:, :2], 1, axis=0), np.percentile(xyz[:, :2], 99, axis=0)
    share = np.zeros((len(C), len(leaves)))
    for i, s in enumerate(seen):
        if len(s):
            share[i] = np.bincount(point_cell[s][point_cell[s] >= 0], minlength=len(leaves)) / len(s)
    tiles, pad = [], 0.0
    for k, (cell, ids) in enumerate(leaves):
        finite = np.array([max(cell[0], lo[0]), max(cell[1], lo[1]), min(cell[2], hi[0]), min(cell[3], hi[1])])
        margin = MARGIN * float(max(finite[2] - finite[0], finite[3] - finite[1], 1e-6))
        if regions is not None:  # an irregular region's box overstates it: its size from the area
            margin = MARGIN * math.sqrt(max(regions.area(k, lo, hi), 1e-6))
        pad = max(pad, margin)
        # looking just across the edge: far-side views of edge objects
        near = regions.inside(looks, k, margin / 2) if regions is not None else _inside(looks, cell, margin / 2)
        joins = (share[:, k] >= MIN_VISIBLE) | ((share[:, k] >= MIN_IN_CELL) & near)
        frames = sorted(set(ids.tolist()) | set(np.flatnonzero(joins).tolist()))
        tiles.append({"cell": [float(v) for v in cell], "margin": margin, "frames": frames, "core": len(ids)}
                     | ({"region": k} if regions is not None else {}))
    box = [float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad)]
    return (tiles, box, regions) if layout == "covisibility" else (tiles, box)


_REGIONS = {}


def regions(project):
    """The set-up's Regions (layout "covisibility"), cached by file time; None without."""
    path = os.path.join(project, REGIONS)
    if not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if key not in _REGIONS:
        from modelforge import tile_regions
        _REGIONS.clear()
        _REGIONS[key] = tile_regions.Regions.load(path)
    return _REGIONS[key]


def skipped(project):
    """The tiles (indices) left out of training and the merge."""
    try:
        return set(int(k) for k in json.load(open(os.path.join(project, SKIP))))
    except (OSError, ValueError, TypeError):
        return set()


def set_skipped(project, k, skip=True):
    """Leave tile k out (skip) or take it back; a running tiled run sees it before its next tile."""
    s = skipped(project)
    s = s | {int(k)} if skip else s - {int(k)}
    path = os.path.join(project, SKIP)
    tmp = path + ".tmp"
    json.dump(sorted(s), open(tmp, "w"))
    os.replace(tmp, path)
    return s


def in_tile(project, tile, xy, pad=0.0):
    """Positions (n, 2) inside the tile's region (layout covisibility) or cell, grown by pad."""
    if "region" in tile:
        return regions(project).inside(xy, tile["region"], pad)
    return _inside(xy, _cell(tile["cell"]), pad)


def clean_tile(project, v, tile, box):
    """clean() for a tile of either layout."""
    if "region" in tile:
        return clean(v, None, tile["margin"], box, region=(regions(project), tile["region"]))
    return clean(v, _cell(tile["cell"]), tile["margin"], box)


def overlay(project, photos=PHOTOS_PER_TILE, layout="rectangles"):
    """The tile cells as boxes for the View tab: [{"tile": k, "rect": [x0, y0, x1, y1] (clipped to the scene box),
    "z": [z0, z1] (the cell's sparse points, 2-98 %), "status", "photos"}], and whether they are only planned:
    the run's tiles.json when there is one, else a partition with these photos per tile (not saved)."""
    reg = None
    try:
        state = json.load(open(os.path.join(project, STATE)))
        tiles, box, planned = [dict(t, cell=_cell(t["cell"])) for t in state["tiles"]], state["box"], False
        reg = regions(project) if any("region" in t for t in tiles) else None
    except (OSError, ValueError, KeyError):
        if layout == "covisibility":
            tiles, box, reg = partition(project, photos, layout=layout)
        else:
            tiles, box = partition(project, photos)
        tiles = [dict(t, cell=_cell(t["cell"])) for t in tiles]
        planned = True
    xyz = splat_plan._points(project, MAX_POINTS)
    skip = skipped(project)
    out = []
    for k, t in enumerate(tiles):
        c = t["cell"]
        rect = [max(c[0], box[0]), max(c[1], box[1]), min(c[2], box[2]), min(c[3], box[3])]
        inside = (reg.inside(xyz[:, :2], t["region"]) if "region" in t else _inside(xyz[:, :2], c)) \
            if xyz is not None else np.zeros(0, bool)
        z = xyz[inside, 2] if xyz is not None else np.zeros(0)
        zr = np.percentile(z, [2, 98]).tolist() if len(z) else [0.0, 1.0]
        item = {"tile": k, "rect": rect, "z": zr, "status": t.get("status") or ("planned" if planned else "pending"),
                "photos": len(t["frames"]), "skipped": not planned and k in skip}
        if "region" in t:  # the region's outline, clipped to the scene box
            item["outlines"] = [np.clip(o, box[:2], box[2:]).tolist() for o in reg.outlines(t["region"])]
            if xyz is not None and inside.any():
                item["center"] = np.median(xyz[inside, :2], axis=0).tolist()
        out.append(item)
    return out, planned


def _json_cell(cell):  # json has no inf
    return [None if not math.isfinite(v) else v for v in cell]


def _cell(cell):
    return tuple(-np.inf if v is None and i < 2 else np.inf if v is None else v for i, v in enumerate(cell))


def tile_dir(project, k):
    return os.path.join(project, DIR, f"tile_{k:02d}")


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def write_tile(project, k, tile, downscale=1):
    """PROJECT/tiles/tile_KK: a nerfstudio dataset of the tile's frames (images linked, points cropped)."""
    import plyfile
    out = tile_dir(project, k)
    os.makedirs(out, exist_ok=True)
    t = json.load(open(os.path.join(project, "transforms.json")))
    t["frames"] = [t["frames"][i] for i in tile["frames"]]
    ply = plyfile.PlyData.read(os.path.join(project, t.get("ply_file_path", "sparse_pc.ply")))
    v = ply["vertex"].data
    xy = np.stack([np.asarray(v["x"], float), np.asarray(v["y"], float)], 1)
    keep = in_tile(project, tile, xy, tile["margin"])
    plyfile.PlyData([plyfile.PlyElement.describe(v[keep], "vertex")], text=False).write(
        os.path.join(out, "sparse_pc.ply"))
    t["ply_file_path"] = "sparse_pc.ply"
    if "lidar_ply_file_path" in t:  # the whole cloud: the plugin projects what each camera sees
        t["lidar_ply_file_path"] = os.path.relpath(os.path.join(project, t["lidar_ply_file_path"]), out)
    json.dump(t, open(os.path.join(out, "transforms.json"), "w"), indent=1)
    _link(os.path.join(project, "images"), os.path.join(out, "images"))
    if downscale > 1:
        os.makedirs(os.path.join(project, f"images_{downscale}"), exist_ok=True)
        _link(os.path.join(project, f"images_{downscale}"), os.path.join(out, f"images_{downscale}"))
    return out, int(keep.sum())


def clean(v, cell, margin, box, region=None):
    """Mask of the Gaussians (structured ENU vertices) a tile keeps. Only the sides shared with another tile cut
    (an outer tile keeps its view away from the scene: distant terrain, horizon): the centre lies in the cell (its
    outer sides are at infinity) and the 3 sigma reach crosses no shared side by more than the margin (reach <=
    distance to the nearest shared side + margin), so a large blob far out stays but none spreads over a
    neighbour. Besides, a far limit drops absurd outliers: FAR x the scene box's size beyond it. No height cut.
    region = (Regions, label) instead of the cell: inside that region, the reach against the nearest seam."""
    x, y = np.asarray(v["x"], float), np.asarray(v["y"], float)
    xy = np.stack([x, y], 1)
    if region is not None:
        reg, k = region
        inside, shared = reg.inside(xy, k), reg.to_seam(xy, k)
    else:
        x0, y0, x1, y1 = cell
        inside = _inside(xy, cell)
        shared = np.full(len(x), np.inf)  # distance to the nearest side shared with another tile
        for side, d in ((x0, x - x0), (y0, y - y0), (x1, x1 - x), (y1, y1 - y)):
            if np.isfinite(side):
                shared = np.minimum(shared, d)
    reach = 3 * np.exp(np.max(np.stack([np.asarray(v[f"scale_{i}"], float) for i in range(3)], 1), axis=1))
    far = FAR * max(box[2] - box[0], box[3] - box[1])
    return inside & (reach <= shared + margin) & _inside(xy, box, far)


def _read(path):
    import plyfile
    return plyfile.PlyData.read(path)["vertex"].data


def _write(v, path):
    import plyfile
    tmp = path + ".part.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")], text=False).write(tmp)
    os.replace(tmp, path)


def merge(project, state):
    """Concatenate the tiles' cleaned splats -> PROJECT/splat/splat_enu.ply (a tile training again keeps its last
    one until replaced). Records them in state["merged"] (and tiles.json). Returns the count."""
    skip = skipped(project)
    have = [k for k in range(len(state["tiles"])) if k not in skip and
            os.path.exists(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply"))]
    parts = [_read(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")) for k in have]
    if not parts:
        return 0
    names = parts[0].dtype.names
    if any(p.dtype.names != names for p in parts):
        raise RuntimeError("the tiles' splats have different fields (SH degree?); they cannot be merged")
    v = np.concatenate([p.astype(parts[0].dtype) for p in parts])
    os.makedirs(os.path.join(project, "splat"), exist_ok=True)
    path = os.path.join(project, "splat", "splat_enu.ply")
    _write(v, path)
    st = os.stat(path)
    state["merged"] = {"tiles": have, "counts": [len(p) for p in parts], "size": st.st_size,
                       "mtime_ns": st.st_mtime_ns}
    json.dump(state, open(os.path.join(project, STATE), "w"), indent=1)
    return len(v)


def merged_tiles(project):
    """The tiles (indices, in merge order) splat/splat_enu.ply is the merge of, or None (not a tiled splat, or
    changed since). Runs before the merge record: the cleaned tiles add up to it and are not newer."""
    state = load(project)
    path = os.path.join(project, "splat", "splat_enu.ply")
    if state is None or not os.path.exists(path):
        return None
    st = os.stat(path)
    m = state.get("merged")
    if m:
        return m["tiles"] if (m["size"], m["mtime_ns"]) == (st.st_size, st.st_mtime_ns) else None
    cleans = [(k, os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")) for k in range(len(state["tiles"]))]
    skip = skipped(project)
    cleans = [(k, c) for k, c in cleans if os.path.exists(c) and k not in skip]
    if cleans and all(os.path.getmtime(c) <= st.st_mtime for _, c in cleans) and \
            sum(splat_plan.ply_count(c) for _, c in cleans) == splat_plan.ply_count(path):
        return [k for k, _ in cleans]
    return None


def _band0(v):
    out = np.zeros(len(v), dtype=[(k, "f4") for k in BAND0])
    for k in BAND0:
        out[k] = v[k]
    return out


def _write_live(project, parts, step):
    """The preview arrays (ENU, band 0) in ckpt_dump's format, with an identity transform."""
    v = np.concatenate(parts) if parts else np.zeros(0, dtype=[(k, "f4") for k in BAND0])
    path = os.path.join(project, LIVE)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # splat/ does not exist before the first merge
    tmp = path + ".tmp.npz"
    np.savez(tmp, step=np.int64(step), means=np.stack([v["x"], v["y"], v["z"]], 1),
             scales=np.stack([v[f"scale_{i}"] for i in range(3)], 1), quats=np.stack([v[f"rot_{i}"] for i in range(4)], 1),
             opacities=v["opacity"][:, None], features_dc=np.stack([v[f"f_dc_{i}"] for i in range(3)], 1))
    os.replace(tmp, path)
    ident = os.path.join(project, IDENTITY)
    if not os.path.exists(ident):
        os.makedirs(os.path.dirname(ident), exist_ok=True)
        json.dump({"transform": np.eye(4)[:3].tolist(), "scale": 1.0}, open(ident, "w"))
    return path, ident


def setup(project, photos=PHOTOS_PER_TILE, on_line=print, layout="rectangles"):
    """Divide the scene into tiles and write each tile's dataset (cameras, cropped points; links to the photos):
    tiles.json with every tile pending, nothing trained. Replaces an earlier set-up and its trained tiles (tiles/
    is removed); the merged splat/splat_enu.ply stays until a run merges again."""
    import shutil
    if layout not in LAYOUTS:
        raise ValueError(f"tile layout must be one of {LAYOUTS}")
    reg = None
    if layout == "covisibility":
        tiles, box, reg = partition(project, photos, layout=layout)
    else:
        tiles, box = partition(project, photos)
    shutil.rmtree(os.path.join(project, DIR), ignore_errors=True)
    if reg is not None:
        os.makedirs(os.path.join(project, DIR), exist_ok=True)
        reg.save(os.path.join(project, REGIONS))
    rf = os.path.join(project, splat.RESUME)
    if os.path.exists(rf) and json.load(open(rf)).get("tiles"):  # its tile is gone
        splat._remove(rf)
    state = {"photos": int(photos), "layout": layout, "box": box,
             "transforms_mtime": os.path.getmtime(os.path.join(project, "transforms.json")),
             "tiles": [dict(t, cell=_json_cell(t["cell"]), status="pending", step=0) for t in tiles]}
    for k, t in enumerate(state["tiles"]):
        t["points"] = write_tile(project, k, t)[1]
    os.makedirs(os.path.join(project, DIR), exist_ok=True)
    json.dump(state, open(os.path.join(project, STATE), "w"), indent=1)
    on_line(f"splat: {len(tiles)} {layout} tiles of {', '.join(str(len(t['frames'])) for t in tiles)} photos "
            f"(~{round(sum(t['core'] for t in tiles) / len(tiles))} looking into each cell, at most {photos}, "
            f"+ views from the neighbours)")
    return state


def load(project):
    """tiles.json, or None."""
    try:
        return json.load(open(os.path.join(project, STATE)))
    except (OSError, ValueError):
        return None


def current(project, state):
    """Whether a set-up (tiles.json) still fits the project's cameras (transforms.json unchanged since)."""
    t = os.path.join(project, "transforms.json")
    return state is not None and os.path.exists(t) and (
        "transforms_mtime" not in state or abs(os.path.getmtime(t) - state["transforms_mtime"]) < 1e-3)


def _reset(project, k, tile):
    """Tile k trains again from scratch; its cleaned splat stays in the merge until the new one replaces it."""
    import shutil
    tdir = tile_dir(project, k)
    shutil.rmtree(os.path.join(tdir, "nerfstudio"), ignore_errors=True)
    splat._remove(os.path.join(tdir, splat.RESUME))
    for key in ("schedule", "iterations", "gaussians", "train_seconds"):
        tile.pop(key, None)
    tile.update(status="pending", step=0)


def run(project, method="splatfacto", plan_args=None, extra=(), on_line=print, on_progress=None, job=None,
        cache="auto", preview_every=None, on_snapshot=None, on_prepare=None, on_tile=None, pause=None,
        lidar_depth=None, photos=PHOTOS_PER_TILE, resume=False, round_steps=ROUND_STEPS, only=None, depth_cloud=None,
        depth_dense=False, layout="rectangles", skip=None):
    """Train the project in tiles and merge them into splat/splat_enu.ply. Returns stats (report.json["splat"]).
    plan_args: splat_plan.suggest overrides (iterations, downscale, strategy, max_gaussians, camera_opt; auto
    values come from each tile). on_tile(k, n, target step, label): a tile starts (label "round r / R" or ""). on_progress(step, percent of the whole run). resume: continue a paused / stopped tiled run (its own settings, PROJECT/splat/resume.json).
    round_steps: every tile advances this many steps per round, from its checkpoint (step, optimizer and schedules
    carry on as in one run), and the tiles are merged after each round; 0 = one tile after the other.
    The tiles of `setup` are used when they fit (same photos per tile, same cameras), else it runs first.
    only: train just that tile (index) from scratch, in one go; the others keep their splats, and all are merged.
    skip: tile indices to leave out (written after a new set-up; None keeps tiles/skip.json as it is)."""
    from modelforge.export import splat_to_enu
    from modelforge.pipeline import Report
    job = job or jobs.Token()
    t0 = time.time()
    plan_args = dict(plan_args or {})
    state_path = os.path.join(project, STATE)
    if resume:
        state = json.load(open(state_path))
    else:
        splat._remove(os.path.join(project, splat.RESUME))
        down = plan_args.get("downscale")
        down = splat_plan.auto_downscale(json.load(open(os.path.join(project, "transforms.json")))["w"]) \
            if down in (None, 0, "", "auto") else int(down)
        state = load(project)
        fits = current(project, state)
        if only is not None:
            if not fits:
                raise RuntimeError("set up the tiles first (Splat tab → Set up tiles): "
                                   + ("none yet" if state is None else "the cameras changed since"))
            if not 0 <= only < len(state["tiles"]):
                raise RuntimeError(f"there is no tile {only + 1} (tiles 1–{len(state['tiles'])})")
        elif not fits or state.get("photos") != int(photos) or state.get("layout", "rectangles") != layout:
            state = setup(project, photos, on_line, layout)
        if skip is not None:
            for k in range(len(state["tiles"])):
                set_skipped(project, k, k in set(skip))
        targets = range(len(state["tiles"])) if only is None else [only]
        for k in targets:
            _reset(project, k, state["tiles"][k])
        state.update(method=method, plan_args=dict(plan_args, downscale=down), extra=list(extra), cache=cache,
                     lidar_depth=lidar_depth, depth_cloud=depth_cloud, depth_dense=bool(depth_dense),
                     round_steps=0 if only is not None else int(round_steps or 0),
                     only=only, round=1)
        json.dump(state, open(state_path, "w"), indent=1)
        if only is not None:
            on_line(f"splat: training tile {only + 1} of {len(state['tiles'])} only; the others keep their splats")
    method, extra, lidar_depth = state["method"], tuple(state["extra"]), state["lidar_depth"]
    depth_cloud = state.get("depth_cloud")  # the whole SfM mesh: each tile's photos see only their part
    depth_dense = state.get("depth_dense", False)
    plan_args, box, round_steps = state["plan_args"], state["box"], int(state.get("round_steps") or 0)
    down = int(plan_args["downscale"])
    splat.prepare_downscaled(project, down, on_line, job=job, on_count=on_prepare)
    n = len(state["tiles"])
    targets = [k for k in range(n) if k not in skipped(project)] if state.get("only") is None else [state["only"]]
    if not targets:
        raise RuntimeError("every tile is left out (View → Tile boundaries: click a tile's ball to take it back)")
    if len(targets) < n and state.get("only") is None:
        on_line(f"splat: tiles {', '.join(str(k + 1) for k in sorted(set(range(n)) - set(targets)))} left out")
    done_parts = {}  # tile -> band-0 arrays of its cleaned splat, for the live preview

    def save_state():
        json.dump(state, open(state_path, "w"), indent=1)

    for k, tile in enumerate(state["tiles"]):  # the trained tiles' datasets and plans first: the total of steps
        if k in targets and "schedule" not in tile:
            if tile.get("status") in (None, "pending", "training"):  # a trained tile's dataset stays as it was
                tile["points"] = write_tile(project, k, tile, down)[1]
            tile["schedule"] = splat_plan.suggest(splat_plan.survey(tile_dir(project, k)), **plan_args, tiles="off")
            tile["iterations"] = tile["schedule"]["iterations"]
        tile.setdefault("step", 0)
        if on_snapshot and os.path.exists(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")):
            done_parts[k] = _band0(_read(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")))
    save_state()
    total_steps = sum(state["tiles"][k]["iterations"] for k in targets)
    longest = max(state["tiles"][k]["iterations"] for k in targets)
    rounds = math.ceil(longest / round_steps) if round_steps else 1

    def live_wrapper(k, tile):
        def snap(step, npz, transforms):
            d = np.load(npz)
            m = len(d["means"])
            v = np.zeros(m, dtype=[(c, "f4") for c in BAND0])
            cols = np.c_[d["means"], d["scales"], d["quats"], d["features_dc"].reshape(m, -1)[:, :3],
                         d["opacities"].reshape(m)]
            for i, c in enumerate(BAND0):
                v[c] = cols[:, i]
            R, t, s = splat_to_enu.load_transform(transforms)
            v = splat_to_enu.transform_vertices(v, R, t, s)
            xy = np.stack([v["x"], v["y"]], 1).astype(float)
            v = v[in_tile(project, tile, xy)]  # its part from the last round is replaced by the training one
            skip = skipped(project)
            path, ident = _write_live(project, [done_parts[j] for j in sorted(done_parts) if j != k and j not in skip]
                                      + [v], step)
            on_snapshot(step, path, ident)
        return snap

    def train(k, tile, rnd):
        """Train tile k to the end of round rnd (its last step if that comes first), from its checkpoint if any."""
        jobs.check(job)
        tdir = tile_dir(project, k)
        info = splat.resume_info(tdir) if tile["step"] > 0 or tile.get("status") in ("paused", "partial") else None
        if tile["step"] > 0 and info is None:
            raise RuntimeError(f"tile {k + 1}: its checkpoint at step {tile['step']:,} is gone; start a new run")
        iters = tile["iterations"]
        stop_at = rnd * round_steps if round_steps and rnd * round_steps < iters else None
        target = stop_at or iters
        tile["status"] = "training"
        save_state()
        tile_cache = state["cache"]
        if tile_cache == "auto":
            tile_cache = splat.auto_cache(tdir, down)[0]
        label = f"round {rnd} / {rounds}" if rounds > 1 else ""
        on_line(f"splat: {label + ', ' if label else ''}tile {k + 1} / {n}: {len(tile['frames'])} photos, "
                f"to step {target:,} of {iters:,}, cache {tile_cache}"
                + (f" (resuming at step {info['step']:,})" if info else ""))
        if on_tile:
            on_tile(k, n, target, label)
        before = sum(state["tiles"][j]["step"] for j in targets if j != k)

        def progress(step, pct):
            if on_progress:
                on_progress(step, 100.0 * (before + step) / total_steps)

        common = dict(on_line=on_line, on_progress=progress, job=job, preview_every=preview_every, pause=pause,
                      on_snapshot=live_wrapper(k, tile) if on_snapshot else None, stop_at=stop_at)
        stopped = None
        try:
            if info:
                stats = splat.resume(tdir, **common)
            else:
                stats = splat.run(tdir, method, iters, extra, cache=tile_cache, downscale=down,
                                  lidar_depth=lidar_depth, schedule=tile["schedule"], depth_cloud=depth_cloud,
                                  depth_dense=depth_dense,
                                  **common)
        except splat.Paused:
            tile["status"] = "paused"
            tile["step"] = json.load(open(os.path.join(tdir, splat.RESUME)))["step"]
            save_state()
            _resume_file(project, state, k)
            raise
        except splat.StoppedPartial as e:
            stopped, tile["status"] = e, "partial"
            stats = Report(tdir).data.get("splat", {})
        except jobs.Cancelled:
            tile["status"] = "round" if tile["step"] else "pending"
            save_state()
            raise
        v = _read(os.path.join(tdir, "splat", "splat_enu.ply"))
        keep = clean_tile(project, v, tile, box)
        _write(v[keep], os.path.join(tdir, "splat", "clean_enu.ply"))
        tile.update(gaussians=int(len(v)), kept=int(keep.sum()), step=int(stats.get("stopped_at_step") or iters),
                    train_seconds=round((tile.get("train_seconds") or 0) + (stats.get("train_seconds") or 0), 1),
                    status="partial" if stopped else "round" if stop_at else "done")
        save_state()
        on_line(f"splat: tile {k + 1} / {n}: kept {int(keep.sum()):,} of {len(v):,} gaussians (inside its cell)")
        if on_snapshot:
            done_parts[k] = _band0(v[keep])
        if stopped:
            total = merge(project, state)
            _resume_file(project, state, k)
            _report(project, state, total, t0, stopped_at=k)
            raise splat.StoppedPartial(f"stopped in tile {k + 1} / {n}; the finished tiles and the partial one are "
                                       "merged into splat/splat_enu.ply, and Resume continues") from None

    rnd = int(state.get("round") or 1)
    def left_out(k):  # clicked away in the View tab meanwhile (an explicit `only` tile always trains)
        return state.get("only") is None and k in skipped(project)

    while any(state["tiles"][k].get("status") != "done" for k in targets if not left_out(k)):
        state["round"] = rnd
        save_state()
        for k in targets:
            if left_out(k):
                continue
            tile = state["tiles"][k]
            if tile.get("status") != "done" and tile["step"] < min(tile["iterations"], rnd * round_steps or
                                                                   tile["iterations"]):
                train(k, tile, rnd)
        if rnd >= rounds:
            break
        total = merge(project, state)
        on_line(f"splat: round {rnd} / {rounds}: all {len(targets)} tiles at step {min(rnd * round_steps, longest):,}, merged into "
                f"splat/splat_enu.ply ({total:,} gaussians)")
        rnd += 1
    on_line("splat: merging the tiles")
    total = merge(project, state)
    splat._remove(os.path.join(project, splat.RESUME))
    splat._remove(os.path.join(project, LIVE))
    return _report(project, state, total, t0)


def _resume_file(project, state, k):
    """PROJECT/splat/resume.json for a tiled run (splat.resume_info / resume read it)."""
    tdir = tile_dir(project, k)
    info = json.load(open(os.path.join(tdir, splat.RESUME)))
    os.makedirs(os.path.join(project, "splat"), exist_ok=True)
    json.dump({"tiles": True, "tile": k, "of": len(state["tiles"]), "method": state["method"],
               "iterations": info["iterations"], "downscale": int(state["plan_args"]["downscale"]),
               "step": info["step"], "load_dir": info["load_dir"],
               "transforms_mtime": os.path.getmtime(os.path.join(project, "transforms.json"))},
              open(os.path.join(project, splat.RESUME), "w"), indent=1)


def resume_info(project, info):
    """The tiled run's resume.json if it can continue (tiles.json present, transforms.json unchanged)."""
    t = os.path.join(project, "transforms.json")
    if not os.path.exists(os.path.join(project, STATE)) or not os.path.exists(t) or \
            abs(os.path.getmtime(t) - info.get("transforms_mtime", 0)) > 1e-3:
        return None
    return info


def _report(project, state, total, t0, stopped_at=None):
    from modelforge.pipeline import Report
    tiles = state["tiles"]
    stats = {"method": splat.ns_method(state["method"], state["lidar_depth"]), "tiles": len(tiles),
             "iterations": int(sum(t.get("iterations", 0) for t in tiles)),
             "downscale": int(state["plan_args"]["downscale"]), "splat": os.path.join(project, "splat", "splat_enu.ply"),
             "num_gaussians": int(total), "train_seconds": round(sum(t.get("train_seconds") or 0 for t in tiles), 1),
             "seconds": round(time.time() - t0, 1),
             "per_tile": [{k: t.get(k) for k in ("core", "points", "iterations", "gaussians", "kept", "train_seconds",
                                                 "status")} | {"photos": len(t["frames"])} for t in tiles]}
    if stopped_at is not None:
        stats["stopped_at_tile"] = stopped_at
    Report(project).set("splat", stats)
    return stats

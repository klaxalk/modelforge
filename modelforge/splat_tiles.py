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
  clean      in ENU: keep the Gaussians whose centre lies inside the cell (the margin is the neighbour's; it is
             trained by fewer views), drop those reaching beyond the margin (3 sigma; distant terrain and sky
             learned as large blobs) and those outside the scene box (points' range + margin).
  merge      the cleaned tiles concatenated -> PROJECT/splat/splat_enu.ply, as an untiled run writes it.

Progress, Stop / Pause / Resume: per tile, through splat.run; PROJECT/tiles/tiles.json records the cells and the
finished tiles, PROJECT/splat/resume.json ({"tiles": true, ...}) continues with the stopped tile, then the rest.
The live preview shows the finished tiles (cleaned) plus the training one (cropped to its cell).
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
MARGIN = 0.25  # of the cell's longer side: initial points and the clean's reach beyond the cell
MAX_POINTS = 50_000  # sparse points sampled for the partition
DIR = "tiles"
STATE = os.path.join(DIR, "tiles.json")
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


def partition(project, photos=PHOTOS_PER_TILE, max_points=MAX_POINTS):
    """Cells and their photos: [{"cell": [x0, y0, x1, y1] (±inf outside), "margin": m, "frames": [frame indices],
    "core": cameras looking into it}], plus the scene box [x0, y0, x1, y1] (sampled points 1-99 % + margin)."""
    C, R, fx, w, h, _ = splat_plan._cameras(project)
    xyz = splat_plan._points(project, max_points)
    if xyz is None or len(xyz) < 10:
        raise RuntimeError("sparse_pc.ply has too few points to divide the scene into tiles")
    seen = _seen(C, R, fx, w, h, xyz)
    looks = np.array([np.median(xyz[s, :2], axis=0) if len(s) else c[:2] for s, c in zip(seen, C)])
    views = np.bincount(np.concatenate(seen), minlength=len(xyz)).astype(float) if seen else np.zeros(len(xyz))
    leaves = []
    _split(np.arange(len(C)), looks, (-np.inf, -np.inf, np.inf, np.inf), math.ceil(len(C) / max(1, int(photos))),
           leaves, xyz[:, :2], views)
    lo, hi = np.percentile(xyz[:, :2], 1, axis=0), np.percentile(xyz[:, :2], 99, axis=0)
    point_cell = np.full(len(xyz), -1)
    for k, (cell, _) in enumerate(leaves):
        point_cell[_inside(xyz[:, :2], cell)] = k
    share = np.zeros((len(C), len(leaves)))
    for i, s in enumerate(seen):
        if len(s):
            share[i] = np.bincount(point_cell[s][point_cell[s] >= 0], minlength=len(leaves)) / len(s)
    tiles, pad = [], 0.0
    for k, (cell, ids) in enumerate(leaves):
        finite = np.array([max(cell[0], lo[0]), max(cell[1], lo[1]), min(cell[2], hi[0]), min(cell[3], hi[1])])
        margin = MARGIN * float(max(finite[2] - finite[0], finite[3] - finite[1], 1e-6))
        pad = max(pad, margin)
        near = _inside(looks, cell, margin / 2)  # looking just across the edge: far-side views of edge objects
        joins = (share[:, k] >= MIN_VISIBLE) | ((share[:, k] >= MIN_IN_CELL) & near)
        frames = sorted(set(ids.tolist()) | set(np.flatnonzero(joins).tolist()))
        tiles.append({"cell": [float(v) for v in cell], "margin": margin, "frames": frames, "core": len(ids)})
    box = [float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad)]
    return tiles, box


def overlay(project, photos=PHOTOS_PER_TILE):
    """The tile cells as boxes for the View tab: [{"tile": k, "rect": [x0, y0, x1, y1] (clipped to the scene box),
    "z": [z0, z1] (the cell's sparse points, 2-98 %), "status", "photos"}], and whether they are only planned:
    the run's tiles.json when there is one, else a partition with these photos per tile (not saved)."""
    try:
        state = json.load(open(os.path.join(project, STATE)))
        tiles, box, planned = [dict(t, cell=_cell(t["cell"])) for t in state["tiles"]], state["box"], False
    except (OSError, ValueError, KeyError):
        tiles, box = partition(project, photos)
        planned = True
    xyz = splat_plan._points(project, MAX_POINTS)
    out = []
    for k, t in enumerate(tiles):
        c = t["cell"]
        rect = [max(c[0], box[0]), max(c[1], box[1]), min(c[2], box[2]), min(c[3], box[3])]
        z = xyz[_inside(xyz[:, :2], c), 2] if xyz is not None else np.zeros(0)
        zr = np.percentile(z, [2, 98]).tolist() if len(z) else [0.0, 1.0]
        out.append({"tile": k, "rect": rect, "z": zr, "status": t.get("status") or ("planned" if planned else "pending"),
                    "photos": len(t["frames"])})
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
    keep = _inside(xy, _cell(tile["cell"]), tile["margin"])
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


def clean(v, cell, margin, box):
    """Mask of the Gaussians (structured ENU vertices) a tile keeps: centre in its cell and in the scene box,
    3 sigma within the margin."""
    xy = np.stack([np.asarray(v["x"], float), np.asarray(v["y"], float)], 1)
    keep = _inside(xy, cell) & _inside(xy, box)
    extent = 3 * np.exp(np.max(np.stack([np.asarray(v[f"scale_{i}"], float) for i in range(3)], 1), axis=1))
    return keep & (extent <= margin)


def _read(path):
    import plyfile
    return plyfile.PlyData.read(path)["vertex"].data


def _write(v, path):
    import plyfile
    tmp = path + ".part.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")], text=False).write(tmp)
    os.replace(tmp, path)


def merge(project, state):
    """Concatenate the cleaned tiles (done, a round's or partial) -> PROJECT/splat/splat_enu.ply. Returns its count."""
    parts = [_read(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply"))
             for k, t in enumerate(state["tiles"]) if t.get("status") in ("done", "partial", "round")]
    if not parts:
        return 0
    names = parts[0].dtype.names
    if any(p.dtype.names != names for p in parts):
        raise RuntimeError("the tiles' splats have different fields (SH degree?); they cannot be merged")
    v = np.concatenate([p.astype(parts[0].dtype) for p in parts])
    os.makedirs(os.path.join(project, "splat"), exist_ok=True)
    _write(v, os.path.join(project, "splat", "splat_enu.ply"))
    return len(v)


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


def run(project, method="splatfacto", plan_args=None, extra=(), on_line=print, on_progress=None, job=None,
        cache="auto", preview_every=None, on_snapshot=None, on_prepare=None, on_tile=None, pause=None,
        lidar_depth=None, photos=PHOTOS_PER_TILE, resume=False, round_steps=ROUND_STEPS):
    """Train the project in tiles and merge them into splat/splat_enu.ply. Returns stats (report.json["splat"]).
    plan_args: splat_plan.suggest overrides (iterations, downscale, strategy, max_gaussians, camera_opt; auto
    values come from each tile). on_tile(k, n, target step, label): a tile starts (label "round r / R" or ""). on_progress(step, percent of the whole run). resume: continue a paused / stopped tiled run (its own settings, PROJECT/splat/resume.json).
    round_steps: every tile advances this many steps per round, from its checkpoint (step, optimizer and schedules
    carry on as in one run), and the tiles are merged after each round; 0 = one tile after the other."""
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
        tiles, box = partition(project, photos)
        state = {"method": method, "plan_args": dict(plan_args, downscale=down), "extra": list(extra),
                 "cache": cache, "lidar_depth": lidar_depth, "photos": photos, "box": box,
                 "round_steps": int(round_steps or 0),
                 "tiles": [dict(t, cell=_json_cell(t["cell"])) for t in tiles]}
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        json.dump(state, open(state_path, "w"), indent=1)
        on_line(f"splat: {len(tiles)} tiles of {', '.join(str(len(t['frames'])) for t in tiles)} photos "
                f"(~{round(sum(t['core'] for t in tiles) / len(tiles))} looking into each cell, at most {photos}, "
                f"+ views from the neighbours)")
    method, extra, lidar_depth = state["method"], tuple(state["extra"]), state["lidar_depth"]
    plan_args, box, round_steps = state["plan_args"], state["box"], int(state.get("round_steps") or 0)
    down = int(plan_args["downscale"])
    splat.prepare_downscaled(project, down, on_line, job=job, on_count=on_prepare)
    n = len(state["tiles"])
    done_parts = {}  # tile -> band-0 arrays of its cleaned splat, for the live preview

    def save_state():
        json.dump(state, open(state_path, "w"), indent=1)

    for k, tile in enumerate(state["tiles"]):  # every tile's dataset and plan first: the total of steps
        if "schedule" not in tile:
            if tile.get("status") in (None, "pending", "training"):  # a trained tile's dataset stays as it was
                tile["points"] = write_tile(project, k, tile, down)[1]
            tile["schedule"] = splat_plan.suggest(splat_plan.survey(tile_dir(project, k)), **plan_args, tiles="off")
            tile["iterations"] = tile["schedule"]["iterations"]
        tile.setdefault("step", 0)
        if tile.get("status") in ("round", "done", "partial") and on_snapshot and \
                os.path.exists(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")):
            done_parts[k] = _band0(_read(os.path.join(tile_dir(project, k), "splat", "clean_enu.ply")))
    save_state()
    total_steps = sum(t["iterations"] for t in state["tiles"])
    longest = max(t["iterations"] for t in state["tiles"])
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
            v = v[_inside(xy, _cell(tile["cell"]))]  # its part from the last round is replaced by the training one
            path, ident = _write_live(project, [done_parts[j] for j in sorted(done_parts) if j != k] + [v], step)
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
        before = sum(t["step"] for j, t in enumerate(state["tiles"]) if j != k)

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
                                  lidar_depth=lidar_depth, schedule=tile["schedule"], **common)
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
        keep = clean(v, _cell(tile["cell"]), tile["margin"], box)
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
    while any(t.get("status") != "done" for t in state["tiles"]):
        state["round"] = rnd
        save_state()
        for k, tile in enumerate(state["tiles"]):
            if tile.get("status") != "done" and tile["step"] < min(tile["iterations"], rnd * round_steps or
                                                                   tile["iterations"]):
                train(k, tile, rnd)
        if rnd >= rounds:
            break
        total = merge(project, state)
        on_line(f"splat: round {rnd} / {rounds}: all {n} tiles at step {min(rnd * round_steps, longest):,}, merged into "
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

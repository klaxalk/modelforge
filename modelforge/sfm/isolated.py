"""Photos no other pass ties in: left out of the nerfstudio export (transforms.json, sparse_pc.ply).

A long flight sees the same ground on several passes. Where the photos of a pass match no other pass (kladno_dira:
a crossing at 24-42 m over ground the other passes saw from ~6 m, a 4-7x scale difference SIFT does not bridge),
nothing in the images places that pass against the rest: only its priors and the chain of its own photos do, and it
triangulates its own copy of the ground, metres beside the others' (kladno_dira 1586-1613 s: 4 m off its priors, its
camera path 3.5 % smaller than the GPS track). In a splat such a copy is a ghost surface, worse than missing views.

`find`: a photo is weakly tied when under `max_link_share` of its 3D points are also seen by photos more than
`min_gap_s` apart in time (another pass). Its own points are compared with the other passes' ground cell by cell
(`cell_m` cells; per cell and pass of `min_gap_s` the median height; reference = the median over the passes not
next to the photo's): it is dropped when more than `max_off` of its cells lie more than `max_dz` off. Per cell,
not per photo: the copies are often tilted or noisy rather than shifted as a whole (kladno_dira 1613 s: median
+0.16 m, but 63 % of its cells > 1 m off). Photos tied to other passes have a median of 7 % such cells (steep quarry
walls, machines); weakly tied ones 31 %. A photo over ground no other pass saw has no reference and stays, and so
does every photo of a flight without revisits.

Such a photo is only a candidate: cells also differ where the passes see different surfaces of the same place (the
canopy from above, the ground under it from a low pass; a hall's roof and, from a flight inside it, its floor). A
copy is off on open flat ground too. So a candidate is a copy when it has >= `MIN_FLAT` flat cells (the other passes'
ground level, within `FLAT_DZ`, over the 3 x 3 cells around) within `FLAT_MAX_DZ` of the others, and at least
`FLAT_OFF` of these lie more than `FLAT_MIN_DZ` off. Flat cells further off are another surface (the priors keep a
copy within metres: kladno_dira's were 0.9 m on flat ground; berlin's ~10 m were floors under roofs) and say nothing. Candidates
less than `STRETCH_S` apart (a pass) are decided together, by those with flat cells: a stretch without any stays.
berlin (2026-10-22; low passes under trees, a flight inside the halls): 1095 of 3142 photos dropped by the cells
alone, 77 now; renders of the splat trained without them lined up with the photos (the poses were right).
kladno_dira: 309 -> 192, all 48 of the known crossing still dropped. kladno_dira (global refine): 309 of 2883 photos dropped; tile 3's
2 m cells where a pass's ground is > 1 m off the others' 175 -> 52 (30 %: 243 photos, 102 cells).
"""
import collections

import numpy as np

MIN_GAP_S = 60.0
MAX_LINK_SHARE = 0.02
CELL_M = 1.0
MAX_DZ = 1.0
MAX_OFF = 0.2
MIN_CELLS = 10
MIN_FLAT = 5  # flat cells (within FLAT_MAX_DZ) a candidate needs for a verdict of its own
FLAT_DZ = 0.3  # m: the other passes' ground over the 3 x 3 cells around varies less -> a flat cell
FLAT_OFF = 0.5  # share of those flat cells more than FLAT_MIN_DZ off that makes a candidate a copy
FLAT_MIN_DZ = 0.5  # m
FLAT_MAX_DZ = 4.0  # m: a flat cell further off is another surface (a floor under a roof), not a copy
STRETCH_S = 5.0  # candidates closer in time are one stretch (one pass), decided together


def find(rec, times, min_gap_s=MIN_GAP_S, max_link_share=MAX_LINK_SHARE, cell_m=CELL_M, max_dz=MAX_DZ,
         max_off=MAX_OFF, min_cells=MIN_CELLS, progress=None):
    """times: image name -> project time [s] (None: unknown). Returns {"drop": [names], "isolated" (weakly tied
    photos), "candidates" (of them, with their cells off), "photos", "linking_points", "stretches": [(t0, t1, photos dropped, median share of cells off)]}.
    progress(text, k, n): the points read, then the photos checked."""
    ids = [i for i in rec.reg_image_ids() if times.get(rec.image(i).name) is not None]
    t_img = {i: float(times[rec.image(i).name]) for i in ids}
    out = {"drop": [], "isolated": 0, "candidates": 0, "photos": rec.num_reg_images(), "linking_points": 0,
           "stretches": []}
    if len(ids) < 2:
        return out
    pids, xyz, tmean, link = [], [], [], []
    n_pts = rec.num_points3D()
    for k, (pid, p) in enumerate(rec.points3D.items()):
        if progress and k % 200_000 == 0:
            progress("finding isolated photos: reading point", k, n_pts)
        ts = [t_img[e.image_id] for e in p.track.elements if e.image_id in t_img]
        if ts:
            pids.append(pid)
            xyz.append(p.xyz)
            tmean.append(sum(ts) / len(ts))
            link.append(max(ts) - min(ts) > min_gap_s)
    row = dict(zip(pids, range(len(pids))))
    xyz, link = np.asarray(xyz), np.asarray(link, bool)
    out["linking_points"] = int(link.sum())
    if not len(xyz):
        return out
    c = np.floor(xyz[:, :2] / cell_m).astype(np.int64)
    ny = int(np.ptp(c[:, 1])) + 3  # a free row and column around, so the 3 x 3 neighbours never wrap
    key = (c[:, 0] - c[:, 0].min() + 1) * ny + (c[:, 1] - c[:, 1].min() + 1)
    around = [dx * ny + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    pas = np.floor(np.asarray(tmean) / min_gap_s).astype(np.int64)
    order = np.lexsort((pas, key))
    k, q, z = key[order], pas[order], xyz[order, 2]
    cut = np.flatnonzero((np.diff(k) != 0) | (np.diff(q) != 0)) + 1
    ground = collections.defaultdict(dict)  # cell -> pass -> median height (>= 5 points)
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(k)]):
        if e - s >= 5:
            ground[k[s]][q[s]] = float(np.median(z[s:e]))
    cand, off_of, verdict = [], {}, {}  # verdict: True copy, False not, None no flat cells to tell
    for k, i in enumerate(ids):
        if progress and k % 25 == 0:
            progress("finding isolated photos: photo", k, len(ids))
        rows = np.array([row[p.point3D_id] for p in rec.image(i).get_observation_points2D()
                         if p.point3D_id in row], int)
        if not len(rows) or link[rows].mean() >= max_link_share:
            continue
        out["isolated"] += 1
        own = rows[~link[rows]]
        b = int(np.floor(t_img[i] / min_gap_s))
        ok = np.argsort(key[own], kind="stable")
        ck, zz = key[own][ok], xyz[own, 2][ok]
        cut = np.flatnonzero(np.diff(ck)) + 1
        refs = {}

        def ref(cell):  # the other passes' height there (not the photo's pass or the ones next to it)
            if cell not in refs:
                h = [h for p_, h in ground.get(cell, {}).items() if abs(p_ - b) > 1]
                refs[cell] = float(np.median(h)) if h else None
            return refs[cell]
        dz, flat = [], []
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(ck)]):
            r0 = ref(ck[s])
            if e - s >= 3 and r0 is not None:
                dz.append(np.median(zz[s:e]) - r0)
                nb = [h for h in (ref(ck[s] + d) for d in around) if h is not None]
                if len(nb) >= 6 and np.ptp(nb) < FLAT_DZ:
                    flat.append(dz[-1])
        if len(dz) >= min_cells:
            off = float(np.mean(np.abs(dz) > max_dz))
            if off > max_off:
                cand.append(i)
                off_of[i] = off
                flat = np.abs(flat)
                flat = flat[flat <= FLAT_MAX_DZ]
                verdict[i] = None if len(flat) < MIN_FLAT else bool(np.mean(flat > FLAT_MIN_DZ) >= FLAT_OFF)
    cand.sort(key=lambda i: t_img[i])
    passes = []
    for i in cand:  # candidates less than STRETCH_S apart: one pass, decided together by its flat-cell verdicts
        if passes and t_img[i] - t_img[passes[-1][-1]] < STRETCH_S:
            passes[-1].append(i)
        else:
            passes.append([i])
    drop = []
    for r in passes:
        v = [verdict[i] for i in r if verdict[i] is not None]
        if v and np.mean(v) >= 0.5:
            drop += r
    out["candidates"] = len(cand)
    out["drop"] = [rec.image(i).name for i in drop]
    runs = []
    for i in drop:  # stretches: consecutive dropped photos less than 5 s apart
        if runs and t_img[i] - t_img[runs[-1][-1]] < STRETCH_S:
            runs[-1].append(i)
        else:
            runs.append([i])
    out["stretches"] = [(round(t_img[r[0]], 1), round(t_img[r[-1]], 1), len(r),
                         round(float(np.median([off_of[i] for i in r])), 2)) for r in runs]
    return out


def remove(rec, names):
    """Delete the images' 3D points that no other photo sees (their copy of the ground); the images stay
    registered (export skips them by name). Returns the number of points deleted."""
    gone = {rec.find_image_with_name(n).image_id for n in names}
    dead = [pid for pid, p in rec.points3D.items() if all(e.image_id in gone for e in p.track.elements)]
    for pid in dead:
        rec.delete_point3D(pid)
    return len(dead)

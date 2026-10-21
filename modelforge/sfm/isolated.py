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
does every photo of a flight without revisits. kladno_dira (global refine): 309 of 2883 photos dropped; tile 3's
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


def find(rec, times, min_gap_s=MIN_GAP_S, max_link_share=MAX_LINK_SHARE, cell_m=CELL_M, max_dz=MAX_DZ,
         max_off=MAX_OFF, min_cells=MIN_CELLS):
    """times: image name -> project time [s] (None: unknown). Returns {"drop": [names], "isolated" (weakly tied
    photos), "photos", "linking_points", "stretches": [(t0, t1, photos dropped, median share of cells off)]}."""
    ids = [i for i in rec.reg_image_ids() if times.get(rec.image(i).name) is not None]
    t_img = {i: float(times[rec.image(i).name]) for i in ids}
    out = {"drop": [], "isolated": 0, "photos": rec.num_reg_images(), "linking_points": 0, "stretches": []}
    if len(ids) < 2:
        return out
    pids, xyz, tmean, link = [], [], [], []
    for pid, p in rec.points3D.items():
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
    key = (c[:, 0] - c[:, 0].min()) * (int(np.ptp(c[:, 1])) + 1) + (c[:, 1] - c[:, 1].min())
    pas = np.floor(np.asarray(tmean) / min_gap_s).astype(np.int64)
    order = np.lexsort((pas, key))
    k, q, z = key[order], pas[order], xyz[order, 2]
    cut = np.flatnonzero((np.diff(k) != 0) | (np.diff(q) != 0)) + 1
    ground = collections.defaultdict(dict)  # cell -> pass -> median height (>= 5 points)
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(k)]):
        if e - s >= 5:
            ground[k[s]][q[s]] = float(np.median(z[s:e]))
    drop, off_of = [], {}
    for i in ids:
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
        dz = []
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(ck)]):
            ref = [h for p_, h in ground.get(ck[s], {}).items() if abs(p_ - b) > 1]
            if e - s >= 3 and ref:
                dz.append(np.median(zz[s:e]) - np.median(ref))
        if len(dz) >= min_cells:
            off = float(np.mean(np.abs(dz) > max_dz))
            if off > max_off:
                drop.append(i)
                off_of[i] = off
    drop.sort(key=lambda i: t_img[i])
    out["drop"] = [rec.image(i).name for i in drop]
    runs = []
    for i in drop:  # stretches: consecutive dropped photos less than 5 s apart
        if runs and t_img[i] - t_img[runs[-1][-1]] < 5.0:
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

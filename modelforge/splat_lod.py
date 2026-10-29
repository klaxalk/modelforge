"""Level of detail for large splats (docs/todo_splat_lod.md): the GUI sends a fixed budget of Gaussians, fine near the
camera, coarse far away, at any splat size.

Built once per splat (stage `splat-lod`, `python -m modelforge.splat_lod PROJECT`):
- cells: splat/splat_enu.ply cut recursively at the median of the longer horizontal axis (the vertical too when it is
  the longest) until a cell holds <= CELL_MAX Gaussians;
- per cell levels 0 (all), 1, 2, ... of ~1/4 the Gaussians each: Gaussians are merged per voxel (cubic, its size
  searched so the count comes out right) by opacity x area weighted moment matching (hierarchical 3DGS, Kerbl et al.
  2024): mean, covariance = sum w (S_i + d_i d_i^T) / sum w, colour = weighted mean, opacity = covered area / the
  merged Gaussian's area (<= 1). Gaussians larger than the voxel stay as they are (they are coarse already). Band 0
  only: the GUI shows base colours.
- PROJECT/splat/lod/: index.json (cells: box, per level offset / count / voxel size; the splat's size + mtime) and
  per level L<k>_centers.npy (float32), L<k>_cov.npy (float16 upper triangle xx xy xz yy yz zz), L<k>_rgba.npy
  (uint8), the cells' Gaussians one after the other (memory-mapped by the GUI); L0_parent.npy = each level-0
  Gaussian's index in splat_enu.ply, L<k>_parent.npy (k >= 1, in level 0's order) = the level-k Gaussian (within its
  cell) it was merged into (the semantic search highlights a merged Gaussian by its best child).

`choose` picks the cut for a camera: every cell starts at its coarsest level; the cell whose voxel looks largest on
screen (outside the view x OUTSIDE_WEIGHT) is refined one level at a time while the budget holds and its voxel is
over PIXELS px.
"""
import argparse
import heapq
import json
import os
import shutil
import time

import numpy as np

OUT = os.path.join("splat", "lod")
SPLAT = os.path.join("splat", "splat_enu.ply")  # the project's splat; others (splat=) get a folder of their own
CELL_MAX = 100_000  # Gaussians per cell at level 0
RATIO = 4  # each level holds ~1/RATIO of the one below
MIN_LEVEL_GAUSSIANS = 1_000  # no coarser level once a cell is this small
MAX_LEVELS = 5
COV_SCALE = 1.0  # merged covariances x this (calibrated: build/lod/calibrate.py)
PIXELS = 2.0  # a level is fine enough once its voxel is under this many pixels (karlak: merged levels beat level 0 up to ~3 px)
OUTSIDE_WEIGHT = 0.25  # cells outside the view refine later
BUDGET = 2_000_000  # Gaussians in the browser (View -> LoD budget)
TRIU = np.array([0, 1, 2, 4, 5, 8])
VERSION = 1


def lod_dir(project, splat=None):
    """The LoD folder of the project's splat, or of another splat file in it (splat: a path in the project, e.g. the
    splat from before Difix: splat/splat_enu.pre_difix.ply -> splat/lod_pre_difix)."""
    if not splat or os.path.normpath(splat) == SPLAT:
        return os.path.join(project, OUT)
    stem = os.path.splitext(os.path.basename(splat))[0].replace("splat_enu.", "").replace(".", "_")
    return os.path.join(project, os.path.dirname(splat), "lod_" + stem)


def _splat_key(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime_ns)]


def index(project, splat=None):
    """The LoD index of the project's splat (or splat, a path in the project), None when missing or built for another
    splat."""
    folder = lod_dir(project, splat)
    splat = os.path.join(project, splat or SPLAT)
    try:
        d = json.load(open(os.path.join(folder, "index.json")))
    except (OSError, ValueError):
        return None
    if not os.path.exists(splat) or d.get("splat") != _splat_key(splat) or d.get("version") != VERSION:
        return None
    return d


def _cells(centers, cell_max=CELL_MAX):
    """Index arrays of the cells: recursive median cuts along the box's longest axis."""
    out, todo = [], [np.arange(len(centers))]
    while todo:
        idx = todo.pop()
        if len(idx) <= cell_max:
            out.append(idx)
            continue
        c = centers[idx]
        lo, hi = np.percentile(c, [1, 99], axis=0)
        ax = int(np.argmax(hi - lo))
        order = np.argsort(c[:, ax], kind="stable")
        half = len(idx) // 2
        todo += [idx[order[:half]], idx[order[half:]]]
    return out


def _voxel_keys(c, v):
    q = np.floor(c / v).astype(np.int64)
    q -= q.min(0)
    span = q.max(0) + 1
    return (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]


def _count(c, size, v):
    small = size <= v
    return int((~small).sum()) + (len(np.unique(_voxel_keys(c[small], v))) if small.any() else 0)


def merge(c, cov, rgb, op, v, cov_scale=COV_SCALE):
    """Gaussians of one cell merged per voxel of size v (those larger than v kept): (c, cov, rgb, op), parent (the
    index of each input Gaussian's output Gaussian)."""
    size = np.sqrt(np.maximum(cov[:, 0, 0] + cov[:, 1, 1] + cov[:, 2, 2], 0))
    small = size <= v
    big = ~small
    cs, covs, rgbs, ops = c[small].astype(np.float64), cov[small].astype(np.float64), rgb[small], op[small, 0]
    parent = np.empty(len(c), np.int32)
    parent[big] = np.arange(int(big.sum()))
    if not len(cs):
        return (c, cov, rgb, op), np.arange(len(c), dtype=np.int32)
    _, g = np.unique(_voxel_keys(cs, v), return_inverse=True)
    m = g.max() + 1
    ev = np.linalg.eigvalsh(covs)  # ascending
    area = np.sqrt(np.maximum(ev[:, 1] * ev[:, 2], 1e-30))  # sigma_1 sigma_2 of the two largest axes
    w = ops * area + 1e-30
    W = np.bincount(g, w, m)
    mu = np.stack([np.bincount(g, w * cs[:, k], m) for k in range(3)], -1) / W[:, None]
    d = cs - mu[g]
    S = covs + d[:, :, None] * d[:, None, :]
    cm = np.stack([np.bincount(g, w * S[:, i, j], m) for i in range(3) for j in range(3)], -1).reshape(m, 3, 3)
    cm = cm / W[:, None, None] * cov_scale
    col = np.stack([np.bincount(g, w * rgbs[:, k], m) for k in range(3)], -1) / W[:, None]
    evm = np.linalg.eigvalsh(cm)
    am = np.sqrt(np.maximum(evm[:, 1] * evm[:, 2], 1e-30))
    opm = np.clip(W / am, 0, 1)
    parent[small] = int(big.sum()) + g
    return (np.r_[c[big], mu.astype(np.float32)], np.r_[cov[big], cm.astype(np.float32)],
            np.r_[rgb[big], col.astype(np.float32)], np.r_[op[big, 0], opm.astype(np.float32)][:, None]), parent


def _voxel_for(c, size, target):
    """The voxel size that leaves ~target Gaussians (bisection in log space)."""
    lo, hi = max(np.median(size), 1e-4), max(np.ptp(c, 0).max(), 1e-3)
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        if _count(c, size, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def build(project, progress=None, token=None, cov_scale=COV_SCALE, cell_max=CELL_MAX, splat=None):
    """Build PROJECT/splat/lod/ for splat/splat_enu.ply (or lod_dir(project, splat) for splat); returns the index."""
    from modelforge.gui import data as gdata
    t0 = time.time()
    out = lod_dir(project, splat)
    splat = os.path.join(project, splat or SPLAT)
    key = _splat_key(splat)
    say = progress or (lambda s: None)
    say("level of detail: reading the splat")
    c, cov, rgb, op = gdata.load_splat(splat, None)
    n = len(c)
    cells = _cells(c, cell_max)
    tmp = out + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    levels = {}  # level -> lists of arrays
    meta = []
    for ci, idx in enumerate(cells):
        if token is not None:
            token.check()
        say(f"level of detail: cell {ci + 1} / {len(cells)}")
        cc, ccov, crgb, cop = c[idx], cov[idx], rgb[idx], op[idx]
        lo, hi = np.percentile(cc, [2, 98], axis=0)
        cell = {"box": [lo.tolist(), hi.tolist()], "levels": []}
        size = np.sqrt(np.maximum(ccov[:, 0, 0] + ccov[:, 1, 1] + ccov[:, 2, 2], 0))
        lvl, parent = (cc, ccov, crgb, cop), idx.astype(np.int32)  # level 0: the index in splat_enu.ply
        k, voxel = 0, 0.0
        while True:
            levels.setdefault(k, []).append(lvl + (parent,))
            cell["levels"].append({"n": len(lvl[0]), "voxel": float(voxel)})
            if len(lvl[0]) < MIN_LEVEL_GAUSSIANS * RATIO or k + 1 >= MAX_LEVELS:
                break
            target = max(len(cc) // RATIO ** (k + 1), MIN_LEVEL_GAUSSIANS)
            voxel = _voxel_for(cc, size, target)
            lvl, parent = merge(cc, ccov, crgb, cop, voxel, cov_scale)  # always from level 0: no error piles up
            k += 1
        meta.append(cell)
    for k, parts in levels.items():
        offs, poffs, o, po = [], [], 0, 0
        for p in parts:
            offs.append(o)
            poffs.append(po)
            o += len(p[0])
            po += len(p[4])
        np.save(os.path.join(tmp, f"L{k}_centers.npy"), np.concatenate([p[0] for p in parts]).astype(np.float32))
        np.save(os.path.join(tmp, f"L{k}_cov.npy"),
                np.concatenate([p[1].reshape(-1, 9)[:, TRIU] for p in parts]).astype(np.float16))
        rgba = np.concatenate([np.c_[p[2], p[3]] for p in parts])
        np.save(os.path.join(tmp, f"L{k}_rgba.npy"), np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8))
        np.save(os.path.join(tmp, f"L{k}_parent.npy"), np.concatenate([p[4] for p in parts]))
        ci = 0
        for cell in meta:
            if len(cell["levels"]) > k:
                cell["levels"][k]["offset"] = offs[ci]
                cell["levels"][k]["parent_offset"] = poffs[ci]
                ci += 1
    idx_out = {"version": VERSION, "splat": key, "gaussians": n, "cells": meta, "cov_scale": cov_scale,
               "levels": {str(k): sum(len(p[0]) for p in v) for k, v in levels.items()},
               "seconds": round(time.time() - t0, 1)}
    json.dump(idx_out, open(os.path.join(tmp, "index.json"), "w"))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return idx_out


def build_in(folder, splat=None):
    """build(folder), for modelforge.worker (a "project" argument would make it a job of the project)."""
    build(folder, splat=splat)


def load_cell(project, k, cell, splat=None):
    """viser arrays (centers, covariances, rgbs, opacities) of one cell at level k."""
    lv = cell["levels"][k]
    s = slice(lv["offset"], lv["offset"] + lv["n"])
    d = lod_dir(project, splat)
    c = np.load(os.path.join(d, f"L{k}_centers.npy"), mmap_mode="r")[s]
    t = np.asarray(np.load(os.path.join(d, f"L{k}_cov.npy"), mmap_mode="r")[s], dtype=np.float32)
    rgba = np.asarray(np.load(os.path.join(d, f"L{k}_rgba.npy"), mmap_mode="r")[s], dtype=np.float32) / 255
    cov = t[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    return np.ascontiguousarray(c), cov, np.ascontiguousarray(rgba[:, :3]), np.ascontiguousarray(rgba[:, 3:])


def cell_sources(project, k, cell, splat=None):
    """(original indices of the cell's level-0 Gaussians, their level-k Gaussian within the cell or None at k 0)."""
    d = lod_dir(project, splat)
    l0 = cell["levels"][0]
    s = slice(l0["offset"], l0["offset"] + l0["n"])
    src = np.asarray(np.load(os.path.join(d, "L0_parent.npy"), mmap_mode="r")[s])
    if k == 0:
        return src, None
    po = cell["levels"][k]["parent_offset"]
    return src, np.asarray(np.load(os.path.join(d, f"L{k}_parent.npy"), mmap_mode="r")[po:po + l0["n"]])


def _box_distance(p, box):
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0), p - hi)))


def choose(idx, position, forward=None, fov_y=1.0, height_px=1080, budget=BUDGET, pixels=PIXELS):
    """{cell: level} for a camera at position (world), looking along forward (unit, None: all around)."""
    f = height_px / 2 / np.tan(fov_y / 2)
    position = np.asarray(position, float)
    cells = idx["cells"]
    cut = {i: len(c["levels"]) - 1 for i, c in enumerate(cells)}
    total = sum(c["levels"][cut[i]]["n"] for i, c in enumerate(cells))
    half_diag = [0.5 * np.linalg.norm(np.subtract(c["box"][1], c["box"][0])) for c in cells]

    def weight(i):
        if forward is None:
            return 1.0
        mid = 0.5 * (np.asarray(cells[i]["box"][0]) + np.asarray(cells[i]["box"][1]))
        d = mid - position
        dist = np.linalg.norm(d)
        if dist <= half_diag[i]:
            return 1.0
        ang = np.arccos(np.clip(np.dot(d, forward) / dist, -1, 1)) - np.arcsin(min(1.0, half_diag[i] / dist))
        return 1.0 if ang <= fov_y * 0.5 * 1.8 else OUTSIDE_WEIGHT  # 1.8: the wider horizontal field of view

    w = [weight(i) for i in range(len(cells))]
    dist = [max(_box_distance(position, c["box"]), 1e-3) for c in cells]

    def err(i):
        k = cut[i]
        return cells[i]["levels"][k]["voxel"] * f / dist[i] * w[i] if k > 0 else 0.0

    heap = [(-err(i), i) for i in cut if cut[i] > 0]
    heapq.heapify(heap)
    while heap:
        e, i = heapq.heappop(heap)
        if -e <= pixels:
            break
        k = cut[i]
        extra = cells[i]["levels"][k - 1]["n"] - cells[i]["levels"][k]["n"]
        if total + extra > budget:
            continue  # this one does not fit; a smaller step elsewhere may
        cut[i], total = k - 1, total + extra
        if cut[i] > 0:
            heapq.heappush(heap, (-err(i), i))
    return cut


def main():
    ap = argparse.ArgumentParser(description="Build the level-of-detail hierarchy of PROJECT/splat/splat_enu.ply")
    ap.add_argument("project")
    ap.add_argument("--cov-scale", type=float, default=COV_SCALE)
    a = ap.parse_args()
    d = build(a.project, progress=print, cov_scale=a.cov_scale)
    print(json.dumps({k: d[k] for k in ("gaussians", "levels", "seconds")}), len(d["cells"]), "cells")


if __name__ == "__main__":
    main()

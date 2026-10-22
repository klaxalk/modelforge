"""Tile layout "covisibility" for splat_tiles: the photos are partitioned by what they see together, and each tile's
region on the ground is a label map that follows the partition, with seams moved onto flat open ground.

Why (the user asked, 2026-10-22): rectangular cells are cut by straight lines, which cannot bend around an orbited
object or a building in the way of every acceptable line; tiles should not cut through such features.

  graph      photos linked by the 3D points they share (the SfM tracks of sfm_mesh.model_dir), plus a weaker link
             (CELL_WEIGHT x the median) for photos whose points fall on the same 2 m ground cells (passes at very
             different heights see the same ground without matching: kladno_dira), plus a tiny link between
             neighbours in capture order (keeps the graph connected).
  split      recursive spectral bisection (normalized cut along the Fiedler vector of the normalized Laplacian),
             each cut within SLACK of the balanced position. Photos circling an object share its
             points, so the cut avoids them by construction.
  regions    on a grid (GRID_M or coarser for large scenes): every sparse point votes for the parts whose photos see
             it, weighted 1 / distance^2 (the ground each part flew over, not its far oblique views); majority
             filter, islands removed, holes filled from the nearest label. Then the seams are redrawn within BAND_M:
             a geodesic watershed from the regions' interiors, where a step through a feature (height above the
             local ground, FEATURE_M = full) is cheap: an object goes wholly to one side, fronts meet on flat ground.
  Regions    lookup by position (beyond the grid: the edge cell's label, so outer regions extend to infinity),
             padded "inside", distance to the nearest seam (the clean's reach), outlines for the View tab.

On berlin (1734 photos, 4 tiles) against the rectangles: points > 3 m above ground within 3 m / 10 m of a seam
1.6 % / 6.3 % (2.4 % / 9.5 %), photo links cut 10 % (19 %), photo slots 2218 (2695), seams 2.2 km (1.7 km); the
orbits stay in one tile (the rectangles cut one). Prototype and sweeps: build/covis_tiles.py.
"""

import heapq
import math
import os

import numpy as np

GRID_M = 1.0  # label-map cell (m); coarser for large scenes (MAX_CELLS)
MAX_CELLS = 4_000_000
CELL_WEIGHT = 0.3  # ground-cell links relative to shared points (median to median)
BAND_M = 25.0  # the watershed may move a seam this far
FEATURE_M = 5.0  # height above the local ground that counts as a full feature
CONDUCT = 0.98  # a step through a full feature costs 1 - CONDUCT of a step over flat ground
SLACK = 0.25  # each split within this share of the smaller side off balance (0.5: berlin's tiles 125-809 photos;
#               0.25: 305-661, and fewer tall points near seams)
MIN_ISLAND_M2 = 400.0  # smaller pieces of a region (cut off from its main part) join their surroundings


def _cv2():
    import cv2
    return cv2


def load_tracks(project, names):
    """The SfM model's observations for the photos `names` (transforms.json order): (obs photo index, obs point
    index, points xyz, camera centres (n, 3)). Photos the model lacks have no observations."""
    from modelforge import sfm_mesh
    from modelforge.gui import data as gdata
    model = sfm_mesh.model_dir(project)
    path = os.path.join(model, "images.bin")
    imgs = gdata.read_images_bin(path)
    index = {n: i for i, n in enumerate(names)}
    obs_img, obs_pid, every = [], [], []
    centers = np.full((len(names), 3), np.nan)
    for name, _, R_cw, t_cw, off, n in imgs:
        kp = gdata.read_keypoints(path, off, n)
        pid = kp["pid"][kp["pid"] >= 0]
        every.append(pid)
        i = index.get(name)
        if i is None:
            continue
        centers[i] = -R_cw.T @ t_cw
        obs_img.append(np.full(len(pid), i, np.int32))
        obs_pid.append(pid)
    ids, tl = np.unique(np.concatenate(every), return_counts=True)  # every record of points3D.bin, for its offsets
    xyz, _ = gdata.read_points3d_bin(os.path.join(model, "points3D.bin"), ids, tl)
    obs_img, obs_pid = np.concatenate(obs_img), np.concatenate(obs_pid)
    obs_pt = np.searchsorted(ids, obs_pid)
    return obs_img, obs_pt.astype(np.int64), xyz, centers


def pair_counts(n, img, key):
    """(n, n) float32: per photo pair the number of keys (points, ground cells) both have."""
    o = np.lexsort((img, key))
    k, im = key[o], img[o]
    keep = np.r_[True, (k[1:] != k[:-1]) | (im[1:] != im[:-1])]
    k, im = k[keep], im[keep]
    start = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    length = np.diff(np.r_[start, len(k)])
    W = np.zeros((n, n), np.float32)
    for L in np.unique(length):
        if L < 2:
            continue
        block = im[start[length == L][:, None] + np.arange(L)]
        np.add.at(W, (np.repeat(block, L, axis=1).ravel(), np.tile(block, (1, L)).ravel()), 1.0)
    np.fill_diagonal(W, 0)
    return W


def affinity(n, obs_img, obs_pt, xyz, order):
    """The photo graph: shared points + CELL_WEIGHT x shared 2 m ground cells + a tiny link along `order`."""
    Wp = pair_counts(n, obs_img, obs_pt)
    cell = np.floor(xyz[:, :2] / 2.0).astype(np.int64)
    cell -= cell.min(0)
    ckey = cell[:, 0] * (int(cell[:, 1].max()) + 1) + cell[:, 1]
    Wc = pair_counts(n, obs_img, ckey[obs_pt])
    mp = float(np.median(Wp[Wp > 0])) if (Wp > 0).any() else 1.0
    mc = float(np.median(Wc[Wc > 0])) if (Wc > 0).any() else 1.0
    A = Wp + np.float32(CELL_WEIGHT * mp / mc) * Wc
    a, b = order[:-1], order[1:]
    A[a, b] += np.float32(1e-3 * mp)
    A[b, a] += np.float32(1e-3 * mp)
    return A


def _bisect(A, ids, frac, slack):
    """ids split ~frac : 1 - frac at the lowest normalized cut along the Fiedler vector, within slack."""
    S = A[np.ix_(ids, ids)].astype(np.float64)
    d = S.sum(1) + 1e-9
    Dm = 1 / np.sqrt(d)
    _, v = np.linalg.eigh(np.eye(len(ids)) - Dm[:, None] * S * Dm[None, :])
    order = np.argsort(v[:, 1] * Dm)
    n = len(ids)
    s = slack * min(frac, 1 - frac) * n
    k = np.arange(max(1, int(math.ceil(frac * n - s))), min(n - 1, int(frac * n + s)) + 1)
    So = S[np.ix_(order, order)]
    vol = np.cumsum(d[order])
    cut = vol[:-1] - 2 * np.cumsum(np.tril(So).sum(1))[:-1]
    ncut = cut / vol[:-1] + cut / (vol[-1] - vol[:-1])
    best = k[np.argmin(ncut[k - 1])]
    return ids[order[:best]], ids[order[best:]]


def split_photos(A, parts, slack):
    """Part index per photo: recursive bisection into `parts` (k -> floor(k/2) + ceil(k/2))."""
    lab = np.zeros(len(A), int)
    out = []

    def rec(ids, k):
        if k <= 1 or len(ids) < 2:
            out.append(ids)
            return
        a, b = _bisect(A, ids, (k // 2) / k, slack)
        rec(a, k // 2)
        rec(b, k - k // 2)
    rec(np.arange(len(A)), parts)
    for t, ids in enumerate(out):
        lab[ids] = t
    return lab, len(out)


class Regions:
    """A label map on the ground (ENU x, y): labels[row = y, col = x] = tile, cell (row, col) covering
    lo + (col, row) * grid ... + grid. Positions beyond the grid take the nearest edge cell's label."""

    def __init__(self, lo, grid, labels):
        self.lo, self.grid, self.labels = np.asarray(lo, float), float(grid), np.asarray(labels, np.int32)
        self._in, self._out = {}, {}

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["lo"], float(d["grid"]), d["labels"])

    def save(self, path):
        np.savez_compressed(path, lo=self.lo, grid=np.float64(self.grid), labels=self.labels)

    def _rc(self, xy):
        h, w = self.labels.shape
        c = np.clip(np.floor((np.asarray(xy, float)[:, 0] - self.lo[0]) / self.grid).astype(int), 0, w - 1)
        r = np.clip(np.floor((np.asarray(xy, float)[:, 1] - self.lo[1]) / self.grid).astype(int), 0, h - 1)
        return r, c

    def label(self, xy):
        r, c = self._rc(xy)
        return self.labels[r, c]

    def seam_distance(self, k):
        """(H, W) distance (m) from each cell of region k to the nearest cell of another region (inf if none)."""
        if k not in self._in:
            cv2 = _cv2()
            m = (self.labels == k).astype(np.uint8)
            self._in[k] = (cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64) * self.grid
                           if (m == 0).any() else np.full(m.shape, np.inf))
        return self._in[k]

    def distance_to(self, k):
        """(H, W) distance (m) from each cell to region k (0 inside)."""
        if k not in self._out:
            cv2 = _cv2()
            m = (self.labels != k).astype(np.uint8)
            self._out[k] = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float64) * self.grid
        return self._out[k]

    def inside(self, xy, k, pad=0.0):
        r, c = self._rc(xy)
        if pad <= 0:
            return self.labels[r, c] == k
        return self.distance_to(k)[r, c] <= pad + 1e-9

    def to_seam(self, xy, k):
        """Per position the distance to the nearest seam of region k (positions inside it)."""
        r, c = self._rc(xy)
        return self.seam_distance(k)[r, c]

    def bbox(self, k):
        rows, cols = np.nonzero(self.labels == k)
        if not len(rows):
            return [0.0, 0.0, 0.0, 0.0]
        h, w = self.labels.shape
        x0, x1 = self.lo[0] + cols.min() * self.grid, self.lo[0] + (cols.max() + 1) * self.grid
        y0, y1 = self.lo[1] + rows.min() * self.grid, self.lo[1] + (rows.max() + 1) * self.grid
        # an edge of the grid is open (the region extends to infinity there)
        return [-np.inf if cols.min() == 0 else x0, -np.inf if rows.min() == 0 else y0,
                np.inf if cols.max() == w - 1 else x1, np.inf if rows.max() == h - 1 else y1]

    def area(self, k, lo, hi):
        """m^2 of region k inside the box lo..hi (its outer regions are open: only the box counts)."""
        h, w = self.labels.shape
        x = self.lo[0] + (np.arange(w) + 0.5) * self.grid
        y = self.lo[1] + (np.arange(h) + 0.5) * self.grid
        m = (self.labels == k) & ((y >= lo[1]) & (y <= hi[1]))[:, None] & ((x >= lo[0]) & (x <= hi[0]))[None, :]
        return float(m.sum()) * self.grid ** 2

    def outlines(self, k, simplify_m=None):
        """Region k's boundary as closed polylines [(n, 2) ENU], simplified to about a cell."""
        cv2 = _cv2()
        m = np.pad((self.labels == k).astype(np.uint8), 1)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cs:
            c = cv2.approxPolyDP(c, (simplify_m or self.grid) / self.grid, True)[:, 0, :].astype(float) - 1
            if len(c) >= 3:
                out.append(self.lo + (c + 0.5) * self.grid)
        return out


def _label_map(xy, votes, grid, lo, shape, k, min_island):
    cv2 = _cv2()
    r = np.clip(((xy[:, 1] - lo[1]) / grid).astype(int), 0, shape[0] - 1)
    c = np.clip(((xy[:, 0] - lo[0]) / grid).astype(int), 0, shape[1] - 1)
    acc = np.zeros((k,) + shape, np.float32)
    for t in range(k):
        np.add.at(acc[t], (r, c), votes[:, t])
    lab = np.where(acc.sum(0) > 0, acc.argmax(0), -1).astype(np.int32)
    for _ in range(2):  # majority filter, islands, fill from the nearest labelled cell
        sm = np.stack([cv2.blur((lab == t).astype(np.float32), (5, 5)) for t in range(k)])
        lab = np.where(sm.sum(0) > 0, sm.argmax(0), -1).astype(np.int32)
        for t in range(k):
            n, cc, stats, _ = cv2.connectedComponentsWithStats((lab == t).astype(np.uint8), connectivity=4)
            if n > 2:
                big = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                small = [j for j in range(1, n) if j != big and stats[j, cv2.CC_STAT_AREA] < min_island]
                lab[np.isin(cc, small)] = -1
        empty = (lab < 0).astype(np.uint8)
        if not empty.any():
            break
        if empty.all():
            return np.zeros(shape, np.int32)
        _, near = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        src = np.zeros(int(near.max()) + 1, np.int32)
        ys, xs = np.nonzero(empty == 0)
        src[near[ys, xs]] = lab[ys, xs]
        lab = src[near]
    return lab


def _watershed(seeds, feature, conduct):
    """Multi-source Dijkstra (8 neighbours) from the seed cells; a step costs 1 - conduct x feature."""
    h, w = seeds.shape
    cost = (1.0 - conduct * np.clip(feature, 0, 1)).astype(np.float64)
    dist = np.full((h, w), np.inf)
    lab = seeds.astype(np.int32).copy()
    ys, xs = np.nonzero(seeds >= 0)
    dist[ys, xs] = 0.0
    heap = [(0.0, int(y), int(x)) for y, x in zip(ys, xs)]
    heapq.heapify(heap)
    steps = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142))
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        t, cy = lab[y, x], cost[y, x]
        for dy, dx, L in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                nd = d + L * 0.5 * (cy + cost[yy, xx])
                if nd < dist[yy, xx]:
                    dist[yy, xx], lab[yy, xx] = nd, t
                    heapq.heappush(heap, (nd, yy, xx))
    return lab


def regions_from_parts(xyz, obs_img, obs_pt, centers, part, k, grid=None):
    """The label map of the photo parts (see the module doc). Returns Regions."""
    cv2 = _cv2()
    ok = np.isfinite(xyz).all(1)
    lo = np.percentile(xyz[ok, :2], 0.5, axis=0) - 5
    hi = np.percentile(xyz[ok, :2], 99.5, axis=0) + 5
    grid = grid or max(GRID_M, math.sqrt(float(np.prod(hi - lo)) / MAX_CELLS))
    shape = tuple(int(v) for v in np.ceil((hi - lo) / grid).astype(int)[::-1])
    inb = ok & np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), axis=1)
    d = np.linalg.norm(xyz[obs_pt] - centers[obs_img], axis=1)
    votes = np.zeros((len(xyz), k), np.float32)
    np.add.at(votes, (obs_pt, part[obs_img]), 1.0 / np.maximum(d, 1.0) ** 2)
    votes /= np.maximum(votes.sum(1, keepdims=True), 1e-12)
    init = _label_map(xyz[inb, :2], votes[inb], grid, lo, shape, k, int(MIN_ISLAND_M2 / grid ** 2))
    # height above the local ground (5th percentile per 5 m cell) as the feature map
    r = np.clip(((xyz[inb, 1] - lo[1]) / grid).astype(int), 0, shape[0] - 1)
    c = np.clip(((xyz[inb, 0] - lo[0]) / grid).astype(int), 0, shape[1] - 1)
    g = max(1, int(round(5.0 / grid)))
    gk = (r // g) * (shape[1] // g + 1) + c // g
    o = np.lexsort((xyz[inb, 2], gk))
    first = np.r_[0, np.flatnonzero(np.diff(gk[o])) + 1]
    cnt = np.diff(np.r_[first, len(o)])
    ground = np.zeros(int(gk.max()) + 1)
    ground[gk[o[first]]] = xyz[inb, 2][o[first + (cnt * 0.05).astype(int)]]
    height = np.zeros(shape, np.float32)
    np.maximum.at(height, (r, c), np.clip(xyz[inb, 2] - ground[gk], 0, 50).astype(np.float32))
    feature = np.clip(cv2.dilate(height, np.ones((3, 3), np.uint8)) / FEATURE_M, 0, 1)
    seam = np.full(shape, np.inf)
    for t in range(k):
        m = (init == t).astype(np.uint8)
        if (m == 0).any():
            seam = np.where(m > 0, cv2.distanceTransform(m, cv2.DIST_L2, 5) * grid, seam)
    labels = _watershed(np.where(seam >= BAND_M, init, -1), feature, CONDUCT) if k > 1 else init
    return Regions(lo, grid, labels)


def partition(project, names, centers, parts, slack=SLACK):
    """Part per photo (transforms.json order `names`, `centers` their camera centres) and the Regions."""
    obs_img, obs_pt, xyz, cc = load_tracks(project, names)
    centers = np.where(np.isfinite(cc), cc, centers)
    order = np.argsort(np.array(names, dtype=object))  # file names sort by video, then frame: capture order
    A = affinity(len(names), obs_img, obs_pt, xyz, order)
    part, k = split_photos(A, parts, slack)
    return part, k, regions_from_parts(xyz, obs_img, obs_pt, centers, part, k)

"""Image pair selection: from pose priors (time neighbours + overlapping views), or without priors
(sequential(): time neighbours, quadratic jumps, exhaustive for small projects).

View overlap is approximated with look-at points: the optical axis intersected
with the ground plane at takeoff height (z = -rel_alt of the ENU origin), capped
at max_depth. Two views overlap when their look-at points are close relative to
the viewing distance and their axes point in similar directions; this also pairs
revisits of the same spot from different passes, and overlaps between videos.
"""

import os

import numpy as np


class Terrain:
    """Surface height over x / y from 3D points (e.g. an earlier sparse model): the median z per grid cell, cells
    without points filled from the nearest one. For look-at points over uneven ground (a pit, a slope, piles),
    where the flat plane at takeoff height puts oblique views' look-ats far off."""

    def __init__(self, xyz, cell=None, max_cells=400):
        import cv2
        xyz = np.asarray(xyz, float)
        lo, hi = np.percentile(xyz[:, :2], 1, axis=0), np.percentile(xyz[:, :2], 99, axis=0)
        self.cell = float(cell or max(float((hi - lo).max()) / max_cells, 0.5))
        self.lo = lo - self.cell
        self.shape = tuple((np.ceil((hi + self.cell - self.lo) / self.cell)).astype(int) + 1)
        ij = self._index(xyz[:, :2])
        keep = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        lin = ij[keep, 0] * self.shape[1] + ij[keep, 1]
        z = xyz[keep, 2]
        order = np.lexsort((z, lin))
        lin, z = lin[order], z[order]
        cells, start, count = np.unique(lin, return_index=True, return_counts=True)
        grid = np.full(self.shape[0] * self.shape[1], np.nan)
        grid[cells] = z[start + count // 2]  # median per cell
        empty = np.flatnonzero(np.isnan(grid))
        if len(empty) and len(cells):
            full = np.stack(np.unravel_index(cells, self.shape), 1).astype(np.float32)
            index = cv2.flann_Index(full, dict(algorithm=1, trees=4))
            nn, _ = index.knnSearch(np.stack(np.unravel_index(empty, self.shape), 1).astype(np.float32), 1,
                                    params=dict(checks=32))
            grid[empty] = grid[cells[nn[:, 0]]]
        self.grid = grid.reshape(self.shape)

    def _index(self, xy):
        return np.floor((np.asarray(xy, float) - self.lo) / self.cell).astype(int)

    def z(self, xy):
        """Surface height at x / y (outside the grid: its nearest edge cell)."""
        ij = self._index(xy)
        return self.grid[np.clip(ij[..., 0], 0, self.shape[0] - 1), np.clip(ij[..., 1], 0, self.shape[1] - 1)]


def look_at_points(priors, ground_z, max_depth=150.0, terrain=None):
    """Each camera's look-at point: its optical axis meets the ground (the plane z = ground_z, or the terrain:
    the first point along the axis at or below the surface), capped at max_depth."""
    names = sorted(priors, key=lambda n: priors[n].order)
    C = np.array([priors[n].center for n in names])
    A = np.array([priors[n].optical_axis for n in names])
    depth = np.full(len(names), max_depth)
    if terrain is None:
        down = A[:, 2] < -1e-3
        depth[down] = np.minimum((ground_z - C[down, 2]) / A[down, 2], max_depth)
    else:
        steps = np.arange(1.0, max_depth + 1e-9, max(terrain.cell / 2, 0.25))
        for a in range(0, len(names), 500):
            ray = C[a:a + 500, None, :] + steps[None, :, None] * A[a:a + 500, None, :]  # (n, K, 3)
            below = ray[..., 2] <= terrain.z(ray[..., :2])
            hit = below.any(1)
            depth[a:a + 500][hit] = steps[below[hit].argmax(1)]
    depth = np.clip(depth, 1.0, max_depth)
    return names, C + depth[:, None] * A, A, depth


def select(priors, ground_z, seq_neighbors=5, spatial_neighbors=15, max_axis_angle_deg=60.0,
           overlap_factor=0.5, terrain=None):
    names, P, A, depth = look_at_points(priors, ground_z, terrain=terrain)
    n = len(names)
    src = [priors[x].source for x in names]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + seq_neighbors)):
            if src[j] == src[i]:  # time neighbours only within one video piece
                pairs.add((i, j))
    cos_max = np.cos(np.radians(max_axis_angle_deg))
    for i in range(n):
        d = np.linalg.norm(P - P[i], axis=1)
        ok = (A @ A[i] > cos_max) & (d < overlap_factor * (depth + depth[i]))
        ok[i] = False
        cand = np.flatnonzero(ok)
        for j in cand[np.argsort(d[cand])][:spatial_neighbors]:
            pairs.add((min(i, j), max(i, j)))
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def sequential(sequences, overlap=10, quadratic=True, exhaustive_below=150, cross_step=5):
    """Pairs without pose priors (videos without metadata). sequences: image names per source, in time
    order. Small projects match exhaustively; otherwise time neighbours (+-overlap), quadratic jumps
    (16, 32, 64, ... frames apart: revisits and loops) within each source, and a coarse grid (every
    cross_step-th image) between sources so separate videos can join."""
    names = [n for seq in sequences for n in seq]
    if len(names) <= exhaustive_below:
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    out = set()
    for seq in sequences:
        n = len(seq)
        for i in range(n):
            for k in range(1, overlap + 1):
                if i + k < n:
                    out.add((seq[i], seq[i + k]))
            j = 1
            while quadratic and (2 ** j) < n:
                if 2 ** j > overlap and i + 2 ** j < n:
                    out.add((seq[i], seq[i + 2 ** j]))
                j += 1
    for si, a_seq in enumerate(sequences):
        for b_seq in sequences[si + 1:]:
            out.update((a, b) for a in a_seq[::cross_step] for b in b_seq[::cross_step])
    return sorted(out)


def write(project, pairs):
    path = os.path.join(project, "pairs.txt")
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
    return path

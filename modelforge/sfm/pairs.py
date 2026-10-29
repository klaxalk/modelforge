"""Image pair selection: from pose priors (time neighbours + overlapping views), or without priors
(sequential(): time neighbours, quadratic jumps, exhaustive for small projects).

View overlap is approximated with look-at points: the optical axis intersected
with the ground plane at takeoff height (z = -rel_alt of the ENU origin), capped
at max_depth. Two views overlap when their look-at points are close relative to
the viewing distance and their axes point in similar directions; this also pairs
revisits of the same spot from different passes, and overlaps between videos.

Several flights (2026-10-25): each flight's look-ats use its own takeoff height as the ground (ground_z per
source: the reference's + that flight's altitude offset), and views of different flights are compared by their
horizontal distance. A wrong altitude offset (karlak3: abs_alt 240 m low) then still pairs the flights, and the
refine measures the true offset from those matches (robust.update_source_offsets); before, the flight's look-ats
landed far from the others' and no pair between the flights was matched.

Covisible pairs between passes (covisible(); 2026-10-28): on berlin_all only 6 % of the photo pairs more than 60 s
apart that see the same surface (by the refined model) were in pairs.txt, while 63 % of those tried were linked. On a
re-run, the earlier model's covisibility adds up to COVIS_PER_PHOTO such pairs per photo.
"""

import os

import numpy as np

COVIS_PER_PHOTO = 10
COVIS_GAP_S = 60.0
COVIS_VOXEL_M = 2.0
COVIS_MIN_OVERLAP = 0.3
COVIS_POINTS = 3000  # observed points sampled per photo


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
    the first point along the axis at or below the surface), capped at max_depth. ground_z: one height, or
    {source: height} (each flight's own takeoff height)."""
    names = sorted(priors, key=lambda n: priors[n].order)
    C = np.array([priors[n].center for n in names])
    A = np.array([priors[n].optical_axis for n in names])
    depth = np.full(len(names), max_depth)
    if terrain is None:
        g = np.array([ground_z.get(priors[n].source, ground_z.get(0, 0.0)) for n in names], float) \
            if isinstance(ground_z, dict) else np.full(len(names), float(ground_z))
        down = A[:, 2] < -1e-3
        depth[down] = np.minimum((g[down] - C[down, 2]) / A[down, 2], max_depth)
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
    src_arr = np.array(src)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + seq_neighbors)):
            if src[j] == src[i]:  # time neighbours only within one video piece
                pairs.add((i, j))
    cos_max = np.cos(np.radians(max_axis_angle_deg))
    for i in range(n):
        diff = P - P[i]
        # another flight: horizontally (its altitude offset may be wrong; the refine measures it from the matches)
        d = np.where(src_arr != src[i], np.linalg.norm(diff[:, :2], axis=1), np.linalg.norm(diff, axis=1))
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


def near_pairs(pts, r):
    """Index pairs (i < j) of points closer than r (a grid of cell r, neighbouring cells compared)."""
    cell = np.floor(pts / r).astype(np.int64)
    buckets = {}
    for i, c in enumerate(map(tuple, cell)):
        buckets.setdefault(c, []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    out = []
    for k, members in buckets.items():
        a = np.array(members)
        for d in offs:
            other = buckets.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))
            if other is None:
                continue
            b = np.array(other)
            ia, ib = np.nonzero(np.linalg.norm(pts[a][:, None] - pts[b][None], axis=2) < r)
            ia, ib = a[ia], b[ib]
            keep = ia < ib
            out.append(np.stack([ia[keep], ib[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def covisible(project, priors, per_photo=COVIS_PER_PHOTO, min_gap_s=COVIS_GAP_S, voxel=COVIS_VOXEL_M,
              min_overlap=COVIS_MIN_OVERLAP, model="refined", seed=0):
    """Photo pairs more than min_gap_s apart (or of two sources) that see the same surface in the project's earlier
    model (sparse/<model>): each photo's observed 3D points binned into `voxel` m voxels, overlap = shared voxels /
    the smaller photo's (separate copies of a surface, passes the refine left apart, still share voxels). Up to
    per_photo per photo, best overlap first, only photos of `priors` (the current images). Returns
    [(a, b, overlap)], [] without a model."""
    from modelforge.gui.data import read_images_bin, read_keypoints, read_points3d_bin
    d = os.path.join(project, "sparse", model)
    if per_photo <= 0 or not os.path.exists(os.path.join(d, "images.bin")):
        return []
    rng = np.random.default_rng(seed)
    imgs, pids, allp = [], [], []
    for im in read_images_bin(os.path.join(d, "images.bin")):
        p = read_keypoints(os.path.join(d, "images.bin"), im[4], im[5])["pid"]
        p = p[p >= 0].astype(np.int64)
        allp.append(p)  # read_points3d_bin wants every point of the file, with its track length
        if im[0] in priors:
            imgs.append(im)
            pids.append(rng.choice(p, COVIS_POINTS, replace=False) if len(p) > COVIS_POINTS else p)
    if len(imgs) < 2:
        return []
    ids, counts = np.unique(np.concatenate(allp), return_counts=True)
    del allp
    xyz, _ = read_points3d_bin(os.path.join(d, "points3D.bin"), ids, counts)
    names = [im[0] for im in imgs]
    C = np.array([-im[2].T @ im[3] for im in imgs])
    look, depth, vox = np.zeros((len(imgs), 3)), np.full(len(imgs), np.nan), []
    mult = np.array([1, 1 << 21, 1 << 42], np.int64)
    for k, p in enumerate(pids):
        X = xyz[np.searchsorted(ids, p)]
        if len(X):
            look[k], depth[k] = np.median(X, axis=0), np.median(np.linalg.norm(X - C[k], axis=1))
        else:
            look[k] = C[k]
        vox.append(np.unique(np.floor(X / voxel).astype(np.int64) @ mult))
    if not np.isfinite(depth).any():
        return []
    cand = near_pairs(look, float(np.nanpercentile(depth, 75)))  # footprints of about that size overlap
    t = np.array([priors[n].time for n in names], float)
    s = np.array([priors[n].source for n in names])
    far = (s[cand[:, 0]] != s[cand[:, 1]]) | (np.abs(t[cand[:, 0]] - t[cand[:, 1]]) > min_gap_s)
    cand = cand[far]
    scored = []
    for a, b in cand:
        o = np.intersect1d(vox[a], vox[b], assume_unique=True).size / max(1, min(vox[a].size, vox[b].size))
        if o >= min_overlap:
            scored.append((o, a, b))
    scored.sort(key=lambda r: -r[0])
    used, out = np.zeros(len(imgs), int), []
    for o, a, b in scored:
        if used[a] < per_photo or used[b] < per_photo:  # a pair counts for both photos' budgets
            out.append((names[a], names[b], round(float(o), 3)))
            used[a] += 1
            used[b] += 1
    return out

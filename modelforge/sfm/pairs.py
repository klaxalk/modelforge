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


def look_at_points(priors, ground_z, max_depth=150.0):
    names = sorted(priors, key=lambda n: priors[n].order)
    C = np.array([priors[n].center for n in names])
    A = np.array([priors[n].optical_axis for n in names])
    depth = np.full(len(names), max_depth)
    down = A[:, 2] < -1e-3
    depth[down] = np.minimum((ground_z - C[down, 2]) / A[down, 2], max_depth)
    depth = np.clip(depth, 1.0, max_depth)
    return names, C + depth[:, None] * A, A, depth


def select(priors, ground_z, seq_neighbors=5, spatial_neighbors=15, max_axis_angle_deg=60.0,
           overlap_factor=0.5):
    names, P, A, depth = look_at_points(priors, ground_z)
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

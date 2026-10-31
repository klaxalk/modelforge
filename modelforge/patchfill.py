"""PatchMatch fill of a hole in a top-down cell grid: content-aware fill for the ground of a splat (modelforge/replace.py;
the user asked, 2026-10-31: "the equivalent of Photoshop's content aware fill").

Every cell has a descriptor (colour, semantic feature, density, roughness; each normalized). For every hole cell an
offset to a source cell is searched whose PATCH x PATCH neighbourhood matches the hole cell's neighbourhood best: known
cells compare as they are, hole cells by their current fill. PatchMatch (Barnes 2009): initialized from the border
inwards (onion peel, each cell taking a random good offset of its filled neighbours or a random source), then ITERS
rounds of propagation (the offsets of the left / upper neighbours, then right / lower) and random search around the
current offset, each followed by updating the hole cells' descriptors from their sources. Distance does not count
(the user, 2026-10-31: what matters is similarity in colour, texture and semantics; a penalty per metre was used
before)."""
import cv2
import numpy as np

PATCH, ITERS, RANDOM = 5, 6, 8


def descriptors(R):
    """(H, W, D) normalized descriptor per cell, valid source mask (H, W)."""
    H, W = R["H"], R["W"]
    have = R["n"] > 0
    parts = [R["col"] * 1.0]
    if R.get("feat") is not None:
        f = R["feat"][have]
        mu = f.mean(0)
        U = np.linalg.svd(f[:: max(1, len(f) // 20000)] - mu, full_matrices=False)[2][:8]
        sem = np.zeros((H, W, 8), np.float32)
        sem[have] = (f - mu) @ U.T
        parts.append(sem * 0.5)
    parts.append(np.log1p(R["n"])[..., None] * 0.5)
    if R.get("rough") is not None:
        parts.append(R["rough"][..., None] * 1.0)
    D = np.concatenate(parts, -1).astype(np.float32)
    sd = D[have].std(0) + 1e-6
    D = (D - D[have].mean(0)) / sd
    D[~have] = 0
    return D


def fill(R, hole, avoid, rng=None):
    """Offsets (dy, dx) in cells for every hole cell (int array (H, W, 2); zeros outside), and the final cost per hole
    cell. hole, avoid: (H, W) bool. Sources: cells with ground, nothing standing on them, outside the hole and avoid,
    whose whole patch lies in the grid."""
    rng = rng or np.random.default_rng(0)
    H, W = R["H"], R["W"]
    D = descriptors(R)
    r = PATCH // 2
    src_ok = (R["n"] > 0) & ~R["obj"] & ~hole & ~avoid
    src_ok[:r], src_ok[-r:], src_ok[:, :r], src_ok[:, -r:] = False, False, False, False
    known = (R["n"] > 0) & ~hole
    sy, sx = np.nonzero(src_ok)
    cur = D.copy()  # the hole's descriptors as filled so far
    filled = known.copy()
    off = np.zeros((H, W, 2), int)
    cost = np.full((H, W), np.inf)
    win = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    def patch_cost(y, x, oy, ox):
        sy_, sx_ = y + oy, x + ox
        if not (r <= sy_ < H - r and r <= sx_ < W - r) or not src_ok[sy_, sx_]:
            return np.inf
        tot, n = 0.0, 0
        for dy, dx in win:
            ty, tx = y + dy, x + dx
            if 0 <= ty < H and 0 <= tx < W and filled[ty, tx]:
                d = cur[ty, tx] - D[sy_ + dy, sx_ + dx]
                tot += float(d @ d)
                n += 1
        if n == 0:
            return np.inf
        return tot / n

    hy, hx = np.nonzero(hole)
    # onion peel: hole cells by their distance to known cells
    dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    order = np.argsort(dist[hy, hx], kind="stable")
    for k in order:
        y, x = hy[k], hx[k]
        cands = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and hole[ny, nx] and filled[ny, nx]:
                    cands.append(tuple(off[ny, nx]))
        for j in rng.integers(len(sy), size=RANDOM * 2):
            cands.append((sy[j] - y, sx[j] - x))
        best = min(cands, key=lambda o: patch_cost(y, x, *o))
        off[y, x] = best
        cost[y, x] = patch_cost(y, x, *best)
        filled[y, x] = True
        if np.isfinite(cost[y, x]):
            cur[y, x] = D[y + best[0], x + best[1]]
    for it in range(ITERS):
        step = 1 if it % 2 == 0 else -1
        seq = range(len(hy)) if step == 1 else range(len(hy) - 1, -1, -1)
        for k in seq:
            y, x = hy[k], hx[k]
            best, bc = tuple(off[y, x]), cost[y, x]
            for ny, nx in ((y - step, x), (y, x - step)):  # propagation
                if 0 <= ny < H and 0 <= nx < W and hole[ny, nx]:
                    o = tuple(off[ny, nx])
                    c = patch_cost(y, x, *o)
                    if c < bc:
                        best, bc = o, c
            rad = max(H, W)
            while rad >= 1:  # random search around the best offset
                o = (best[0] + int(rng.integers(-rad, rad + 1)), best[1] + int(rng.integers(-rad, rad + 1)))
                c = patch_cost(y, x, *o)
                if c < bc:
                    best, bc = o, c
                rad //= 2
            off[y, x], cost[y, x] = best, bc
            if np.isfinite(bc):
                cur[y, x] = D[y + best[0], x + best[1]]
    return off, cost

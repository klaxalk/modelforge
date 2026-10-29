"""An octree over the semantic features (the user asked, 2026-10-27): each cell holds up to MAX_CLUSTERS embedding
vectors, the clusters of its Gaussians' features, so a search descends from the root to the cells that can match
instead of scoring every Gaussian.

**One space for everything.** A Gaussian's feature is the PCA reconstruction mean + B f (semantic.Features); a
query's cosine with it is (m.q + f.B^T q) / |m + B f|. With G the Gram matrix of [m, B] (65 x 65) = L L^T, the unit
vector z = L^T [1, f] / |L^T [1, f]| and the projected query q' = L^-1 [m.q, B^T q] give cos = z . q' exactly: the
features become plain unit vectors in 65 dimensions, clusters plain spherical k-means.

**Cells.** The seen Gaussians sorted along a Morton curve (16 levels); a cell splits while it holds more than LEAF
Gaussians. A leaf clusters its Gaussians' z into up to MAX_CLUSTERS (at least MIN_PER_CLUSTER per cluster) by
spherical k-means; an inner cell clusters its children's clusters (weighted by their counts) into up to
MAX_CLUSTERS. Each cluster stores its unit mean direction mu, an angular radius r (every member, and every member
of a child cluster, lies within r of mu: for inner cells by the triangle inequality on the sphere) and its count.

**Search** (`candidates`): from the root, a cell is entered when one of its clusters can hold a Gaussian with
relevance >= floor: the relevance (semantic.Features.relevance: min over negatives of sigmoid(T (cos_q - cos_n)))
depends on cos_q - cos_n = z . (q' - n'), which over a cap of radius r around mu is at most
|q' - n'| cos(max(0, angle(mu, q' - n') - r)). In a leaf only the members of the passing clusters are scored
(a leaf keeps its Gaussians grouped by cluster). So the leaf clusters reached hold every Gaussian at or above
the floor, and only they are scored exactly (`relevance`): the result equals the full scan wherever it is >= the
floor, the rest are 0 (nothing below the threshold is shown or grouped).

Stored as PROJECT/splat/semantic/octree.npz, keyed by features.npz (size + mtime) and the Gaussian count; built at
the end of the semantic stage, or when the Search tab first loads features without it.
"""

import os
import time

import numpy as np

from modelforge import semantic

NAME = "octree.npz"
LEAF = 2048  # a cell splits above this many Gaussians
MAX_CLUSTERS = 100  # embedding vectors per cell
MIN_PER_CLUSTER = 16  # a leaf of n Gaussians gets at most n / this clusters
DEPTH = 16  # Morton levels (cells down to 1/65536 of the scene's size)
ITERATIONS = 8  # spherical k-means
MARGIN = 1e-4  # cosine slack for float rounding in the bounds
VERSION = 2


def path(project):
    return os.path.join(project, semantic.DIR, NAME)


def _key(project, n):
    st = os.stat(os.path.join(project, semantic.DIR, "features.npz"))
    return np.array([st.st_size, int(st.st_mtime_ns), n, VERSION], np.int64)


class Space:
    """The features as unit vectors in the 1 + dims space (module doc): z per Gaussian, q' per text embedding."""

    def __init__(self, feats):
        m, B = feats.mean.astype(np.float64), feats.basis.astype(np.float64)
        Btm = B.T @ m
        G = np.block([[np.array([[m @ m]]), Btm[None]], [Btm[:, None], B.T @ B]])
        G += np.eye(len(G)) * 1e-9 * np.trace(G)
        self.L = np.linalg.cholesky(G)
        self.m, self.B = m, B

    def z(self, f):
        """Unit vectors (n, 1 + dims) of reduced features f (n, dims)."""
        u = np.c_[np.ones(len(f)), np.asarray(f, np.float64)] @ self.L
        return (u / np.linalg.norm(u, axis=1, keepdims=True)).astype(np.float32)

    def query(self, emb):
        """Projected queries (Q, 1 + dims) of unit text embeddings (Q, D): z . q' = the full-space cosine."""
        emb = np.atleast_2d(np.asarray(emb, np.float64))
        v = np.c_[emb @ self.m, emb @ self.B]
        return np.linalg.solve(self.L, v.T).T


def _spread(v):
    """Bits of v (< 2^21) moved to every third position (Morton interleave)."""
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                        (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


def morton(cells):
    """Morton codes of integer cell coordinates (n, 3)."""
    return _spread(cells[:, 0]) << np.uint64(2) | _spread(cells[:, 1]) << np.uint64(1) | _spread(cells[:, 2])


def _kmeans(z, k, w=None, iterations=ITERATIONS, rng=None):
    """Spherical k-means of unit vectors z (n, d), weights w: (centres (k', d) unit, labels (n,)); empty clusters
    dropped."""
    n = len(z)
    if k >= n:
        return z.copy(), np.arange(n)
    rng = rng or np.random.default_rng(0)
    C = z[rng.choice(n, k, replace=False)]
    w = np.ones(n, np.float32) if w is None else w.astype(np.float32)
    for _ in range(iterations):
        lab = np.argmax(z @ C.T, 1)
        S = np.zeros_like(C)
        np.add.at(S, lab, z * w[:, None])
        norm = np.linalg.norm(S, axis=1)
        keep = norm > 0
        C = np.where(keep[:, None], S / np.maximum(norm, 1e-12)[:, None], C)
    lab = np.argmax(z @ C.T, 1)
    used, lab = np.unique(lab, return_inverse=True)
    S = np.zeros((len(used), z.shape[1]), np.float32)
    np.add.at(S, lab, z * w[:, None])
    return S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12), lab


class Tree:
    """The octree (module doc). Nodes in breadth-first order (children of a node contiguous): level, Gaussian range
    [start, end) in `order`, first child and child count (0: a leaf), cluster range [c0, c1), cell corner + size.
    Clusters: mu (unit), r (angle, radians), count, members (a leaf cluster's range in `order`; -1 inner)."""

    def __init__(self, arrays):
        for k, v in arrays.items():
            setattr(self, k, v)

    # ------------------------------------------------------------ build
    @classmethod
    def build(cls, feats, means, on_line=None, rng=None):
        t0 = time.time()
        rng = rng or np.random.default_rng(0)
        space = Space(feats)
        seen = np.flatnonzero(feats.seen)
        P = np.asarray(means, np.float64)[seen]
        lo, hi = np.percentile(P, 0.1, axis=0), np.percentile(P, 99.9, axis=0)
        size = float(max((hi - lo).max(), 1e-6)) * 1.001
        lo = (lo + hi) / 2 - size / 2  # a cube; outliers clamped into the border cells
        cells = np.clip(((P - lo) / size * (1 << DEPTH)).astype(np.int64), 0, (1 << DEPTH) - 1)
        code = morton(cells)
        sort = np.argsort(code, kind="stable")
        order, code = seen[sort], code[sort]

        level, start, end, child0, nchild = [0], [0], [len(order)], [0], [0]
        i = 0
        while i < len(level):  # breadth first: a node's children are appended together
            s, e, lv = start[i], end[i], level[i]
            if e - s > LEAF and lv < DEPTH:
                shift = np.uint64(3 * (DEPTH - lv - 1))
                digit = (code[s:e] >> shift) & np.uint64(7)
                cuts = s + np.searchsorted(digit, np.arange(9, dtype=np.uint64))
                child0[i], nchild[i] = len(level), 0
                for a, b in zip(cuts[:-1], cuts[1:]):
                    if b > a:
                        level.append(lv + 1)
                        start.append(int(a))
                        end.append(int(b))
                        child0.append(0)
                        nchild.append(0)
                        nchild[i] += 1
            i += 1
        level, start, end = np.array(level), np.array(start), np.array(end)
        child0, nchild = np.array(child0), np.array(nchild)
        n_nodes = len(level)

        per = [None] * n_nodes  # (mu, r, count) per node
        leaves = np.flatnonzero(nchild == 0)
        for j, i in enumerate(leaves):
            idx = order[start[i]:end[i]]
            z = space.z(feats.f[idx])
            k = int(min(MAX_CLUSTERS, max(1, len(z) // MIN_PER_CLUSTER)))
            sample = z if len(z) <= 8 * LEAF else z[rng.choice(len(z), 8 * LEAF, replace=False)]
            mu, _ = _kmeans(sample, k, rng=rng)
            lab = np.argmax(z @ mu.T, 1)  # every member assigned, so the radius covers all of them
            cos = np.einsum("nd,nd->n", z, mu[lab])
            r = np.zeros(len(mu), np.float32)
            np.maximum.at(r, lab, np.arccos(np.clip(cos, -1, 1)).astype(np.float32))
            cnt = np.bincount(lab, minlength=len(mu))
            keep = cnt > 0
            within = np.argsort(lab, kind="stable")  # the leaf's Gaussians grouped by cluster: a range per cluster
            order[start[i]:end[i]] = idx[within]
            ranges = start[i] + np.r_[0, np.cumsum(cnt[keep])]
            per[i] = (mu[keep], r[keep] + MARGIN, cnt[keep], np.c_[ranges[:-1], ranges[1:]])
            if on_line and j % 500 == 499:
                on_line(f"semantic octree: leaf {j + 1:,} / {len(leaves):,}")
        for i in np.flatnonzero(nchild > 0)[::-1]:  # bottom up: children come after their parent
            kids = range(child0[i], child0[i] + nchild[i])
            mu = np.concatenate([per[c][0] for c in kids])
            r = np.concatenate([per[c][1] for c in kids])
            cnt = np.concatenate([per[c][2] for c in kids])
            if len(mu) > MAX_CLUSTERS:
                pmu, lab = _kmeans(mu, MAX_CLUSTERS, w=cnt, rng=rng)
                ang = np.arccos(np.clip(np.einsum("nd,nd->n", mu, pmu[lab]), -1, 1)) + r
                pr = np.zeros(len(pmu), np.float32)
                np.maximum.at(pr, lab, np.minimum(ang, np.pi).astype(np.float32))
                per[i] = (pmu, pr + MARGIN, np.bincount(lab, weights=cnt, minlength=len(pmu)).astype(np.int64))
            else:
                per[i] = (mu, r, cnt)
            per[i] = per[i][:3] + (np.full((len(per[i][0]), 2), -1, np.int64),)
        nc = np.array([len(p[0]) for p in per])
        c1 = np.cumsum(nc)
        corner = lo + (cells[sort][start] >> (DEPTH - level)[:, None] << (DEPTH - level)[:, None]) * size / (1 << DEPTH)
        tree = cls(dict(
            level=level.astype(np.int16), start=start.astype(np.int64), end=end.astype(np.int64),
            child0=child0.astype(np.int64), nchild=nchild.astype(np.int16), c0=(c1 - nc).astype(np.int64),
            c1=c1.astype(np.int64), corner=corner.astype(np.float32),
            size=(size / (1 << level.astype(np.int64))).astype(np.float32), order=order.astype(np.int64),
            mu=np.concatenate([p[0] for p in per]).astype(np.float32),
            r=np.concatenate([p[1] for p in per]).astype(np.float32),
            count=np.concatenate([p[2] for p in per]).astype(np.int64),
            members=np.concatenate([p[3] for p in per]).astype(np.int64),
            n=np.int64(len(feats.f)), seconds=np.float64(time.time() - t0)))
        tree.space = space
        if on_line:
            on_line(f"semantic octree: {n_nodes:,} cells ({len(leaves):,} leaves, depth {int(level.max())}), "
                    f"{len(tree.mu):,} cluster vectors for {len(order):,} gaussians, {tree.seconds:.1f} s")
        return tree

    # ------------------------------------------------------------ storage
    FIELDS = ("level", "start", "end", "child0", "nchild", "c0", "c1", "corner", "size", "order", "mu", "r", "count",
              "members", "n", "seconds")

    def save(self, project, key):
        out = path(project)
        tmp = out + ".part.npz"
        np.savez(tmp, key=key, **{k: getattr(self, k) for k in self.FIELDS})
        os.replace(tmp, out)

    @classmethod
    def load(cls, project, key, feats):
        try:
            d = np.load(path(project))
            if not np.array_equal(d["key"], key):
                return None
            tree = cls({k: d[k] for k in cls.FIELDS})
        except (OSError, KeyError, ValueError):
            return None
        tree.space = Space(feats)
        return tree

    # ------------------------------------------------------------ search
    def upper_bounds(self, clusters, qp, negp, temperature=semantic.TEMPERATURE):
        """Upper bound of the relevance of any member of the given clusters (indices). The relevance depends on
        cos_q - cos_n = z . (q' - n') per negative, bounded over the cluster's cap directly: q' and n' are close
        (text embeddings share most of their direction), so the difference is much tighter than bounding the two
        cosines apart."""
        mu, r = self.mu[clusters].astype(np.float64), self.r[clusters].astype(np.float64)
        D = qp[None] - negp  # (negatives, d)
        nd = np.maximum(np.linalg.norm(D, axis=1), 1e-12)
        ang = np.arccos(np.clip(mu @ (D / nd[:, None]).T, -1, 1))  # (clusters, negatives)
        ub = nd[None] * np.cos(np.maximum(ang - r[:, None], 0.0)) + MARGIN
        return (1.0 / (1.0 + np.exp(-temperature * ub))).min(1)

    def candidates(self, q_emb, neg_embs, floor):
        """Indices of the Gaussians in the leaf clusters that can hold a relevance >= floor, and the search's
        stats."""
        P = self.space.query(np.vstack([q_emb, neg_embs]))
        qp, negp = P[0], P[1:]
        nodes, hits, visited = np.array([0]), [], 0
        while len(nodes):
            visited += len(nodes)
            counts = self.c1[nodes] - self.c0[nodes]
            first = np.r_[0, np.cumsum(counts)[:-1]]
            cl = np.repeat(self.c0[nodes] - first, counts) + np.arange(counts.sum())
            ok = self.upper_bounds(cl, qp, negp) >= floor
            leaf = np.repeat(self.nchild[nodes] == 0, counts)
            hits.append(cl[ok & leaf])  # a leaf: only its passing clusters are scored
            enter = np.maximum.reduceat(ok, first)
            inner = nodes[enter & (self.nchild[nodes] > 0)]
            nodes = (np.repeat(self.child0[inner] - np.r_[0, np.cumsum(self.nchild[inner])[:-1]],
                               self.nchild[inner]) + np.arange(int(self.nchild[inner].sum())))
        hits = np.concatenate(hits)
        m = self.members[hits]
        idx = (np.concatenate([self.order[a:b] for a, b in m]) if len(m) else np.zeros(0, np.int64))
        return idx, {"cells_visited": int(visited), "clusters": int(len(hits)), "scored": int(len(idx)),
                     "seen": int(len(self.order))}

    def relevance(self, feats, q_emb, neg_embs, floor):
        """semantic.Features.relevance where it is >= floor (exactly), 0 elsewhere; and the search's stats."""
        idx, stats = self.candidates(q_emb, neg_embs, floor)
        out = np.zeros(len(feats.f), np.float32)
        out[idx] = feats.relevance(q_emb, neg_embs, idx=idx)
        return out, stats


def load_or_build(project, feats, means, on_line=None):
    """The project's octree: loaded when it matches features.npz, else built and saved."""
    key = _key(project, len(feats.f))
    tree = Tree.load(project, key, feats)
    if tree is None:
        tree = Tree.build(feats, means, on_line=on_line)
        tree.save(project, key)
    return tree


def build_for(project, on_line=print):
    """Build (or load) the octree of a project's current features (the semantic stage's last step)."""
    import plyfile

    from modelforge import splat_mesh
    feats = semantic.Features(project)
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data
    return load_or_build(project, feats, np.stack([v["x"], v["y"], v["z"]], 1), on_line=on_line)

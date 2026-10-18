"""A small TF buffer for ROS 2 bags: static transforms + timestamped dynamic ones, lookups along the
frame tree with interpolation (linear translation, quaternion slerp), no ROS needed.

Conventions follow tf2: a TransformStamped with header.frame_id = parent and child_frame_id = child
is T_parent_child, which maps points from the child frame into the parent frame
(p_parent = R p_child + t). lookup(target, source, t) returns T_target_source.
"""

from collections import defaultdict, deque

import numpy as np


def quat_to_rot(q):
    """(x, y, z, w) -> 3x3."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    """3x3 -> (x, y, z, w)."""
    m = np.asarray(R, float)
    tr = np.trace(m)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def slerp(q0, q1, a):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def make(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return make(R.T, -R.T @ t)


class TFBuffer:
    def __init__(self):
        self.static = {}  # (parent, child) -> 4x4
        self.dynamic = defaultdict(list)  # (parent, child) -> [(t, translation, quat)]
        self._sorted = True

    def add(self, parent, child, t, translation, quat, static=False):
        parent, child = parent.lstrip("/"), child.lstrip("/")
        if static:
            self.static[(parent, child)] = make(quat_to_rot(quat), translation)
        else:
            self.dynamic[(parent, child)].append((float(t), np.asarray(translation, float), np.asarray(quat, float)))
            self._sorted = False

    def _finalize(self):
        if not self._sorted:
            for k, v in self.dynamic.items():
                v.sort(key=lambda x: x[0])
                self.dynamic[k] = v
            self._times = {k: np.array([x[0] for x in v]) for k, v in self.dynamic.items()}
            self._sorted = True

    @property
    def frames(self):
        out = set()
        for p, c in list(self.static) + list(self.dynamic):
            out.update((p, c))
        return sorted(out)

    def time_range(self, parent, child):
        self._finalize()
        v = self.dynamic.get((parent, child))
        return (v[0][0], v[-1][0]) if v else (-np.inf, np.inf)

    def _edge(self, parent, child, t, tol):
        """T_parent_child at time t (None outside the recorded range, beyond tol seconds)."""
        if (parent, child) in self.static:
            return self.static[(parent, child)]
        self._finalize()
        v, times = self.dynamic[(parent, child)], self._times[(parent, child)]
        i = int(np.searchsorted(times, t))
        if i == 0:
            return make(quat_to_rot(v[0][2]), v[0][1]) if times[0] - t <= tol else None
        if i == len(v):
            return make(quat_to_rot(v[-1][2]), v[-1][1]) if t - times[-1] <= tol else None
        (t0, p0, q0), (t1, p1, q1) = v[i - 1], v[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return make(quat_to_rot(slerp(q0, q1, a)), p0 + a * (p1 - p0))

    def path(self, target, source):
        """Frame chain target ... source as [(a, b, forward)] edges, or None if not connected."""
        nbr = defaultdict(list)
        for p, c in list(self.static) + list(self.dynamic):
            nbr[p].append((c, True))  # parent -> child: T_p_c
            nbr[c].append((p, False))
        prev, q = {target: None}, deque([target])
        while q:
            f = q.popleft()
            if f == source:
                break
            for g, fwd in nbr[f]:
                if g not in prev:
                    prev[g] = (f, fwd)
                    q.append(g)
        if source not in prev:
            return None
        edges, f = [], source
        while prev[f] is not None:
            g, fwd = prev[f]
            edges.append((g, f, fwd))
            f = g
        return edges[::-1]

    def lookup(self, target, source, t, tol=0.1):
        """T_target_source at time t, or None (not connected, or t outside a dynamic edge's range)."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        edges = self.path(target, source)
        if edges is None:
            return None
        T = np.eye(4)
        for a, b, fwd in edges:  # a is closer to target
            E = self._edge(a, b, t, tol) if fwd else self._edge(b, a, t, tol)
            if E is None:
                return None
            T = T @ (E if fwd else inv(E))
        return T

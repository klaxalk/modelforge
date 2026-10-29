"""Articulated objects in the splat (docs/todo_rigs.md; the user asked, 2026-10-27): an object found by the
semantic search is cut out of the splat, the ground under it is filled, its rigid parts are hinged by joints, and it
is shown and exported in any pose.

**Cut out** (`cut_out`): the search's object (its Gaussians at or above the threshold) seeds a region grown over a
voxel grid (VOXEL) through neighbouring voxels above the ground whose best relevance is at least `grow_threshold`
(better than the generic words and than the background around the object), never into the search's other objects
(walls), the search box capped at the object's radius + 10 m, and at the end only what hangs on the seeds through
opaque Gaussians (`_solid`): the parts the query matched weakly (thin frames, belts) join, the gravel and the ground
around do not. Then every Gaussian in the region (dilated by a voxel) above the ground joins, faint ones too. The
ground is a height map (`Ground`: the median of the opaque Gaussians per cell), made twice: first without the
seeds (for the growing), then without the grown core's footprint, inpainted under it (under the object a cell sees
mostly the object: the first ground sits on it there, and the object's faint parts would be left as a ghost).

**Fill** (`fill`): flat opaque Gaussians on the ground height map over the object's footprint, coloured from the
ground around it (the remaining splat's colours rasterized from above, the footprint inpainted). The ground under
the object was never seen: plausible, not measured.

**Joints and parts.** A joint hinges (revolute, degrees) or slides (prismatic, metres) a part off its parent part,
about an axis through `origin` (rest pose, ENU). Its part is cut from the parent: the parent's Gaussians on the
seed's side of a plane through `origin` (`cut_normal`: the plane holds the axis, or is across it when the seed, a
point clicked on the moving part, lies along it; for a turntable also what hangs below its ring beyond the base's
`footprint`, an excavator's arm), the voxel-connected piece containing the seed (`assign_parts`). Part 0 is the rest of the object. Poses compose along
the chain (`transforms`): a part's transform = its parent's x its joint's.

**Output.** PROJECT/rig/<name>/: rig.json (query, threshold, splat fingerprint, joints with their values),
object.npy (indices of the object's Gaussians in splat_enu.ply), fill.npz (the fill Gaussians), labels.npy (part per
object Gaussian). `posed_ply` writes the whole splat with the object posed (positions, rotations and SH bands
rotated per part) and the fill. The GUI (gui/rig.py) draws the parts as their own splat nodes and moves them by
their node poses.
"""

import json
import os
import shutil
import time

import numpy as np

from modelforge import splat_mesh
from modelforge.export import splat_to_enu

DIR = "rig"
VOXEL = 0.2  # m: the region grown over this grid
PART_VOXEL = 0.12  # m: the connectivity of a part's cut
GROUND_CELL = 0.5  # m
GROUND_PCT = 0.5  # the ground height of a cell: this quantile of its opaque Gaussians' heights (thick, fuzzy ground
# in a quarry, kladno_bagr: at 0.2, 68 % of the Gaussians around an excavator counted as above the ground)
FILL_CELL = 0.08  # m: one fill Gaussian per cell
ABOVE = 0.25  # m: Gaussians this far above the ground count as the object
GROW = 0.5  # relevance a voxel needs to join the grown region (0.5 = no better than the generic words)
PAD = 3.0  # m around the object's marker searched
MAX_EXTRA = 10.0  # m: the search box grows to at most the object's radius + this
SH_C0 = 0.28209479177387814


def rig_dir(project, name):
    return os.path.join(project, DIR, name)


def list_rigs(project):
    d = os.path.join(project, DIR)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.exists(os.path.join(d, n, "rig.json")))


def splat_key(project):
    st = os.stat(splat_mesh.splat_path(project))
    return [st.st_size, round(st.st_mtime, 3)]


def load(project, name):
    return json.load(open(os.path.join(rig_dir(project, name), "rig.json")))


def save(project, name, rig):
    d = rig_dir(project, name)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "rig.json.part")
    json.dump(rig, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(d, "rig.json"))


def delete(project, name):
    shutil.rmtree(rig_dir(project, name), ignore_errors=True)


def current(project, name):
    """The rig belongs to the current splat (its indices are valid)."""
    try:
        return load(project, name).get("splat") == splat_key(project)
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------- ground
def fill_holes(grid, known, relax=200):
    """Unknown cells of a float grid filled smoothly from the known ones: spread outwards by normalized 3 x 3
    averaging, then relaxed towards the average of their neighbours (a Laplace fill), the known cells fixed.
    (cv2.inpaint returns garbage on float grids.)"""
    import cv2
    g = np.where(known, grid, 0).astype(np.float32)
    if not known.any():
        return g
    w = known.astype(np.float32)
    have = known.copy()
    while not have.all():
        s = cv2.boxFilter(g * w, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        n = cv2.boxFilter(w, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        new = ~have & (n > 0)
        g[new] = s[new] / n[new]
        w[new] = 1.0
        have |= new
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for _ in range(relax if not known.all() else 0):
        g = np.where(known, g, cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REPLICATE))
    return g


class Ground:
    """Height map of the ground from Gaussians (positions, opacity in [0, 1]) over box: per GROUND_CELL cell the
    20th percentile height of the opaque ones not excluded; cells without them filled from around them
    (`fill_holes`). Read bilinearly."""

    def __init__(self, means, opacity, exclude, box, cell=GROUND_CELL, pct=None):
        x0, y0, x1, y1 = box
        self.x0, self.y0, self.cell = x0, y0, cell
        W, H = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
        m = (opacity > 0.5) & ~exclude & (means[:, 0] >= x0) & (means[:, 0] < x1) & (means[:, 1] >= y0) & \
            (means[:, 1] < y1)
        p = means[m]
        c = ((p[:, 1] - y0) // cell).astype(int) * W + ((p[:, 0] - x0) // cell).astype(int)
        order = np.lexsort((p[:, 2], c))
        c, z = c[order], p[order, 2]
        cells, first, counts = np.unique(c, return_index=True, return_counts=True)
        z20 = z[first + (counts * (GROUND_PCT if pct is None else pct)).astype(int)]
        grid = np.zeros(H * W, np.float32)
        known = np.zeros(H * W, bool)
        ok = counts >= 3
        grid[cells[ok]], known[cells[ok]] = z20[ok], True
        grid, known = grid.reshape(H, W), known.reshape(H, W)
        self.grid = fill_holes(grid, known)

    def __call__(self, x, y):
        H, W = self.grid.shape
        u = np.clip((np.asarray(x, float) - self.x0) / self.cell - 0.5, 0, W - 1.001)
        v = np.clip((np.asarray(y, float) - self.y0) / self.cell - 0.5, 0, H - 1.001)
        j, i = u.astype(int), v.astype(int)
        fu, fv = u - j, v - i
        g = self.grid
        return ((1 - fv) * ((1 - fu) * g[i, j] + fu * g[i, j + 1]) + fv * ((1 - fu) * g[i + 1, j] + fu * g[i + 1, j + 1]))


# ---------------------------------------------------------------------------------------------- cut out
def _neighbours(keys):
    """Pairs (a, b) of 26-neighbouring voxels among unique integer keys (M, 3), each pair once."""
    lo = keys.min(0)
    dims = keys.max(0) - lo + 3
    lin = ((keys - lo + 1) * np.array([dims[1] * dims[2], dims[2], 1])).sum(1)
    order = np.argsort(lin)
    slin = lin[order]
    a_idx, b_idx = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                nb = lin + dx * dims[1] * dims[2] + dy * dims[2] + dz
                pos = np.clip(np.searchsorted(slin, nb), 0, len(keys) - 1)
                hit = slin[pos] == nb
                a_idx.append(np.flatnonzero(hit))
                b_idx.append(order[pos[hit]])
    return np.concatenate(a_idx), np.concatenate(b_idx)


_OFFSETS = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)


def _member(keys, table):
    """Whether each integer key (n, 3) is a row of table (m, 3, unique)."""
    lo = np.minimum(keys.min(0), table.min(0))
    span = np.maximum(keys.max(0), table.max(0)) - lo + 1
    lin = lambda k: ((k - lo) * np.array([span[1] * span[2], span[2], 1])).sum(1)  # noqa: E731
    t = np.sort(lin(table))
    q = lin(keys)
    pos = np.clip(np.searchsorted(t, q), 0, len(t) - 1)
    return t[pos] == q


def grow(keys, seed, allowed):
    """Voxels (unique keys) reached from the seed voxels (bool) through 26-neighbours that are allowed (bool)."""
    a, b = _neighbours(keys)
    inside = seed.copy()
    while True:
        new = inside.copy()
        new[b[inside[a] & allowed[b]]] = True
        new[a[inside[b] & allowed[a]]] = True
        if np.array_equal(new, inside):
            return inside
        inside = new


def cut_out(means, opacity, rel, thr, obj, grow_thr=None, voxel=VOXEL, above=ABOVE, pad=PAD, tries=4, others=()):
    """Indices of the object's Gaussians and its Ground: obj is a semantic.objects entry (center, radius, z0,
    z1). Seeds: Gaussians >= thr within the object's cylinder; grown over voxels above the ground (module doc).
    `others`: the search's other objects: their matches (dilated by a voxel) are walls the growing cannot enter, so
    a neighbouring instance (another excavator, a truck) never joins over the ground between them. The search box
    (the cylinder + pad) grows x 1.5 while the object reaches its edge (weakly matched parts can stick out far
    beyond the matches), up to the object's radius + MAX_EXTRA metres (on leaky ground it kept growing)."""
    r = float(obj["radius"])
    reach = r * 1.5 + pad
    cap = max(reach, r + MAX_EXTRA)
    if grow_thr is None:
        grow_thr = grow_threshold(means, rel, obj)
    for _ in range(tries):
        idx, ground, touches = _cut(means, opacity, rel, thr, obj, reach, grow_thr, voxel, above, pad, others)
        if not touches or reach >= cap:
            break
        reach = min(reach * 1.5, cap)
    return idx, ground


def grow_threshold(means, rel, obj, width=15.0):
    """The relevance a voxel needs to join: better than the generic words (GROW) and than the background around the
    object (the median over a ring `width` m wide outside its marker, + 0.02). For "excavator" in a quarry the ground
    itself scores 0.535 (kladno_bagr), for "a machine" around the screener 0.551 (kladno_dopravnik)."""
    c = np.asarray(obj["center"], float)
    d = np.linalg.norm(means[:, :2] - c[:2], axis=1)
    r = float(obj["radius"])
    ring = (d > 1.5 * r) & (d < 1.5 * r + width)
    return max(GROW, float(np.median(rel[ring])) + 0.02) if ring.sum() > 100 else GROW


def _cut(means, opacity, rel, thr, obj, reach, grow_thr, voxel, above, pad, others=()):
    c, r = np.asarray(obj["center"], float), float(obj["radius"])
    box = (c[0] - reach, c[1] - reach, c[0] + reach, c[1] + reach)
    near = np.flatnonzero((np.abs(means[:, 0] - c[0]) <= reach) & (np.abs(means[:, 1] - c[1]) <= reach)
                          & (means[:, 2] <= obj["z1"] + pad) & (means[:, 2] >= obj["z0"] - pad))
    P, rel_n, op_n = means[near], rel[near], opacity[near]
    seed_g = (np.linalg.norm(P[:, :2] - c[:2], axis=1) <= r * 1.2) & (rel_n >= thr)
    wall = np.zeros(len(P), bool)  # the other instances' matches
    for o in others:
        oc = np.asarray(o["center"], float)
        wall |= (np.linalg.norm(P[:, :2] - oc[:2], axis=1) <= float(o["radius"]) * 1.2) & (rel_n >= thr)
    wall &= ~seed_g
    walls = (np.unique((np.unique(np.floor(P[wall] / voxel).astype(np.int64), axis=0)[:, None, :] + _OFFSETS[None])
                       .reshape(-1, 3), axis=0) if wall.any() else None)

    def grown(ground, seeds, within=True):
        use = (P[:, 2] - ground(P[:, 0], P[:, 1]) > above) & (op_n > 0.05) & within
        keys, inv = np.unique(np.floor(P[use] / voxel).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        vrel = np.zeros(len(keys))
        np.maximum.at(vrel, inv, rel_n[use])
        vseed = np.zeros(len(keys), bool)
        vseed[inv[seeds[use]]] = True
        allowed = vrel >= grow_thr
        if walls is not None:
            allowed &= ~_member(keys, walls)
        inside = grow(keys, vseed, allowed)
        return keys[inside], np.flatnonzero(use)[inside[inv]]

    # 1. the core: grown from the seeds over voxels above a first ground (the seeds left out of it)
    _, core = grown(Ground(P, op_n, seed_g, box), seed_g)
    # 2. the ground again without the core's footprint, inpainted under it over short distances: under the object a
    # cell sees mostly the object, so the first ground sat on it there (its lower parts and faint ones were left
    # behind as a ghost); grown again over that ground from the core, within its footprint
    cell = GROUND_CELL
    fk = np.unique(np.floor(P[core, :2] / cell).astype(np.int64), axis=0)
    fk = np.unique((fk[:, None, :] + np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])[None]).reshape(-1, 2),
                   axis=0)
    in_foot = _member2(np.floor(P[:, :2] / cell).astype(np.int64), fk)
    ground = Ground(P, op_n, in_foot, box)
    is_core = np.zeros(len(P), bool)
    is_core[core] = True
    keys, _ = grown(ground, is_core, in_foot)  # down into the lower parts, not sideways over the ground
    # 3. every Gaussian in those voxels (dilated by one) above that ground, faint ones too; and every one above
    # that ground under the object's voxel columns (its underside, the tracks: weakly matched, below its matches)
    region = (keys[:, None, :] + _OFFSETS[None]).reshape(-1, 3)
    up = P[:, 2] - ground(P[:, 0], P[:, 1]) > above
    vk = np.floor(P / voxel).astype(np.int64)
    cols = np.unique(keys[:, :2], axis=0)
    top = np.full(len(cols), -np.inf)
    np.maximum.at(top, np.searchsorted(cols[:, 0] * (1 << 32) + cols[:, 1], keys[:, 0] * (1 << 32) + keys[:, 1]),
                  keys[:, 2])
    lin = cols[:, 0] * (1 << 32) + cols[:, 1]
    q = vk[:, 0] * (1 << 32) + vk[:, 1]
    pos = np.clip(np.searchsorted(lin, q), 0, len(lin) - 1)
    under = (lin[pos] == q) & (vk[:, 2] <= top[pos])
    keep = up & (_member(vk, np.unique(region, axis=0)) | under)
    if walls is not None:
        keep &= ~_member(vk, walls)
    keep &= _solid(P, op_n, keep, seed_g, voxel)
    sel = np.flatnonzero(keep)
    edge = np.abs(P[sel, :2] - c[:2]).max() > reach - 2 * voxel if len(sel) else False
    return near[sel], ground, edge


def _solid(P, op, keep, seeds, voxel, solid=0.3):
    """The part of the selection connected to the seeds through opaque Gaussians (opacity > solid): gravel and rocks
    beside an object hang on to it through faint fuzz only (kladno_bagr's excavator). Faint Gaussians inside the
    kept voxels (dilated by one) stay."""
    hard = keep & (op > solid)
    if not (hard & seeds).any():
        return keep
    keys, inv = np.unique(np.floor(P[hard] / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    start = np.zeros(len(keys), bool)
    start[inv[seeds[hard]]] = True
    inside = grow(keys, start, np.ones(len(keys), bool))
    kept = np.unique((keys[inside][:, None, :] + _OFFSETS[None]).reshape(-1, 3), axis=0)
    return keep & _member(np.floor(P / voxel).astype(np.int64), kept)


def _member2(keys, table):
    return _member(np.c_[keys, np.zeros(len(keys), np.int64)], np.c_[table, np.zeros(len(table), np.int64)])


# ---------------------------------------------------------------------------------------------- fill
def fill(means, rgb, opacity, obj_mask, ground, cell=FILL_CELL, margin=0.3):
    """Flat Gaussians on the ground over the object's footprint (module doc): dict of means, quats, scales,
    opacities (logit), sh0 (N, 1, 3)."""
    import cv2
    P = means[obj_mask]
    x0, y0 = P[:, :2].min(0) - margin - 1.0
    x1, y1 = P[:, :2].max(0) + margin + 1.0
    W, H = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    foot = np.zeros((H, W), np.uint8)
    foot[((P[:, 1] - y0) // cell).astype(int), ((P[:, 0] - x0) // cell).astype(int)] = 1
    k = int(np.ceil(margin / cell)) * 2 + 1
    foot = cv2.dilate(foot, np.ones((k, k), np.uint8))
    # the ground around, from above: colours of the opaque non-object Gaussians near the ground height
    m = ~obj_mask & (opacity > 0.5) & (means[:, 0] >= x0) & (means[:, 0] < x1) & (means[:, 1] >= y0) & \
        (means[:, 1] < y1)
    m &= np.abs(means[:, 2] - ground(means[:, 0], means[:, 1])) < 0.3
    i = ((means[m, 1] - y0) // cell).astype(int)
    j = ((means[m, 0] - x0) // cell).astype(int)
    acc = np.zeros((H, W, 3))
    n = np.zeros((H, W))
    np.add.at(acc, (i, j), rgb[m])
    np.add.at(n, (i, j), 1)
    img = (np.clip(acc / np.maximum(n, 1)[..., None], 0, 1) * 255).astype(np.uint8)
    hole = ((n == 0) | (foot > 0)).astype(np.uint8)
    img = cv2.inpaint(img, hole, 7, cv2.INPAINT_TELEA)
    ii, jj = np.nonzero(foot)
    x = x0 + (jj + 0.5) * cell
    y = y0 + (ii + 0.5) * cell
    col = img[ii, jj].astype(np.float32) / 255
    N = len(x)
    return {"means": np.c_[x, y, ground(x, y) + 0.02].astype(np.float32),
            "quats": np.tile(np.array([1.0, 0, 0, 0], np.float32), (N, 1)),
            "scales": np.log(np.tile([cell * 0.7, cell * 0.7, cell * 0.05], (N, 1))).astype(np.float32),
            "opacities": np.full(N, 4.0, np.float32),  # sigmoid 0.98
            "sh0": ((col - 0.5) / SH_C0)[:, None, :].astype(np.float32)}


# ---------------------------------------------------------------------------------------------- joints
def rotation(axis, angle_rad):
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def joint_matrix(joint, value):
    T = np.eye(4)
    o, a = np.asarray(joint["origin"], float), np.asarray(joint["axis"], float)
    a = a / max(np.linalg.norm(a), 1e-12)
    if joint["type"] == "prismatic":
        T[:3, 3] = a * value
    else:
        R = rotation(a, np.radians(value))
        T[:3, :3], T[:3, 3] = R, o - R @ o
    return T


def transforms(joints, values=None):
    """4 x 4 world transform per part (part 0 the base: identity; part k + 1 moved by joint k)."""
    out = [np.eye(4)]
    for k, j in enumerate(joints):
        v = j.get("value", 0.0) if values is None else values[k]
        out.append(out[j.get("parent", 0)] @ joint_matrix(j, v))
    return out


def cut_normal(origin, axis, seed):
    """Normal of a joint's cut plane through origin: towards the seed, perpendicular to the axis (the plane holds
    the hinge); along the axis when the seed lies mostly along it (a turntable: the part above the ring)."""
    a = axis / max(np.linalg.norm(axis), 1e-12)
    v = seed - origin
    perp = v - (v @ a) * a
    n = perp if np.linalg.norm(perp) >= 0.5 * np.linalg.norm(v) else np.sign(v @ a or 1.0) * a
    return n / max(np.linalg.norm(n), 1e-12)


def ring(means, click, reach=4.0, above=1.5):
    """A turntable's pin from a click on the machine's side at the gap where the upper body sits on its base: the
    click's height, and the centre of the upper body just above it (the median of the Gaussians within `reach` m
    horizontally, up to `above` m higher, re-centred until it settles: from any side of kladno_bagr's excavator the
    same centre within 5 cm). The shape itself shows no neck (an excavator's house sits right on its ring:
    kladno_bagr's cross-section stays ~14 m² from the tracks to the house). Returns (x, y, z)."""
    c = np.asarray(click, float)
    band = means[(means[:, 2] >= c[2]) & (means[:, 2] <= c[2] + above)]
    xy = c[:2].copy()
    for _ in range(10):  # re-centred on the estimate: from a click on one side the window sees half the house
        sl = band[np.linalg.norm(band[:, :2] - xy, axis=1) < reach]
        if len(sl) < 20:
            break
        new = np.median(sl[:, :2], 0)  # the median: the middle of the 5-95 % box was pulled by the boom's foot
        if np.linalg.norm(new - xy) < 0.05:
            xy = new
            break
        xy = new
    return np.array([xy[0], xy[1], c[2]])


def footprint(r, step=0.25, start=1.0, thin=0.1):
    """Radius of the base under a turntable: from the horizontal distances r of the Gaussians below the ring, the
    first ring (step wide, beyond `start`) holding < `thin` x the densest ring's Gaussians per area (the tracks end
    there; an arm's bucket further out is not the base)."""
    if len(r) < 20:
        return float(np.max(r)) if len(r) else 0.0
    edges = np.arange(0, r.max() + step, step)
    cnt, _ = np.histogram(r, edges)
    dens = cnt / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))
    peak = dens.max()
    for k in range(len(dens)):
        if edges[k] >= start and dens[k] < thin * peak:
            return float(edges[k])
    return float(edges[-1])


def _on_floor(P, below, voxel, reach=0.5):
    """Of the points `below` (bool), those voxel-connected (among themselves) to the floor: within `reach` m of the
    lowest of them. True for every point not below."""
    out = np.ones(len(P), bool)
    idx = np.flatnonzero(below)
    if not len(idx):
        return out
    Q = P[idx]
    keys, inv = np.unique(np.floor(Q / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    floor = np.percentile(Q[:, 2], 1)
    start = np.zeros(len(keys), bool)
    start[inv[Q[:, 2] < floor + reach]] = True
    reached = grow(keys, start, np.ones(len(keys), bool))
    out[idx] = reached[inv]
    return out


def assign_parts(means, joints, voxel=PART_VOXEL, parts=None):
    """Part label per object Gaussian (module doc): joint k cuts part k + 1 from its parent; a joint naming a
    PartField part ("part", rig_partfield) takes that part's Gaussians (parts: the part id per Gaussian)."""
    labels = np.zeros(len(means), np.int32)
    for k, j in enumerate(joints):
        if "part" in j:
            if parts is not None and len(parts) == len(labels):
                labels[parts == j["part"]] = k + 1
            continue
        cand = np.flatnonzero(labels == j.get("parent", 0))
        if not len(cand):
            continue
        o, s = np.asarray(j["origin"], float), np.asarray(j["seed"], float)
        a = np.asarray(j["axis"], float)
        n = cut_normal(o, a, s)
        if j.get("slew"):  # a turntable always cuts across its axis (a door's vertical hinge cuts along it)
            au = a / max(np.linalg.norm(a), 1e-12)
            n = au * (1.0 if (s - o) @ au >= 0 else -1.0)
        side = (means[cand] - o) @ n > 0
        if abs(n @ a) > 0.9 and abs(a[2] / max(np.linalg.norm(a), 1e-12)) > 0.9:  # a turntable cut across its axis
            r = np.linalg.norm((means[cand] - o)[:, :2], axis=1)
            side |= r > footprint(r[~side])  # what hangs below the ring beyond the base (an excavator's arm)
            side |= ~_on_floor(means[cand], ~side, voxel)  # ... and what hangs below it without standing on the
            # ground (an excavator's counterweight): only what reaches the floor below the ring stays the base
        cand = cand[side]
        if not len(cand):
            continue
        keys, inv = np.unique(np.floor(means[cand] / voxel).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        start = np.zeros(len(keys), bool)  # the candidates near the seed (a single nearest may be a stray speck)
        dist = np.linalg.norm(means[cand] - s, axis=1)
        start[inv[dist <= max(0.3, float(dist.min()))]] = True
        inside = grow(keys, start, np.ones(len(keys), bool))
        labels[cand[inside[inv]]] = k + 1
    return labels


def new_joint(origin, axis_point, seed, parent=0, kind="revolute", name=None, limits=None):
    """A joint from three clicks: the axis through origin and axis_point, the moving part at seed."""
    o, b = np.asarray(origin, float), np.asarray(axis_point, float)
    axis = b - o
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("the two axis points coincide")
    lim = limits or ([-90.0, 90.0] if kind == "revolute" else [-2.0, 2.0])
    return {"name": name or kind, "type": kind, "parent": int(parent), "origin": o.tolist(),
            "axis": (axis / np.linalg.norm(axis)).tolist(), "seed": list(map(float, seed)), "limits": lim,
            "value": 0.0}


# ---------------------------------------------------------------------------------------------- animation
DEMO_RATE = {"revolute": 30.0, "prismatic": 0.5}  # degrees / metres per second in the demo sweep


def demo_track(joints, hold=0.5):
    """Keyframes (time s, values per joint) of a demo: each joint in turn from rest to its upper limit, its lower
    limit and back to rest, then all together to their upper limits and back; a short hold between moves."""
    rest = [0.0] * len(joints)
    keys, t = [(0.0, rest)], 0.0

    def go(values, dt):
        nonlocal t
        t += max(dt, 0.5)
        keys.append((t, list(values)))
        t += hold
        keys.append((t, list(values)))

    def span(k, a, b):
        return abs(b - a) / DEMO_RATE.get(joints[k]["type"], 30.0)

    for k, j in enumerate(joints):
        lo, hi = j["limits"]
        cur = 0.0
        for target in (hi, lo, 0.0):
            v = list(rest)
            v[k] = float(target)
            go(v, span(k, cur, target))
            cur = target
    if len(joints) > 1:
        top = [float(j["limits"][1]) for j in joints]
        dt = max(span(k, 0, top[k]) for k in range(len(joints)))
        go(top, dt)
        go(rest, dt)
    return keys


def keyframe_track(frames, step):
    """Keyframes (time, values) of user poses `step` seconds apart, back to the first one at the end (a loop)."""
    if not frames:
        return []
    keys = [(k * step, list(f)) for k, f in enumerate(frames)]
    keys.append((len(frames) * step, list(frames[0])))
    return keys


def transition_time(joints, a, b):
    """Seconds to move smoothly from pose a to pose b at the demo's rates (at least 0.5 s)."""
    return max([abs(y - x) / DEMO_RATE.get(j["type"], 30.0) for j, x, y in zip(joints, a, b)] + [0.5])


def closed(keys, joints, hold=0.5):
    """The keyframes with a hold and the motion played backwards appended when they end elsewhere (a loop would
    otherwise jump back): it unfolds the way it folded, at its own pace."""
    if not keys or list(keys[-1][1]) == list(keys[0][1]):
        return keys
    end = keys[-1][0]
    back = [(end + hold + (end - t), list(v)) for t, v in reversed(keys)]
    return keys + back


def lead_in(keys, joints, current):
    """Keyframes moving smoothly from the current pose to the first keyframe (empty when already there)."""
    if not keys or list(current) == list(keys[0][1]):
        return []
    return [(0.0, list(current)), (transition_time(joints, current, keys[0][1]), list(keys[0][1]))]


def sample(keys, t, loop=True):
    """Joint values at time t along keyframes, eased (smoothstep) between them; held at the end without loop."""
    if not keys:
        return []
    end = keys[-1][0]
    if end <= 0:
        return list(keys[0][1])
    t = t % end if loop else min(max(t, 0.0), end)
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            return [x + (y - x) * u for x, y in zip(a, b)]
    return list(keys[-1][1])


# ---------------------------------------------------------------------------------------------- build / pose
def create(project, name, idx, ground_fill, query="", threshold=None):
    """Store a new rig: the object's indices and its fill."""
    d = rig_dir(project, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    np.save(os.path.join(d, "object.npy"), np.asarray(idx, np.int64))
    np.savez(os.path.join(d, "fill.npz"), **ground_fill)
    rig = {"name": name, "query": query, "threshold": threshold, "splat": splat_key(project), "count": int(len(idx)),
           "fill": int(len(ground_fill["means"])), "joints": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    save(project, name, rig)
    np.save(os.path.join(d, "labels.npy"), np.zeros(len(idx), np.int32))
    return rig


def relabel(project, name, means):
    """Recompute the part labels from the rig's joints (means: the object's Gaussian centres)."""
    joints = load(project, name)["joints"]
    parts = None
    if any("part" in j for j in joints):
        from modelforge import rig_partfield
        parts = rig_partfield.load_parts(project, name)
    labels = assign_parts(means, joints, parts=parts)
    np.save(os.path.join(rig_dir(project, name), "labels.npy"), labels)
    return labels


def arrays(project, name):
    """(object indices, part labels, fill dict) of a rig."""
    d = rig_dir(project, name)
    f = np.load(os.path.join(d, "fill.npz"))
    return (np.load(os.path.join(d, "object.npy")), np.load(os.path.join(d, "labels.npy")),
            {k: f[k] for k in f.files})


def _rotate_rows(v, R, t):
    """Structured splat rows moved by x -> R x + t: positions, rotations and the SH bands."""
    out = v.copy()
    X = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64) @ R.T + t
    for i, c in enumerate("xyz"):
        out[c] = X[:, i]
    q = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    qn = splat_to_enu.quat_mul(splat_to_enu.quat_from_rot(R), q)
    for i in range(4):
        out[f"rot_{i}"] = qn[:, i]
    rest = sorted((n for n in v.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest:
        K = len(rest) // 3
        degree = int(round(np.sqrt(K + 1))) - 1
        X_sh = splat_to_enu.sh_rotation(R.T, degree)  # c'(d) = c(R^T d): the colour turns with the part
        F = np.stack([v[n] for n in rest], -1).astype(np.float64).reshape(-1, 3, K) @ X_sh.T
        for i, n in enumerate(rest):
            out[n] = F.reshape(-1, 3 * K)[:, i]
    return out


def fill_rows(dtype, f):
    """The fill as structured splat rows of the splat's dtype (higher SH bands 0)."""
    rows = np.zeros(len(f["means"]), dtype)
    for i, c in enumerate("xyz"):
        rows[c] = f["means"][:, i]
    for i in range(4):
        rows[f"rot_{i}"] = f["quats"][:, i]
    for i in range(3):
        rows[f"scale_{i}"] = f["scales"][:, i]
        rows[f"f_dc_{i}"] = f["sh0"][:, 0, i]
    rows["opacity"] = f["opacities"]
    return rows


def posed_ply(project, name, out, values=None, with_scene=True, with_object=True):
    """Write the splat with the rig's object posed (values per joint; default the stored ones) and its fill; or
    only the posed object (with_scene False), or the scene with the object removed (with_object False). Returns the
    path."""
    import plyfile
    rig = load(project, name)
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data
    idx, labels, f = arrays(project, name)
    Ts = transforms(rig["joints"], values)
    parts = [_rotate_rows(v[idx[labels == k]], T[:3, :3], T[:3, 3]) for k, T in enumerate(Ts)] if with_object else []
    keep = []
    if with_scene:
        rest = np.ones(len(v), bool)
        rest[idx] = False
        keep = [v[rest], fill_rows(v.dtype, f)]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plyfile.PlyData([plyfile.PlyElement.describe(np.concatenate(keep + parts), "vertex")]).write(out)
    return out


def object_mesh(project, name, voxel=0.04, on_line=print, token=None):
    """A mesh of the rig's object alone, in its rest pose (for the rigging networks): the splat-mesh TSDF
    (modelforge.splat_mesh) over a workspace rig/<name>/mesh/ whose splat is only the object, rendered from the
    project's cameras (pixels the object does not cover are not fused). Returns the mesh path."""
    from modelforge import splat_mesh as sm
    work = os.path.join(rig_dir(project, name), "mesh")
    os.makedirs(os.path.join(work, "splat"), exist_ok=True)
    rest = [0.0] * len(load(project, name)["joints"])
    posed_ply(project, name, os.path.join(work, "splat", "splat_enu.ply"), values=rest, with_scene=False)
    link = os.path.join(work, "transforms.json")
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(os.path.join(project, "transforms.json"), work), link)
    sm.run(work, voxel=voxel, on_line=on_line, job=token)
    return os.path.join(work, sm.OUT, "mesh.ply")

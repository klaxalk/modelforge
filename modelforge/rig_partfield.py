"""A rig's parts and joints from PartField (nv-tlabs/PartField, ICCV 2025; the user asked, 2026-10-27).

**Licence: NVIDIA License, non-commercial use only** (research and education). It is installed apart from the image
(`scripts/install_partfield` -> build/partfield/) and only this optional button uses it.

**Features** (`features`): PartField's feed-forward network on the rig's object as a point cloud (its Gaussians'
centres, in rest pose; partfield_inference.py `is_pc True`, ~5 s): a 448-d feature per Gaussian, stored as
rig/<name>/partfield/features.npy (float16). **Parts** (`cluster`, `split`): KMeans (k-means++, a fixed seed, the
best of 3) on the L2-normalized features, as PartField's own clustering does, in-process (no rerun): any number of
parts at once, or one part split further (an excavator's arm into boom and stick). rig/<name>/partfield/parts.npy.

**Joints** (`joints_from_parts`): parts touching (26-neighbouring 0.3 m voxels) form a graph; the largest part is the
base and the others hang on it along the strongest contacts (a contact counts when it reaches 2 % of the smaller
part's voxels; parts that only touch weakly stay with the base). Low parts at the object's floor (top < floor +
DEBRIS_TOP, height < DEBRIS_HEIGHT: gravel left on the cut) stay with the base. Per child part the pin is the centre
of its Gaussians next to the parent; a flat, horizontal contact (normal within ~37° of vertical) covering > 30 % of
the part's footprint is a turntable (vertical axis: an excavator's house on its tracks), anything else a hinge
across the part's horizontal direction (a lift). The joint records its part (`part`), and rig.assign_parts takes
the part's Gaussians instead of a plane cut. kladno_bagr's excavator at 4 parts: undercarriage -> house (slew) ->
arm (hinge), the gravel to the base.
"""

import os
import shlex
import shutil

import numpy as np

from modelforge import jobs, rig

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.environ.get("MODELFORGE_PARTFIELD", os.path.join(_REPO, "build", "partfield"))
VOXEL = 0.3
DEBRIS_TOP = 1.5  # m above the object's floor
DEBRIS_HEIGHT = 1.5  # m


def available():
    if not os.path.exists(os.path.join(HOME, "venv", "bin", "python")):
        return f"PartField is not installed (scripts/install_partfield; looked in {HOME})"
    if not os.path.exists(os.path.join(HOME, "PartField", "model", "model_objaverse.ckpt")):
        return f"PartField's model is missing in {HOME}/PartField/model"
    return None


def work_dir(project, name):
    return os.path.join(rig.rig_dir(project, name), "partfield")


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env.update(PATH=os.path.join(HOME, "venv", "bin") + os.pathsep + env.get("PATH", ""),
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")  # its checkpoint pickles its config
    return env


def object_rows(project, name):
    """The rig's object as structured splat rows in object order (rest pose) and its centres."""
    import plyfile

    from modelforge import splat_mesh
    idx = rig.arrays(project, name)[0]
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data[idx]
    return v, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def features(project, name, on_line=print, token=None):
    """Run PartField on the rig's object (module doc); returns the features (N, 448) float32."""
    import plyfile
    why = available()
    if why:
        raise RuntimeError(why)
    token = token or jobs.Token()
    work = work_dir(project, name)
    if os.path.exists(work):
        shutil.rmtree(work)
    data = os.path.join(work, "data")
    os.makedirs(data)
    v, _ = object_rows(project, name)
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(os.path.join(data, "object.ply"))
    repo = os.path.join(HOME, "PartField")
    tag = "mf_" + os.urandom(4).hex()
    on_line("partfield: features of the object")
    log = []
    try:
        token.run(["bash", "-c", f"cd {shlex.quote(repo)} && python partfield_inference.py -c configs/final/demo.yaml "
                   f"--opts continue_ckpt model/model_objaverse.ckpt result_name {tag} "
                   f"dataset.data_path {shlex.quote(os.path.abspath(data))} is_pc True"], log.append, env=_env(),
                  group=True)
    finally:
        open(os.path.join(work, "partfield.log"), "w").write("\n".join(log))
    out = os.path.join(repo, "exp_results", tag)
    try:
        F = np.load(os.path.join(out, "part_feat_object_0.npy")).astype(np.float32)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    if len(F) != len(v):
        raise RuntimeError(f"PartField returned {len(F)} features for {len(v)} Gaussians")
    np.save(os.path.join(work, "features.npy"), F.astype(np.float16))
    return F


def load_features(project, name):
    p = os.path.join(work_dir(project, name), "features.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def kmeans(X, k, seed=0, iterations=50, restarts=3):
    """Labels of KMeans (k-means++ starts, the best of `restarts` by inertia) on rows of X."""
    best, best_cost = None, np.inf
    xx = (X * X).sum(1)
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        C = [X[rng.integers(len(X))]]
        d2 = np.full(len(X), np.inf)
        for _ in range(1, k):  # k-means++
            d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(C)
        lab = None
        for _ in range(iterations):
            dist = xx[:, None] - 2 * X @ C.T + (C * C).sum(1)[None]
            new = dist.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = X[m].mean(0)
        cost = float(np.take_along_axis(xx[:, None] - 2 * X @ C.T + (C * C).sum(1)[None], lab[:, None], 1).sum())
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


def _ordered(lab):
    """Labels renumbered by part size, largest first."""
    ids, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    rank = np.empty(len(ids), int)
    rank[np.argsort(-cnt, kind="stable")] = np.arange(len(ids))
    return rank[inv.reshape(-1)]


def cluster(project, name, k):
    """Split the object into k parts (module doc); stored and returned (part id per object Gaussian)."""
    F = load_features(project, name)
    if F is None:
        raise RuntimeError("no PartField features yet")
    X = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    parts = _ordered(kmeans(X, int(k)))
    np.save(os.path.join(work_dir(project, name), "parts.npy"), parts)
    return parts


def load_parts(project, name):
    p = os.path.join(work_dir(project, name), "parts.npy")
    return np.load(p) if os.path.exists(p) else None


def split(project, name, part, k=2):
    """Split one part further into k (its Gaussians' features only); stored and returned."""
    F, parts = load_features(project, name), load_parts(project, name)
    if F is None or parts is None:
        raise RuntimeError("propose the parts first")
    m = parts == part
    if m.sum() < 2 * k:
        raise RuntimeError("that part is too small to split")
    X = F[m] / np.maximum(np.linalg.norm(F[m], axis=1, keepdims=True), 1e-9)
    sub = kmeans(X, int(k))
    out = parts.copy()
    nxt = int(parts.max()) + 1
    for j in range(1, int(k)):
        out[np.flatnonzero(m)[sub == j]] = nxt + j - 1
    out = _ordered(out)
    np.save(os.path.join(work_dir(project, name), "parts.npy"), out)
    return out


def _area(xy):
    import cv2
    if len(xy) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.ascontiguousarray(xy, np.float32))))


def joints_from_parts(means, parts, voxel=VOXEL):
    """Joints (rig.new_joint dicts with "part") from the parts (module doc), base first."""
    n = int(parts.max()) + 1
    vk = np.floor(means / voxel).astype(np.int64)
    keys, inv = np.unique(vk, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    # which parts each voxel holds, as a (voxel, part) incidence; touching = same or neighbouring voxel
    pairs = np.unique(np.c_[inv, parts], axis=0)
    hold = np.zeros((len(keys), n), bool)
    hold[pairs[:, 0], pairs[:, 1]] = True
    a, b = rig._neighbours(keys)
    C = hold[a].astype(np.int64).T @ hold[b].astype(np.int64)
    C = C + C.T + hold.astype(np.int64).T @ hold.astype(np.int64)
    np.fill_diagonal(C, 0)
    nvox = hold.sum(0)
    size = np.bincount(parts, minlength=n)
    floor = float(np.percentile(means[:, 2], 2))
    debris = set()
    for q in range(n):
        z = means[parts == q, 2]
        if len(z) and np.percentile(z, 95) < floor + DEBRIS_TOP and np.ptp(np.percentile(z, [5, 95])) < DEBRIS_HEIGHT:
            debris.add(q)
    root = int(np.argmax(size))
    debris.discard(root)
    strong = C >= 0.02 * np.minimum.outer(nvox, nvox)
    parent, order = {root: -1}, [root]
    while True:  # hang each part on the attached part it touches most (strongest contact first)
        cand = [(C[p, q], p, q) for p in parent for q in range(n)
                if q not in parent and q not in debris and strong[p, q]]
        if not cand:
            break
        _, p, q = max(cand)
        parent[q] = p
        order.append(q)
    joints, joint_of = [], {root: 0}
    pkeys = {p: set(map(tuple, np.unique(vk[parts == p], axis=0))) for p in parent}
    offs = [tuple(o) for o in rig._OFFSETS]
    for q in order[1:]:
        p = parent[q]
        Pq = means[parts == q]
        Kq = vk[parts == q]
        near = np.array([any((x + o[0], y + o[1], z + o[2]) in pkeys[p] for o in offs) for x, y, z in map(tuple, Kq)])
        cp = Pq[near] if near.sum() >= 10 else Pq[np.argsort(np.linalg.norm(Pq - means[parts == p].mean(0), axis=1))[:50]]
        o = cp.mean(0)
        w, V = np.linalg.eigh(np.cov((cp - o).T)) if len(cp) > 3 else (None, np.eye(3))
        normal = V[:, 0]
        cover = _area(cp[:, :2]) / max(_area(Pq[:, :2]), 1e-6)
        if abs(normal[2]) > 0.8 and cover > 0.3:
            axis, kind = np.array([0.0, 0.0, 1.0]), "slew"
        else:
            d = Pq.mean(0) - o
            d[2] = 0.0
            if np.linalg.norm(d) < 1e-6:
                d = np.array([1.0, 0.0, 0.0])
            axis, kind = np.cross(d / np.linalg.norm(d), [0.0, 0.0, 1.0]), "hinge"
        seed = Pq[np.argmax(np.linalg.norm(Pq - o, axis=1))]
        j = rig.new_joint(o, o + axis, seed, parent=joint_of[p], name=f"{kind} {len(joints) + 1}",
                          limits=[-180.0, 180.0] if kind == "slew" else [-60.0, 60.0])
        j["part"] = int(q)
        if kind == "slew":
            j["slew"] = True
        joints.append(j)
        joint_of[q] = len(joints)
    return joints


def propose(project, name, k=4, on_line=print, token=None):
    """Features (if missing), k parts and their joints: (joints, parts)."""
    if load_features(project, name) is None:
        features(project, name, on_line=on_line, token=token)
    parts = cluster(project, name, k)
    _, means = object_rows(project, name)
    joints = joints_from_parts(means, parts)
    on_line(f"partfield: {k} parts, {len(joints)} joint(s)")
    return joints, parts

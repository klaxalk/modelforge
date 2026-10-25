"""Manual tie points (the user asked, 2026-10-25: human in the loop SfM; docs/todo_tie_points.md).

Where automatic matching leaves flights apart (other light, season, altitude), the user clicks the same physical spot
in two or more photos of each flight / pass (gui/tiepoints.py). Per tie point, the clicks of each pass (flight =
source, passes = clicks > GROUP_GAP_S apart in time) are triangulated on their own; the difference between two
passes' points is how far apart those passes are there (split_pairs). That goes where the refine moves whole flights
and passes: the per-flight offset (robust.update_source_offsets; tie points there override the SIFT split points),
the drift over time (update_drift) and, in a chunked refine, the loop closures of the pose graph, as SPLIT_COPIES
split matches.

Not as extra BA residuals: tried first, they were satisfied by turning the few cameras clicked in (dragging their own
SIFT points along), cheaper for the BA than moving a flight against its priors; the flight stayed off and the offset
then measured ~0 (synthetic two flights 0.6 m apart: 0.39 m left).

Stored per project in PROJECT/tie_points.json: {"points": [{"name", "obs": [{"image", "x", "y"}]}]}, x / y in the
photo's own pixels, COLMAP's convention (continuous, origin at the top-left corner of the top-left pixel).
"""
import contextlib
import json
import os

import numpy as np

FILE = "tie_points.json"

_ACTIVE = []  # the tie points of the refine running in this process (`active`; robust._split_pairs reads them)


def path(project):
    return os.path.join(project, FILE)


def load(project):
    try:
        pts = json.load(open(path(project))).get("points", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [p for p in pts if isinstance(p, dict) and "name" in p and isinstance(p.get("obs"), list)]


def save(project, points):
    tmp = path(project) + ".part"
    json.dump({"points": points}, open(tmp, "w"), indent=1)
    os.replace(tmp, path(project))


def new_name(points):
    names = {p["name"] for p in points}
    k = 1
    while f"T{k}" in names:
        k += 1
    return f"T{k}"


def set_obs(project, name, image, x, y):
    """Add (or move) the click of tie point `name` in `image`; creates the point when new. Returns the points."""
    pts = load(project)
    p = next((p for p in pts if p["name"] == name), None)
    if p is None:
        p = {"name": name, "obs": []}
        pts.append(p)
    p["obs"] = [o for o in p["obs"] if o["image"] != image] + [{"image": image, "x": float(x), "y": float(y)}]
    save(project, pts)
    return pts


def remove(project, name, image=None):
    """Remove a tie point, or only its click in `image`."""
    pts = load(project)
    if image is None:
        pts = [p for p in pts if p["name"] != name]
    else:
        for p in pts:
            if p["name"] == name:
                p["obs"] = [o for o in p["obs"] if o["image"] != image]
    save(project, pts)
    return pts


@contextlib.contextmanager
def active(project):
    """The project's tie points take part in the refine within (this process): robust._split_pairs yields them."""
    _ACTIVE[:] = load(project)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE.clear()


def _ray(rec, image_id, x, y):
    """(camera centre, unit direction) in the world of a pixel of a registered image."""
    img = rec.image(image_id)
    cam = rec.cameras[img.camera_id]
    n = np.asarray(cam.cam_from_img(np.array([x, y], float)), float).ravel()
    d_cam = np.r_[n[:2], 1.0]
    cfw = img.cam_from_world().matrix()
    R, t = cfw[:, :3], cfw[:, 3]
    d = R.T @ d_cam
    return -R.T @ t, d / np.linalg.norm(d)


def triangulate(rec, obs):
    """3D point closest to the rays of obs [(image_id, x, y)] (least squares), None when the rays are near
    parallel (< 1°) or it lies behind a camera."""
    rays = [_ray(rec, i, x, y) for i, x, y in obs]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    ds = np.array([d for _, d in rays])
    if len(rays) < 2 or np.max(np.degrees(np.arccos(np.clip(ds @ ds.T, -1, 1)))) < 1.0:
        return None
    X = np.linalg.solve(A, b)
    if any((X - c) @ d <= 0 for c, d in rays):
        return None
    return X


def project(rec, image_id, X):
    """Pixel (x, y) of world point X in a registered image, None behind it."""
    img = rec.image(image_id)
    cfw = img.cam_from_world().matrix()
    p = cfw[:, :3] @ X + cfw[:, 3]
    if p[2] <= 0:
        return None
    return np.asarray(rec.cameras[img.camera_id].img_from_cam(p), float).ravel()[:2]


def resolve(rec, points=None):
    """[(name, [(image_id, x, y)])] of the tie points (default: the active ones) with >= 2 clicks in registered
    images."""
    by_name = {rec.image(i).name: i for i in rec.reg_image_ids()}
    out = []
    for p in (_ACTIVE if points is None else points):
        obs = [(by_name[o["image"]], o["x"], o["y"]) for o in p["obs"] if o["image"] in by_name]
        if len(obs) >= 2:
            out.append((p["name"], obs))
    return out


def errors(rec, points=None):
    """{name: {"clicks", "error_px" (median), "max_px", "xyz"}} of the tie points in rec: triangulated from all
    their clicks with rec's cameras (refined: how well the passes agree there)."""
    out = {}
    for name, obs in resolve(rec, points):
        X = triangulate(rec, obs)
        if X is None:
            out[name] = {"clicks": len(obs), "error_px": None}
            continue
        e = []
        for i, x, y in obs:
            q = project(rec, i, X)
            e.append(float(np.hypot(q[0] - x, q[1] - y)) if q is not None else np.inf)
        out[name] = {"clicks": len(obs), "error_px": round(float(np.median(e)), 2), "max_px": round(max(e), 2),
                     "xyz": np.round(X, 3).tolist()}
    return out


MIN_ANGLE_DEG = 5.0  # a pass's clicks count for the offset only when their rays span this much (else poor depth)
SPLIT_COPIES = 400  # a tie point's offset counts as this many split matches (drift / loop-closure weights saturate)
GROUP_GAP_S = 60.0  # clicks of one flight more than this apart in time: different passes (a revisit)


def _span(rec, obs):
    """The largest angle (°) between the rays of clicks [(image_id, x, y)]."""
    d = np.array([_ray(rec, i, x, y)[1] for i, x, y in obs])
    return float(np.degrees(np.arccos(np.clip(d @ d.T, -1, 1))).max())


def split_pairs(rec, priors, same_source, gap_s=GROUP_GAP_S, points=None):
    """Offsets between passes measured by the tie points: per tie point its clicks grouped by flight (source) and
    pass (gap_s apart in time), each group with >= 2 clicks triangulated on its own; for every two groups (same
    flight: same_source True, else different flights) yields (an image of group a, one of group b, X_a - X_b). As
    the split points of a SIFT match that triangulated as two copies, but chosen by the user: the per-flight offset
    and drift estimates (robust.py) and the chunked refine's loop closures move whole flights / passes with it
    (as BA terms alone, tie points only turned the few cameras they were clicked in)."""
    for name, obs in resolve(rec, points):
        groups = {}
        for i, x, y in sorted(obs, key=lambda o: getattr(priors.get(rec.image(o[0]).name), "time", 0.0) or 0.0):
            p = priors.get(rec.image(i).name)
            if p is None:
                continue
            g = groups.setdefault(p.source, [[]])
            t = p.time or 0.0
            if g[-1] and t - (getattr(priors[rec.image(g[-1][-1][0]).name], "time", 0.0) or 0.0) > gap_s:
                g.append([])
            g[-1].append((i, x, y))
        flat = [(s, grp) for s, gs in groups.items() for grp in gs if len(grp) >= 2 and _span(rec, grp) >= MIN_ANGLE_DEG]
        tri = [(s, grp, triangulate(rec, grp)) for s, grp in flat]
        tri = [(s, grp, X) for s, grp, X in tri if X is not None]
        for a in range(len(tri)):
            for b in range(a + 1, len(tri)):
                sa, ga, Xa = tri[a]
                sb, gb, Xb = tri[b]
                if (sa == sb) == same_source:
                    yield rec.image(ga[0][0]), rec.image(gb[0][0]), Xa - Xb


def per_flight(rec, priors, point):
    """For one tie point (a points-file entry): per flight (source) its clicks in registered photos, the angle their
    rays span (°), their own consistency (median reprojection px of the point they alone give) and that point; and the
    distance between the flights' points (how far apart the flights still are there). {"flights": {source: {...}},
    "apart_m"}."""
    res = resolve(rec, [point])
    if not res:
        return {"flights": {}, "apart_m": None}
    out, xs = {}, {}
    for s in sorted({getattr(priors.get(rec.image(i).name), "source", 0) for i, _, _ in res[0][1]}):
        o = [(i, x, y) for i, x, y in res[0][1] if getattr(priors.get(rec.image(i).name), "source", 0) == s]
        d = np.array([_ray(rec, i, x, y)[1] for i, x, y in o])
        ang = float(np.degrees(np.arccos(np.clip(d @ d.T, -1, 1))).max()) if len(o) > 1 else 0.0
        X = triangulate(rec, o) if len(o) >= 2 else None
        err = None
        if X is not None:
            err = float(np.median([np.hypot(*(project(rec, i, X) - [x, y])) for i, x, y in o]))
            xs[s] = X
        out[s] = {"clicks": len(o), "angle_deg": round(ang, 1), "error_px": None if err is None else round(err, 1)}
    apart = None
    if len(xs) >= 2:
        ks = sorted(xs)
        apart = round(float(max(np.linalg.norm(xs[a] - xs[b]) for a in ks for b in ks if a < b)), 3)
    return {"flights": out, "apart_m": apart}

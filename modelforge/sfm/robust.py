"""Prior model and its self-calibration between refinement rounds (M3).

The prior model holds everything about the telemetry that refinement may learn:
  - absolute prior sigmas (world-frame shape, see PriorSigmas) and relative
    (consecutive-frame) sigmas, sigma_pos = a + b * baseline;
  - systematic errors shared by the whole flight: boresight B (camera-frame
    rotation, R_cw_true = Exp(B) R_cw_prior) and a residual time offset dt
    (C_true = C_prior + v * dt);
  - priors rejected as outliers.

After each BA the residuals are computed exactly as the Ceres functors define them
(pycolmap AbsolutePosePriorCost / RelativePosePriorCost), then:
  systematic re-estimate B (median camera-frame rotation residual) and dt (trimmed least
             squares of position residual on velocity), over all frames and against the raw
             priors, then fold them into the priors;
  calibrate  rescale each block (abs rot, abs pos, rel rot, rel pos) so the median
             whitened squared norm matches chi2_3's median; clamped per round and
             floored, because the poses also feel the priors (no spiral to zero);
  flag       priors beyond chi2_6 at `outlier_quantile` lose their term next round;
Multi-video projects also get a position offset per source (update_source_offsets),
estimated after every round's BA, starting with round 0. Within a source, a smooth position
drift over time (update_drift) does the same for revisits of one long flight.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from modelforge.sfm.priors import Prior, PriorSigmas, by_order, rot_to_quat_xyzw, same_source_consecutive  # noqa: F401

CHI2_3_MEDIAN = 2.3660
CHI2_6 = {0.99: 16.812, 0.999: 22.458}


def exp_so3(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def log_so3(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


@dataclass
class RelativeSigmas:
    """sigma_pos = pos_m + pos_per_m * baseline.

    COLMAP's RelativePosePriorCost is rigid in the camera frame: its translation
    residual is R_i [(C_j - C_i) - R_j^T R_pj (C_pj - C_pi)], i.e. the prior
    displacement is rotated by camera j's ABSOLUTE attitude error. Telemetry
    displacement is accurate in the world frame (integrated velocity, ~1 % of the
    baseline), so the camera-frame term also carries ~sigma_abs_rot * baseline:
    pos_per_m ~= 0.008 (velocity) + 0.0175 (1 deg heading).

    sigma_rot = rot_deg + rot_per_deg * turn: 10 Hz attitude interpolation error grows
    with the rotation between the two images (~2 % of the turn on berlin_3_1).
    """
    rot_deg: float = 0.3
    rot_per_deg: float = 0.02
    pos_m: float = 0.03
    pos_per_m: float = 0.025

    def covariance(self, baseline, turn_deg=0.0):
        sp = self.pos_m + self.pos_per_m * baseline
        sr = self.rot_deg + self.rot_per_deg * turn_deg
        return np.diag([np.radians(sr) ** 2] * 3 + [sp ** 2] * 3)


@dataclass
class PriorModel:
    abs_sigmas: PriorSigmas = field(default_factory=PriorSigmas)
    rel_sigmas: RelativeSigmas = field(default_factory=RelativeSigmas)
    use_relative: bool = True
    calibrate: bool = True
    estimate_systematic: bool = True
    estimate_source_offsets: bool = True  # multi-video projects: per-source position offset
    source_offsets: dict = field(default_factory=dict)  # source -> ENU offset [m] added to its priors
    # which ENU axes the per-source offsets may move: dji/flight_check.py only z (its few far-away points measure
    # the height between flights well and the horizontal badly: karlak's east swung -33 / +43 m between rounds)
    source_offset_axes: tuple = (1.0, 1.0, 1.0)
    # within one source: ENU offset [m] linear in time between knots every drift_knot_s, added to the priors
    # (update_drift; long flights drift by metres, so revisits triangulate as offset copies)
    estimate_drift: bool = True
    drift_knot_s: float = 30.0
    drift: dict = field(default_factory=dict)  # source -> (knot times [s], K x 3 offsets [m])
    boresight: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad, camera frame
    # several cameras (multi-camera DJI projects): a boresight per COLMAP camera id, and each image's camera id
    boresight_by_camera: dict = field(default_factory=dict)
    image_camera: dict = field(default_factory=dict)
    dt: float = 0.0  # s
    abs_rejected: set = field(default_factory=set)  # image names
    rel_rejected: set = field(default_factory=set)  # (name_i, name_j)
    outlier_quantile: float = 0.999
    # A camera seeing few points cannot overrule its prior: rejecting it lets the pose drift
    # (berlin_3_1 pieces: 56 observations -> 40 m off). Its large residual means the image
    # solution is weak, not that the prior is wrong.
    min_obs_to_reject: int = 100
    start_round: int = 1  # round 0 runs with unconverged intrinsics and 40 px thresholds: no calibration / drift
    # relative priors from this round on, with this loss (2026-10-25, the user: round 0 broke the smooth flight into a
    # jagged path): before they started in round 1 with Cauchy, whose pull fades far out, so cameras round 0 had
    # moved off (karlak: off a local smooth fit p90 87 cm, the priors 21 cm) stayed there. From round 0 with Huber
    # (constant pull, as the absolute priors): 54 cm, the reprojection error unchanged (0.346 -> 0.349 px)
    relative_start_round: int = 0
    rel_loss: str = "huber"  # "huber" | "cauchy"
    guard_relative: bool = True  # min_obs_to_reject also guards the relative priors (either camera; False = before)
    max_scale_change: float = 2.0  # per round
    floors: tuple = (0.1, 0.1, 0.02, 0.01)  # abs rot deg, abs pos m, rel rot deg, rel pos m

    def summary(self):
        a, r = self.abs_sigmas, self.rel_sigmas
        return {"abs_sigmas": {"pos_horizontal_m": round(a.pos_horizontal_m, 3), "pos_vertical_m": round(a.pos_vertical_m, 3),
                               "roll_pitch_deg": round(a.roll_pitch_deg, 3), "yaw_deg": round(a.yaw_deg, 3)},
                "rel_sigmas": {"rot_deg": round(r.rot_deg, 3), "rot_per_deg": round(r.rot_per_deg, 4), "pos_m": round(r.pos_m, 4), "pos_per_m": round(r.pos_per_m, 5)},
                "boresight_deg": np.round(np.degrees(self.boresight), 3).tolist(),
                "boresight_norm_deg": round(float(np.degrees(np.linalg.norm(self.boresight))), 3),
                **({"boresight_by_camera_deg": {str(k): np.round(np.degrees(v), 3).tolist()
                                                for k, v in sorted(self.boresight_by_camera.items())}}
                   if self.boresight_by_camera else {}),
                "dt_ms": round(self.dt * 1000, 1),
                "source_offsets_m": {str(k): np.round(v, 3).tolist() for k, v in sorted(self.source_offsets.items())},
                "drift_m": {str(k): {"knots": len(t), "max": round(float(np.abs(o).max()), 3) if len(o) else 0.0,
                                     "range_z": round(float(np.ptp(o[:, 2])), 3) if len(o) else 0.0}
                            for k, (t, o) in sorted(self.drift.items())},
                "abs_rejected": sorted(self.abs_rejected), "rel_rejected": len(self.rel_rejected)}


def corrected_priors(priors, model):
    """Priors with the systematic corrections (boresight, time offset, source offsets) applied."""
    B = exp_so3(model.boresight)
    Bc = {c: exp_so3(v) for c, v in model.boresight_by_camera.items()}
    out = {}
    for name, p in priors.items():
        R = Bc.get(model.image_camera.get(name), B) @ p.R_cw
        C = p.center + (p.velocity if p.velocity is not None else 0.0) * model.dt
        C = C + model.source_offsets.get(p.source, 0.0) + drift_at(model, p.source, p.time)
        out[name] = Prior.from_pose(p.name, p.frame, p.pts, R, C, p.velocity, p.source, p.order, p.time)
    return out


MAX_ORDER_GAP = 10


def consecutive_pairs(priors, names):
    """Relative-prior pairs: time-consecutive images of the same source, at most MAX_ORDER_GAP apart in
    capture order (a missing image or two is bridged; the gaps between the blocks of a calibration
    subset are not)."""
    return [(a, b) for a, b in same_source_consecutive(priors, names)
            if priors[b].order - priors[a].order <= MAX_ORDER_GAP]


def relative_prior(pi, pj):
    """i_from_j = T_i * T_j^-1 from two priors, as (R, t)."""
    R = pi.R_cw @ pj.R_cw.T
    t = -pi.R_cw @ pi.center - R @ (-pj.R_cw @ pj.center)
    return R, t


def relative_covariance(model, pi, pj):
    R, _ = relative_prior(pi, pj)
    baseline = float(np.linalg.norm(pj.center - pi.center))
    turn = float(np.degrees(np.linalg.norm(log_so3(R))))
    return model.rel_sigmas.covariance(baseline, turn)


def _pose(rec, image):
    from modelforge.gui.data import xyzw_to_rot  # shared quaternion helper
    p = np.array(rec.frame(image.frame_id).rig_from_world.params)
    R = xyzw_to_rot(p[:4])
    return R, p[4:]


def residuals(rec, priors, model):
    """Absolute and relative prior residuals of the current reconstruction.

    abs[name]  = (r_rot, r_pos, cov 6x6)   r_rot = log(R_param R_prior^T), r_pos = R_param (C_prior - C_param)
    rel[(i,j)] = (r_rot, r_pos, cov 6x6)   log/translation of T_i T_j^-1 (T_i^prior T_j^prior^-1)^-1
    """
    images = {rec.image(i).name: rec.image(i) for i in rec.reg_image_ids()}
    poses = {n: _pose(rec, im) for n, im in images.items() if n in priors}
    abs_res = {}
    for n, (R, t) in poses.items():
        p = priors[n]
        C = -R.T @ t
        abs_res[n] = (log_so3(R @ p.R_cw.T), R @ (p.center - C), p.covariance(model.abs_sigmas))
    rel_res = {}
    for a, b in consecutive_pairs(priors, poses):
        Ri, ti = poses[a]
        Rj, tj = poses[b]
        Rp, tp = relative_prior(priors[a], priors[b])
        R_ij, t_ij = Ri @ Rj.T, ti - Ri @ Rj.T @ tj  # T_i T_j^-1
        R_e = R_ij @ Rp.T  # T_i T_j^-1 (T_ij^prior)^-1
        t_e = t_ij - R_e @ tp
        rel_res[(a, b)] = (log_so3(R_e), t_e, relative_covariance(model, priors[a], priors[b]))
    return abs_res, rel_res


def _d2_blocks(r_rot, r_pos, cov):
    return (float(r_rot @ np.linalg.solve(cov[:3, :3], r_rot)), float(r_pos @ np.linalg.solve(cov[3:, 3:], r_pos)))


def _estimate_systematic(model, rec, priors_raw):
    """Boresight = median camera-frame rotation residual; dt = trimmed least squares of the
    world position residual on velocity. Against the RAW priors and over ALL frames, so a
    large systematic error cannot first get every prior flagged as an outlier."""
    raw_abs, _ = residuals(rec, priors_raw, model)
    names = list(raw_abs)
    if len(names) < 10:
        return
    rot = np.array([raw_abs[n][0] for n in names])
    model.boresight = np.median(rot, axis=0)
    cams = np.array([rec.find_image_with_name(n).camera_id for n in names])
    model.boresight_by_camera, model.image_camera = {}, {}
    if len(set(cams.tolist())) > 1:  # each camera its own mount (several drones / lenses)
        model.image_camera = dict(zip(names, cams.tolist()))
        model.boresight_by_camera = {int(c): (np.median(rot[cams == c], axis=0) if (cams == c).sum() >= 10
                                              else model.boresight) for c in set(cams.tolist())}
    V = np.array([priors_raw[n].velocity for n in names])
    # world-frame position error C_param - C_prior = -R^T r_pos
    D = np.array([-_pose(rec, rec.find_image_with_name(n))[0].T @ raw_abs[n][1] for n in names])
    D -= np.array([model.source_offsets.get(priors_raw[n].source, np.zeros(3))
                   + drift_at(model, priors_raw[n].source, priors_raw[n].time) for n in names])
    keep = np.ones(len(names), bool)
    for _ in range(2):  # fit, drop the worst 20 %, refit
        v, d = V[keep].ravel(), D[keep].ravel()
        dt = float(v @ d / max(v @ v, 1e-9))
        err = np.linalg.norm(D - V * dt, axis=1)
        keep = err <= np.percentile(err, 80)
    model.dt = dt


def update(model, rec, priors_raw):
    """One self-calibration step after a BA: systematic -> calibrate -> flag.
    Returns a report dict; mutates model."""
    rep = {}
    if model.estimate_systematic:
        _estimate_systematic(model, rec, priors_raw)
    priors = corrected_priors(priors_raw, model)
    abs_res, rel_res = residuals(rec, priors, model)

    # --- calibrate sigmas: median whitened squared norm -> chi2_3 median, per block.
    # The median ignores the outliers flagged below, so no exclusion is needed here.
    if model.calibrate:
        def scale(d2s):
            if not d2s:
                return 1.0
            s = np.sqrt(np.median(d2s) / CHI2_3_MEDIAN)
            return float(np.clip(s, 1 / model.max_scale_change, model.max_scale_change))
        ab = [_d2_blocks(*r) for r in abs_res.values()]
        rb = [_d2_blocks(*r) for r in rel_res.values()]
        s_ar, s_ap = scale([x[0] for x in ab]), scale([x[1] for x in ab])
        s_rr, s_rp = scale([x[0] for x in rb]), scale([x[1] for x in rb])
        a, r, f = model.abs_sigmas, model.rel_sigmas, model.floors
        a.roll_pitch_deg, a.yaw_deg = max(f[0], a.roll_pitch_deg * s_ar), max(f[0], a.yaw_deg * s_ar)
        a.pos_horizontal_m, a.pos_vertical_m = max(f[1], a.pos_horizontal_m * s_ap), max(f[1], a.pos_vertical_m * s_ap)
        if model.use_relative:
            r.rot_deg, r.rot_per_deg = max(f[2], r.rot_deg * s_rr), r.rot_per_deg * s_rr
            r.pos_m, r.pos_per_m = max(f[3], r.pos_m * s_rp), r.pos_per_m * s_rp
        rep["scales"] = {"abs_rot": round(s_ar, 3), "abs_pos": round(s_ap, 3),
                         "rel_rot": round(s_rr, 3), "rel_pos": round(s_rp, 3)}
        abs_res, rel_res = residuals(rec, priors, model)  # re-whiten with the new sigmas

    # --- flag outlier priors (full 6-DoF Mahalanobis against the corrected priors)
    thr = CHI2_6[model.outlier_quantile]
    d2_abs = {n: sum(_d2_blocks(*r)) for n, r in abs_res.items()}
    d2_rel = {k: sum(_d2_blocks(*r)) for k, r in rel_res.items()}
    nobs = {rec.image(i).name: rec.image(i).num_points3D for i in rec.reg_image_ids()}
    model.abs_rejected = {n for n, d in d2_abs.items() if d > thr and nobs.get(n, 0) >= model.min_obs_to_reject}
    # Relative: the same guard, on either camera. A weakly seen camera that jumped looks like an outlier against its
    # neighbours, and losing its smoothness term too leaves the jump in place. karlak (2026-10-26): rejected after
    # rounds 0 / 1 289 / 369 -> 208 / 311, final path off a local smooth fit p90 / p99 29 / 134 -> 23 / 129 cm (the
    # priors 21 / 124); kladno_dopravnik (150 photos, one clean flight): identical to before.
    model.rel_rejected = ({k for k, d in d2_rel.items()
                           if d > thr and (not model.guard_relative
                                            or min(nobs.get(k[0], 0), nobs.get(k[1], 0)) >= model.min_obs_to_reject)}
                          if model.use_relative else set())
    rep["abs_d2_median"] = round(float(np.median(list(d2_abs.values()))), 3)
    rep["rel_d2_median"] = round(float(np.median(list(d2_rel.values()))), 3) if d2_rel else None
    rep["abs_rejected"] = len(model.abs_rejected)
    rep["rel_rejected"] = len(model.rel_rejected)
    rep["model"] = model.summary()
    return rep


# --- multi-video: per-source position offsets ----------------------------------------
# Each flight has its own GNSS bias (~1-3 m) and possibly takeoff-height error, so the
# videos' priors disagree by a near-constant shift. Before the shift is known, each video
# triangulates its own copy of the shared scene ("split" points), and the median
# difference between the copies measures the shift directly.

def _point_ids(image):
    return np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D], np.int64)


def _split_pairs(rec, priors, db_path, same_source):
    """Verified image pairs (same_source: within a source, else across sources) as (image a, image b,
    X_a - X_b for every inlier match whose keypoints have different 3D points, merged count)."""
    import pycolmap
    reg = {i: rec.image(i) for i in rec.reg_image_ids()}
    src = {i: priors[im.name].source for i, im in reg.items() if im.name in priors}
    ids = {}
    pids = np.fromiter(rec.points3D.keys(), np.int64, rec.num_points3D())
    xyz = np.array([rec.points3D[int(k)].xyz for k in pids]).reshape(-1, 3)
    sort = np.argsort(pids)
    db = pycolmap.Database.open(db_path)
    pair_ids, geoms = db.read_two_view_geometries()
    db.close()
    for pid, g in zip(pair_ids, geoms):
        i1, i2 = pycolmap.pair_id_to_image_pair(pid)
        if i1 not in src or i2 not in src or (src[i1] == src[i2]) != same_source:
            continue
        m = np.asarray(g.inlier_matches)
        if not len(m):
            continue
        for i in (i1, i2):
            if i not in ids:
                ids[i] = _point_ids(reg[i])
        p1, p2 = ids[i1][m[:, 0]], ids[i2][m[:, 1]]
        both = (p1 >= 0) & (p2 >= 0)
        split = both & (p1 != p2)
        x1 = xyz[sort[np.searchsorted(pids, p1[split], sorter=sort)]] if split.any() else np.zeros((0, 3))
        x2 = xyz[sort[np.searchsorted(pids, p2[split], sorter=sort)]] if split.any() else np.zeros((0, 3))
        yield reg[i1], reg[i2], x1 - x2, int(np.sum(both & (p1 == p2)))
    from modelforge.sfm import tie_points  # the user's tie points: as split matches between their passes
    if tie_points._ACTIVE:
        for a, b, d in tie_points.split_pairs(rec, priors, same_source):
            yield a, b, np.tile(d, (tie_points.SPLIT_COPIES, 1)), 0


def split_point_edges(rec, priors, db_path):
    """Per source pair (a, b), a < b: X_a - X_b over verified cross-source matches whose
    keypoints have different 3D points (n_split), and the count already merged."""
    edges = {}
    for im1, im2, d, merged in _split_pairs(rec, priors, db_path, same_source=False):
        s1, s2 = priors[im1.name].source, priors[im2.name].source
        if s1 > s2:
            s1, s2, d = s2, s1, -d
        e = edges.setdefault((s1, s2), {"diffs": [], "merged": 0})
        e["merged"] += merged
        e["diffs"] += list(d)
    out = {k: {"n_split": len(v["diffs"]), "merged": v["merged"],
               "median": np.median(np.array(v["diffs"]), axis=0) if v["diffs"] else np.zeros(3)}
           for k, v in edges.items()}
    from modelforge.sfm import tie_points  # where the user tied two flights together, their clicks decide
    ties = {}
    if tie_points._ACTIVE:
        for im1, im2, d in tie_points.split_pairs(rec, priors, same_source=False):
            s1, s2 = priors[im1.name].source, priors[im2.name].source
            ties.setdefault((min(s1, s2), max(s1, s2)), []).append(d if s1 < s2 else -d)
    for k, ds in ties.items():
        e = out.setdefault(k, {"n_split": 0, "merged": 0})
        e["median"], e["tie_points"] = np.median(np.array(ds), axis=0), len(ds)
    return out


def _solve_offsets(edges, ref):
    """Least squares o_b - o_a = median(X_a - X_b) on the source graph, o_ref = 0;
    sources not connected to ref get no increment."""
    connected, grow = {ref}, True
    while grow:
        grow = False
        for a, b in edges:
            if (a in connected) != (b in connected):
                connected |= {a, b}
                grow = True
    unknown = sorted(connected - {ref})
    if not unknown:
        return {}
    col = {s: k for k, s in enumerate(unknown)}
    rows, rhs = [], []
    for (a, b), e in edges.items():
        if a not in connected or b not in connected:
            continue
        w = np.sqrt(e["n_split"])
        row = np.zeros(len(unknown))
        if b in col:
            row[col[b]] += w
        if a in col:
            row[col[a]] -= w
        rows.append(row)
        rhs.append(w * e["median"])
    x = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return {s: x[col[s]] for s in unknown}


def _shift_cameras(rec, priors, inc):
    """Move the cameras of each source by its offset increment (C += d)."""
    _shift_images(rec, {n: inc[p.source] for n, p in priors.items() if p.source in inc})


def update_source_offsets(model, rec, priors_raw, db_path, min_split=30):
    """One per-source offset update after a BA (no-op for single-source projects).
    Reference = source of the first image. The increment comes from split points only:
    a direct geometric measurement of how far apart the sources' copies of the scene are.
    The cameras are shifted with it, so the next triangulation merges the copies. Once
    merged, the split points left over measure ~0 and the offsets settle.

    (Per-source median prior residuals were tried as a second estimator and rejected: on
    two pieces of one flight they reported a ~1 m offset that wandered by 0.5 m per round,
    i.e. GNSS drift between the pieces, while the split points measured 6 mm.)"""
    sources = {p.source for p in priors_raw.values()}
    if len(sources) < 2 or not model.estimate_source_offsets:
        return None
    ref = priors_raw[by_order(priors_raw)[0]].source
    edges = split_point_edges(rec, priors_raw, db_path)
    use = {k: e for k, e in edges.items() if e["n_split"] >= min_split or e.get("tie_points")}
    inc = _solve_offsets(use, ref) if use else {}
    inc = {s: np.asarray(d, float) * np.asarray(model.source_offset_axes, float) for s, d in inc.items()}
    _shift_cameras(rec, priors_raw, inc)
    model._offset_inc = {s: np.asarray(d, float) for s, d in inc.items()}  # update_drift: already applied
    for s, d in inc.items():
        model.source_offsets[s] = model.source_offsets.get(s, np.zeros(3)) + d
    return {"method": "split_points" if use else "none", "reference": int(ref),
            "increment_m": {str(s): np.round(d, 3).tolist() for s, d in sorted(inc.items())},
            "offsets_m": {str(s): np.round(d, 3).tolist() for s, d in sorted(model.source_offsets.items())},
            "edges": {f"{a}-{b}": {"n_split": e["n_split"], "merged": e["merged"],
                                   "median_m": np.round(e["median"], 3).tolist()} for (a, b), e in edges.items()}}


# --- one long flight: smooth position drift over time ----------------------------------
# GNSS / barometric altitude drift over a long flight (kladno_dira, 30 min: 1-3.4 m between passes over the
# same ground). Each time segment follows its own prior, so a revisit triangulates as a new, offset copy of the
# scene that COLMAP cannot merge (3 m ~ 200 px at 50 m), and the reprojection error does not show it. The
# split points between images of one source measure the drift difference between their capture times.

DRIFT_MIN_SPLIT = 20  # split matches an image pair needs to count
DRIFT_HUBER_M = 0.5  # robust scale of a pair's median difference
DRIFT_SMOOTH = 1.0  # weight of the differences between neighbouring knots
DRIFT_RIDGE = 0.05  # weight pulling each knot's increment to 0 (knots no pair constrains)


def drift_at(model, source, t):
    """Drift offset [m] of a source at time(s) t (linear between knots, constant beyond)."""
    d = model.drift.get(source)
    if d is None or t is None:
        return np.zeros(3) if np.ndim(t) == 0 else np.zeros((len(t), 3))
    return _interp(*d, t)


def _interp(knots, off, t):
    return np.stack([np.interp(t, knots, off[:, k]) for k in range(3)], -1)


def _hat(t, knots):
    """Linear-interpolation weights (n x K) of times t on knots."""
    eye = np.eye(len(knots))
    return np.stack([np.interp(t, knots, eye[k]) for k in range(len(knots))], -1)


def _solve_drift(ti, tj, diff, n, knots, times):
    """Increments (K x 3): o(t_j) - o(t_i) = diff per pair (weight sqrt(n), Huber), neighbouring knots alike
    (first differences: a stretch without revisits stays flat; second differences extrapolated noisy slopes
    across the gaps of a calibration subset into metres), a small ridge, and the mean over the source's
    images = 0 (the priors keep the absolute position)."""
    K = len(knots)
    A = _hat(tj, knots) - _hat(ti, knots)
    w0 = np.sqrt(np.minimum(n, 400.0))
    rows = [np.eye(K) * DRIFT_RIDGE, _hat(times, knots).mean(0, keepdims=True) * 1e3]
    D1 = np.zeros((K - 1, K))
    for k in range(K - 1):
        D1[k, k:k + 2] = (1.0, -1.0)
    rows.append(D1 * DRIFT_SMOOTH * np.sqrt(len(ti) / K))
    fixed = np.vstack(rows)
    w = w0.copy()
    for _ in range(5):  # IRLS (Huber on the pair residual norm)
        M = np.vstack([A * w[:, None], fixed])
        b = np.vstack([diff * w[:, None], np.zeros((len(fixed), 3))])
        x = np.linalg.lstsq(M, b, rcond=None)[0]
        r = np.linalg.norm(A @ x - diff, axis=1)
        w = w0 * np.sqrt(np.minimum(1.0, DRIFT_HUBER_M / np.maximum(r, 1e-9)))
    return x


MIN_CROSS = 5  # cross-source pairs a source needs before they, not its priors' mean, place its drift
# revisits across sources (flights) in the drift, one solve over all sources. berlin (4 flights, global refine,
# 2026-10-24): merged matches between flights 48-80 % -> 87-93 %, split medians 0.17-0.85 -> 0.04-0.30 m, error
# 0.478 -> 0.496 px, 89 -> 43 min; only flights 3_2-3_4 (~600 matches) stay apart (4.1 -> 3.4 m)
CROSS_SOURCE_DRIFT = True


def _solve_drift_joint(blocks, eqs, anchored):
    """All sources' drift increments in one least squares. blocks: {source: (knots, image times)}; eqs: (s_i, t_i,
    s_j, t_j, diff, n) with o_sj(t_j) - o_si(t_i) = diff (within a source and across sources); anchored: the sources
    whose mean over their images stays 0 (the reference, and sources too few cross pairs place). Otherwise as
    _solve_drift (first differences, ridge, Huber). Returns {source: increments (K x 3)}."""
    order = sorted(blocks)
    base, tot = {}, 0
    for s in order:
        base[s] = tot
        tot += len(blocks[s][0])

    def hat(s, t):
        h = np.zeros((len(t), tot))
        h[:, base[s]:base[s] + len(blocks[s][0])] = _hat(np.asarray(t, float), blocks[s][0])
        return h
    A = np.vstack([hat(sj, [tj]) - hat(si, [ti]) for si, ti, sj, tj, _, _ in eqs])
    diff = np.array([e[4] for e in eqs])
    w0 = np.sqrt(np.minimum(np.array([e[5] for e in eqs], float), 400.0))
    rows = [np.eye(tot) * DRIFT_RIDGE]
    for s in order:
        knots, times = blocks[s]
        K, o = len(knots), base[s]
        if s in anchored:
            m = np.zeros((1, tot))
            m[0, o:o + K] = _hat(np.asarray(times, float), knots).mean(0) * 1e3
            rows.append(m)
        n_s = sum(1 for e in eqs if s in (e[0], e[2]))
        D1 = np.zeros((K - 1, tot))
        for k in range(K - 1):
            D1[k, o + k], D1[k, o + k + 1] = 1.0, -1.0
        rows.append(D1 * DRIFT_SMOOTH * np.sqrt(max(n_s, 1) / K))
    fixed = np.vstack(rows)
    w = w0.copy()
    for _ in range(5):  # IRLS (Huber on the pair residual norm)
        M = np.vstack([A * w[:, None], fixed])
        b = np.vstack([diff * w[:, None], np.zeros((len(fixed), 3))])
        x = np.linalg.lstsq(M, b, rcond=None)[0]
        r = np.linalg.norm(A @ x - diff, axis=1)
        w = w0 * np.sqrt(np.minimum(1.0, DRIFT_HUBER_M / np.maximum(r, 1e-9)))
    return {s: x[base[s]:base[s] + len(blocks[s][0])] for s in order}


def update_drift(model, rec, priors_raw, db_path, min_split=DRIFT_MIN_SPLIT):
    """One drift update after a BA (or a triangulation): per source, the increments of a position offset
    linear in time between knots every model.drift_knot_s, from the split points between its images (as
    update_source_offsets does across sources); the cameras are shifted with them, so the next
    triangulation merges the copies. A source without revisits measures ~0 and keeps 0 (the ridge)."""
    if not model.estimate_drift:
        return None
    reg_names = {rec.image(i).name for i in rec.reg_image_ids()}
    by_src = {}
    for n, p in priors_raw.items():
        if n in reg_names and p.time is not None:
            by_src.setdefault(p.source, []).append(p.time)
    pairs = {}
    for im1, im2, d, merged in _split_pairs(rec, priors_raw, db_path, same_source=True):
        if len(d) < min_split:
            continue
        p1, p2 = priors_raw[im1.name], priors_raw[im2.name]
        pairs.setdefault(p1.source, []).append((p1.time, p2.time, np.median(d, axis=0), len(d), merged))
    offset_inc = getattr(model, "_offset_inc", None) or {}  # this round's source offsets moved those cameras already
    model._offset_inc = {}
    cross = []
    if len(by_src) > 1 and CROSS_SOURCE_DRIFT:  # revisits across sources (flights) tie the sources' drifts together
        for im1, im2, d, merged in _split_pairs(rec, priors_raw, db_path, same_source=False):
            if len(d) < min_split:
                continue
            p1, p2 = priors_raw[im1.name], priors_raw[im2.name]
            if p1.source not in by_src or p2.source not in by_src:
                continue
            dd = np.median(d, axis=0) - (offset_inc.get(p2.source, 0.0) - offset_inc.get(p1.source, 0.0))
            cross.append((p1.source, p1.time, p2.source, p2.time, dd, len(d), merged))
    if cross:
        return _update_drift_joint(model, rec, priors_raw, by_src, pairs, cross)
    inc_by_image, rep = {}, {}
    for s, times in by_src.items():
        times = np.asarray(times, float)
        t0, t1 = float(times.min()), float(times.max())
        if s not in model.drift:
            K = max(2, int(np.ceil((t1 - t0) / model.drift_knot_s)) + 1)
            model.drift[s] = (np.linspace(t0, t1, K), np.zeros((K, 3)))
        knots, off = model.drift[s]
        P = pairs.get(s, [])
        if not P:  # nothing split (or no revisits): keep the drift
            rep[str(s)] = {"pairs": 0, "increment_max_m": 0.0, "drift_max_m": round(float(np.abs(off).max()), 3)}
            continue
        ti, tj = np.array([q[0] for q in P]), np.array([q[1] for q in P])
        diff, n = np.array([q[2] for q in P]), np.array([q[3] for q in P], float)
        x = _solve_drift(ti, tj, diff, n, knots, times)
        model.drift[s] = (knots, off + x)
        for i in rec.reg_image_ids():
            p = priors_raw.get(rec.image(i).name)
            if p is not None and p.source == s and p.time is not None:
                inc_by_image[p.name] = _interp(knots, x, p.time)
        far = np.abs(ti - tj) > model.drift_knot_s
        rep[str(s)] = {"pairs": len(P), "split": int(n.sum()), "merged": int(sum(q[4] for q in P)),
                       "far_pairs": int(far.sum()),
                       "far_split_median_m": np.round(np.median(np.linalg.norm(diff[far], axis=1)), 3).item()
                       if far.any() else None,
                       "increment_max_m": round(float(np.abs(x).max()), 3),
                       "drift_max_m": round(float(np.abs(off + x).max()), 3),
                       "drift_range_z_m": round(float(np.ptp((off + x)[:, 2])), 3)}
    _shift_images(rec, inc_by_image)
    return {"knots_s": model.drift_knot_s, "sources": rep}


def _update_drift_joint(model, rec, priors_raw, by_src, pairs, cross):
    """update_drift with revisits across sources: one solve over all sources (_solve_drift_joint)."""
    blocks = {}
    for s, times in by_src.items():
        times = np.asarray(times, float)
        if s not in model.drift:
            t0, t1 = float(times.min()), float(times.max())
            K = max(2, int(np.ceil((t1 - t0) / model.drift_knot_s)) + 1)
            model.drift[s] = (np.linspace(t0, t1, K), np.zeros((K, 3)))
        blocks[s] = (model.drift[s][0], times)
    eqs = [(s, ti, s, tj, d, n) for s, P in pairs.items() for ti, tj, d, n, _ in P]
    eqs += [(a, ta, b, tb, d, n) for a, ta, b, tb, d, n, _ in cross]
    ref = priors_raw[by_order(priors_raw)[0]].source
    n_cross = {s: sum(1 for c in cross if s in (c[0], c[2])) for s in blocks}
    anchored = {ref} | {s for s, k in n_cross.items() if k < MIN_CROSS}
    xs = _solve_drift_joint(blocks, eqs, anchored)
    inc_by_image, rep = {}, {}
    for s, x in xs.items():
        knots, off = model.drift[s]
        model.drift[s] = (knots, off + x)
        P = pairs.get(s, [])
        rep[str(s)] = {"pairs": len(P), "cross_pairs": n_cross[s], "anchored": s in anchored,
                       "increment_max_m": round(float(np.abs(x).max()), 3),
                       "drift_max_m": round(float(np.abs(off + x).max()), 3),
                       "drift_range_z_m": round(float(np.ptp((off + x)[:, 2])), 3)}
        if P:
            ti, tj = np.array([q[0] for q in P]), np.array([q[1] for q in P])
            diff = np.array([q[2] for q in P])
            far = np.abs(ti - tj) > model.drift_knot_s
            rep[str(s)].update(split=int(sum(q[3] for q in P)), merged=int(sum(q[4] for q in P)),
                               far_pairs=int(far.sum()),
                               far_split_median_m=np.round(np.median(np.linalg.norm(diff[far], axis=1)), 3).item()
                               if far.any() else None)
    for i in rec.reg_image_ids():
        p = priors_raw.get(rec.image(i).name)
        if p is not None and p.source in xs and p.time is not None:
            inc_by_image[p.name] = _interp(model.drift[p.source][0], xs[p.source], p.time)
    pairs_split = {}
    for a, _, b, _, d, n, _ in cross:
        pairs_split.setdefault(f"{min(a, b)}-{max(a, b)}", []).append(float(np.linalg.norm(d)))
    _shift_images(rec, inc_by_image)
    return {"knots_s": model.drift_knot_s, "sources": rep,
            "cross": {k: {"pairs": len(v), "split_median_m": round(float(np.median(v)), 3)}
                      for k, v in sorted(pairs_split.items())}}


def _shift_images(rec, inc_by_name):
    """Move cameras by per-image increments (C += d)."""
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        d = inc_by_name.get(img.name)
        if d is None:
            continue
        frame = rec.frame(img.frame_id)
        params = np.array(frame.rig_from_world.params)
        params[4:] -= _pose(rec, img)[0] @ d  # t = -R C
        pose = frame.rig_from_world
        pose.params = params
        frame.rig_from_world = pose


def copy_model(model):
    return copy.deepcopy(model)

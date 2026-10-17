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
estimated after every round's BA, starting with round 0.
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
    boresight: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad, camera frame
    dt: float = 0.0  # s
    abs_rejected: set = field(default_factory=set)  # image names
    rel_rejected: set = field(default_factory=set)  # (name_i, name_j)
    outlier_quantile: float = 0.999
    start_round: int = 1  # round 0 runs with unconverged intrinsics and 40 px thresholds
    max_scale_change: float = 2.0  # per round
    floors: tuple = (0.1, 0.1, 0.02, 0.01)  # abs rot deg, abs pos m, rel rot deg, rel pos m

    def summary(self):
        a, r = self.abs_sigmas, self.rel_sigmas
        return {"abs_sigmas": {"pos_horizontal_m": round(a.pos_horizontal_m, 3), "pos_vertical_m": round(a.pos_vertical_m, 3),
                               "roll_pitch_deg": round(a.roll_pitch_deg, 3), "yaw_deg": round(a.yaw_deg, 3)},
                "rel_sigmas": {"rot_deg": round(r.rot_deg, 3), "rot_per_deg": round(r.rot_per_deg, 4), "pos_m": round(r.pos_m, 4), "pos_per_m": round(r.pos_per_m, 5)},
                "boresight_deg": np.round(np.degrees(self.boresight), 3).tolist(),
                "boresight_norm_deg": round(float(np.degrees(np.linalg.norm(self.boresight))), 3),
                "dt_ms": round(self.dt * 1000, 1),
                "source_offsets_m": {str(k): np.round(v, 3).tolist() for k, v in sorted(self.source_offsets.items())},
                "abs_rejected": sorted(self.abs_rejected), "rel_rejected": len(self.rel_rejected)}


def corrected_priors(priors, model):
    """Priors with the systematic corrections (boresight, time offset, source offsets) applied."""
    B = exp_so3(model.boresight)
    out = {}
    for name, p in priors.items():
        R = B @ p.R_cw
        C = p.center + (p.velocity if p.velocity is not None else 0.0) * model.dt
        C = C + model.source_offsets.get(p.source, 0.0)
        out[name] = Prior.from_pose(p.name, p.frame, p.pts, R, C, p.velocity, p.source, p.order, p.time)
    return out


def consecutive_pairs(priors, names):
    """Relative-prior pairs: time-consecutive images of the same source."""
    return same_source_consecutive(priors, names)


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
    model.boresight = np.median(np.array([raw_abs[n][0] for n in names]), axis=0)
    V = np.array([priors_raw[n].velocity for n in names])
    # world-frame position error C_param - C_prior = -R^T r_pos
    D = np.array([-_pose(rec, rec.find_image_with_name(n))[0].T @ raw_abs[n][1] for n in names])
    D -= np.array([model.source_offsets.get(priors_raw[n].source, np.zeros(3)) for n in names])
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
    model.abs_rejected = {n for n, d in d2_abs.items() if d > thr}
    model.rel_rejected = {k for k, d in d2_rel.items() if d > thr} if model.use_relative else set()
    rep["abs_d2_median"] = round(float(np.median(list(d2_abs.values()))), 3)
    rep["rel_d2_median"] = round(float(np.median(list(d2_rel.values()))), 3) if d2_rel else None
    rep["abs_rejected"] = len(model.abs_rejected)
    rep["rel_rejected"] = len(model.rel_rejected)
    rep["model"] = model.summary()
    return rep


# --- multi-video: per-source position offsets ----------------------------------------
# Each flight has its own GNSS bias (~1-3 m) and possibly takeoff-height error, so the
# videos' priors disagree by a near-constant shift. Before the shift is known, each video
# triangulates its own copy of the shared scene ("split" points); once corrected, the
# tracks merge and the remaining shift shows up in the per-source prior residuals.

def _point_ids(image):
    return np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D], np.int64)


def split_point_edges(rec, priors, db_path):
    """Per source pair (a, b), a < b: X_a - X_b over verified cross-source matches whose
    keypoints have different 3D points (n_split), and the count already merged."""
    import pycolmap
    reg = {i: rec.image(i) for i in rec.reg_image_ids()}
    src = {i: priors[im.name].source for i, im in reg.items() if im.name in priors}
    ids, xyz = {}, {pid: p.xyz for pid, p in rec.points3D.items()}
    db = pycolmap.Database.open(db_path)
    pair_ids, geoms = db.read_two_view_geometries()
    db.close()
    edges = {}
    for pid, g in zip(pair_ids, geoms):
        i1, i2 = pycolmap.pair_id_to_image_pair(pid)
        if i1 not in src or i2 not in src or src[i1] == src[i2]:
            continue
        m = np.asarray(g.inlier_matches)
        if not len(m):
            continue
        for i in (i1, i2):
            if i not in ids:
                ids[i] = _point_ids(reg[i])
        p1, p2 = ids[i1][m[:, 0]], ids[i2][m[:, 1]]
        both = (p1 >= 0) & (p2 >= 0)
        if src[i1] > src[i2]:
            i1, i2, p1, p2 = i2, i1, p2, p1
        e = edges.setdefault((src[i1], src[i2]), {"diffs": [], "merged": 0})
        e["merged"] += int(np.sum(both & (p1 == p2)))
        split = both & (p1 != p2)
        e["diffs"] += [xyz[a] - xyz[b] for a, b in zip(p1[split], p2[split])]
    return {k: {"n_split": len(v["diffs"]), "merged": v["merged"],
                "median": np.median(np.array(v["diffs"]), axis=0) if v["diffs"] else np.zeros(3)}
            for k, v in edges.items()}


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


def _residual_offsets(rec, priors_raw, model, ref, min_images=10):
    """Per-source median of C_param - C_prior (corrected), relative to the reference source."""
    priors = corrected_priors(priors_raw, model)
    res = {}
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        p = priors.get(img.name)
        if p is None:
            continue
        R, t = _pose(rec, img)
        res.setdefault(p.source, []).append(-R.T @ t - p.center)
    med = {s: np.median(np.array(v), axis=0) for s, v in res.items() if len(v) >= min_images}
    if ref not in med:
        return {}
    return {s: m - med[ref] for s, m in med.items() if s != ref}


def _shift_cameras(rec, priors, inc):
    """Move the cameras of each source by its offset increment (C += d)."""
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        p = priors.get(img.name)
        if p is None or p.source not in inc:
            continue
        frame = rec.frame(img.frame_id)
        params = np.array(frame.rig_from_world.params)
        R = _pose(rec, img)[0]
        params[4:] -= R @ inc[p.source]  # t = -R C
        pose = frame.rig_from_world
        pose.params = params
        frame.rig_from_world = pose


def update_source_offsets(model, rec, priors_raw, db_path=None, min_split=30):
    """One per-source offset update after a BA (no-op for single-source projects).
    Reference = source of the first image. Split points drive the update while they
    outnumber merged ones on an edge (bootstrap; the cameras are shifted too, because
    their copies are not linked yet); afterwards the prior residuals do."""
    sources = {p.source for p in priors_raw.values()}
    if len(sources) < 2 or not model.estimate_source_offsets:
        return None
    ref = priors_raw[by_order(priors_raw)[0]].source
    edges = split_point_edges(rec, priors_raw, db_path) if db_path else {}
    use = {k: e for k, e in edges.items() if e["n_split"] >= max(min_split, e["merged"])}
    if use:
        inc, method = _solve_offsets(use, ref), "split_points"
        _shift_cameras(rec, priors_raw, inc)  # so the next triangulation already merges the copies
    else:
        inc, method = _residual_offsets(rec, priors_raw, model, ref), "residuals"
    for s, d in inc.items():
        model.source_offsets[s] = model.source_offsets.get(s, np.zeros(3)) + d
    return {"method": method, "reference": int(ref),
            "increment_m": {str(s): np.round(d, 3).tolist() for s, d in sorted(inc.items())},
            "offsets_m": {str(s): np.round(d, 3).tolist() for s, d in sorted(model.source_offsets.items())},
            "edges": {f"{a}-{b}": {"n_split": e["n_split"], "merged": e["merged"],
                                   "median_m": np.round(e["median"], 3).tolist()} for (a, b), e in edges.items()}}


def copy_model(model):
    return copy.deepcopy(model)

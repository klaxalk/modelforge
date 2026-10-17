"""Prior-initialized reconstruction and refinement.

init:   cameras/rigs/frames/images from the COLMAP database, every frame registered
        at its prior pose (no incremental registration).
rounds: triangulate with poses fixed (stock COLMAP point_triangulator path), then
        BA with free poses + 6-DoF covariance-weighted absolute pose priors;
        thresholds tighten each round because priors start ~100 px off.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions



@dataclass
class Round:
    max_reproj_px: float  # triangulation merge/complete + mapper filter threshold
    loss_scale_px: float  # Cauchy scale on reprojection residuals in BA
    max_angle_error_deg: float = 2.0


DEFAULT_ROUNDS = (Round(40.0, 8.0, 5.0), Round(12.0, 2.0, 3.0), Round(4.0, 1.0, 2.0))


def init_from_database(db_path, priors):
    """Reconstruction with all database images registered at their prior poses."""
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.Reconstruction()
    for cam in db.read_all_cameras():
        rec.add_camera(cam)
    for rig in db.read_all_rigs():
        rec.add_rig(rig)
    for frame in db.read_all_frames():
        rec.add_frame(frame)
    for img in db.read_all_images():
        rec.add_image(img)
    db.close()
    missing = []
    for image_id, img in rec.images.items():
        prior = priors.get(img.name)
        if prior is None:
            missing.append(img.name)
            continue
        frame = rec.frame(img.frame_id)
        frame.rig_from_world = prior.rigid()  # trivial rig: camera == rig
        rec.register_frame(img.frame_id)
    if missing:
        raise ValueError(f"no prior for {len(missing)} images, e.g. {missing[:3]}")
    return rec


def triangulate(rec, db_path, image_path, out_dir, rnd):
    opts = pycolmap.IncrementalPipelineOptions()
    tri = opts.triangulation
    tri.merge_max_reproj_error = rnd.max_reproj_px
    tri.complete_max_reproj_error = rnd.max_reproj_px
    tri.create_max_angle_error = rnd.max_angle_error_deg
    tri.continue_max_angle_error = rnd.max_angle_error_deg
    opts.mapper.filter_max_reproj_error = rnd.max_reproj_px
    os.makedirs(out_dir, exist_ok=True)
    return pycolmap.triangulate_points(rec, db_path, image_path, out_dir, clear_points=True,
                                       options=opts, refine_intrinsics=False)


class _Progress(pyceres.IterationCallback):
    def __init__(self, fn):
        pyceres.IterationCallback.__init__(self)
        self.fn, self.costs = fn, []

    def __call__(self, summary):
        self.costs.append(summary.cost)
        if self.fn:
            self.fn(summary)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def prior_ba(rec, priors, model=None, loss_scale_px=1.0, prior_loss_scale=3.0,
             refine_intrinsics=True, max_iterations=500, on_iteration=None, relative=True):
    """BA over all registered images with reprojection + 6-DoF prior terms.

    Absolute priors on every image (except model.abs_rejected), and, if
    model.use_relative, relative priors between time-consecutive images of one source (except
    model.rel_rejected). No gauge fixing: the priors pin position, rotation and
    metric scale in ENU.
    """
    from modelforge.sfm import robust
    model = model or robust.PriorModel(use_relative=False)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = refine_intrinsics
    ba_options.refine_extra_params = refine_intrinsics
    ba_options.refine_principal_point = False
    ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba_options.ceres.loss_function_scale = loss_scale_px
    ba_options.ceres.solver_options.max_num_iterations = max_iterations
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        ba_config.add_image(image_id)

    ba = pycolmap.create_default_ceres_bundle_adjuster(ba_options, ba_config, rec)
    problem = ba.problem
    # Losses on whitened residuals (sigma units). Absolute: Huber, whose pull on a far-off camera
    # stays constant. Cauchy's fades, so a camera seeing few points (each in 2-view tracks that
    # slide with it) ran off 35 m in round 0 on berlin_3_1 pieces; gross prior outliers are
    # handled by explicit rejection instead. Relative: Cauchy (heavy-tailed, ~23 % rejected).
    abs_loss = pyceres.HuberLoss(prior_loss_scale)
    prior_loss = pyceres.CauchyLoss(prior_loss_scale)
    poses = {}
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        pose = rec.frame(img.frame_id).rig_from_world
        if problem.has_parameter_block(pose.params):  # else: no observations, stays at its prior
            poses[img.name] = pose
    n_abs = n_rel = 0
    for name, pose in poses.items():
        if name in model.abs_rejected:
            continue
        prior = priors[name]
        cost = pycolmap.cost_functions.AbsolutePosePriorCost(prior.covariance(model.abs_sigmas), prior.rigid())
        problem.add_residual_block(cost, abs_loss, [pose.params])
        n_abs += 1
    if model.use_relative and relative:
        for a, b in robust.consecutive_pairs(priors, poses):
            if (a, b) in model.rel_rejected:
                continue
            R, t = robust.relative_prior(priors[a], priors[b])
            i_from_j = pycolmap.Rigid3d()
            i_from_j.params = np.r_[robust.rot_to_quat_xyzw(R), t]
            cost = pycolmap.cost_functions.RelativePosePriorCost(
                robust.relative_covariance(model, priors[a], priors[b]), i_from_j)
            problem.add_residual_block(cost, prior_loss, [poses[a].params, poses[b].params])
            n_rel += 1

    solver_options = pyceres.SolverOptions(ba_options.ceres.create_solver_options(ba_config, problem))
    progress = _Progress(on_iteration)
    solver_options.callbacks = [progress]
    solver_options.update_state_every_iteration = True
    summary = pyceres.SolverSummary()
    t0 = time.time()
    pyceres.solve(solver_options, problem, summary)
    return {"iterations": len(progress.costs), "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost, "usable": summary.IsSolutionUsable(),
            "termination": str(summary.termination_type).split(".")[-1],
            "num_abs_priors": n_abs, "num_rel_priors": n_rel, "seconds": round(time.time() - t0, 1)}


def rot_angle_deg(qa, qb):  # (x, y, z, w)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


def center_of(params):
    x, y, z, w = params[:4]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return -R.T @ params[4:]


def evaluate(rec, priors):
    """Reconstruction statistics and per-image deviation from the priors."""
    rec.update_point_3d_errors()  # point errors are cached; refresh after poses/points moved
    dpos, drot, dhoriz, dvert = [], [], [], []
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        p = np.array(rec.frame(img.frame_id).rig_from_world.params)
        pr = priors[img.name]
        d = center_of(p) - pr.center
        dpos.append(np.linalg.norm(d))
        dhoriz.append(np.linalg.norm(d[:2]))
        dvert.append(abs(d[2]))
        drot.append(rot_angle_deg(p[:4], pr.params[:4]))
    cam = next(iter(rec.cameras.values()))

    def q(x):
        x = np.asarray(x)
        return {"median": round(float(np.median(x)), 3), "p90": round(float(np.percentile(x, 90)), 3),
                "max": round(float(x.max()), 3)} if len(x) else {}

    # per-point errors; the mean alone is useless right after BA, before filtering,
    # when a few points behind a camera or near zero depth have huge errors
    err = np.array([p.error for p in rec.points3D.values()])
    return {
        "reg_images": rec.num_reg_images(), "points3D": rec.num_points3D(),
        "observations": rec.compute_num_observations(),
        "mean_track_length": round(rec.compute_mean_track_length(), 2),
        "mean_reproj_error_px": round(rec.compute_mean_reprojection_error(), 3),
        "point_error_px": {**q(err), "frac_over_4px": round(float(np.mean(err > 4.0)), 4)} if len(err) else {},
        "camera": {"model": cam.model.name, "params": [round(float(v), 4) for v in cam.params]},
        "prior_deviation": {"position_m": q(dpos), "horizontal_m": q(dhoriz), "vertical_m": q(dvert),
                            "rotation_deg": q(drot)},
    }

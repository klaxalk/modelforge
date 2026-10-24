"""Prior-initialized reconstruction and refinement.

init:   cameras/rigs/frames/images from the COLMAP database, every frame registered
        at its prior pose (no incremental registration).
rounds: triangulate with poses fixed (stock COLMAP point_triangulator path), then
        BA with free poses + 6-DoF covariance-weighted absolute pose priors;
        thresholds tighten each round because priors start ~100 px off.
BA stops once the cameras have settled (Settle), not at Ceres' gradient tolerance: COLMAP's
solver options have no function / parameter tolerance, and on these prior-anchored problems the
cameras stop moving after ~10-80 iterations while Ceres runs hundreds more nudging points.
"""

import contextlib
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

from modelforge import jobs



@dataclass
class Round:
    max_reproj_px: float  # triangulation merge/complete + mapper filter threshold
    loss_scale_px: float  # Cauchy scale on reprojection residuals in BA
    max_angle_error_deg: float = 2.0


DEFAULT_ROUNDS = (Round(40.0, 8.0, 5.0), Round(12.0, 2.0, 3.0), Round(4.0, 1.0, 2.0))


@dataclass
class Settle:
    """BA is done when, over the last `window` iterations, no camera centre moved more than
    `pos_m`, no camera rotated more than `rot_deg` and no focal length changed more than `focal_px`.
    Displacement is measured across the whole window (not per step), so a slowly creeping camera
    keeps the solver going. Only accepted steps count (rejected ones move nothing). berlin_test (80 images): 834 -> ~125 iterations over 3 rounds, poses
    within 0.5 mm / 0.001 deg of running to Ceres' convergence."""
    pos_m: float = 0.0005
    rot_deg: float = 0.001
    focal_px: float = 0.01
    window: int = 10
    quantile: float = 100.0  # over the cameras: 100 = every camera; lower lets a slow tail creep on


def init_from_database(db_path, priors, names=None):
    """Reconstruction with all database images registered at their prior poses (names: only these)."""
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
    keep = None if names is None else set(names)
    for image_id, img in rec.images.items():
        if keep is not None and img.name not in keep:
            continue
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


_TRIANGULATING = re.compile(r"Triangulating image #\d+ \((\d+)\)")  # COLMAP's log: (k) = images done

POINT_BATCH = 100_000  # points per refine_points problem (~0.5 GB)


def refine_points(rec, loss_scale_px=1.0, max_iterations=50, batch=POINT_BATCH, progress=None):
    """Points-only BA: poses and intrinsics constant, so every point is its own 3x3 problem (sparse
    normal Cholesky on a block-diagonal system, no Schur complement over thousands of cameras).

    Solved in batches of `batch` points: a config with variable points and no images gives only
    constant-pose residuals for those points' tracks. One problem over all points ran out of memory
    on 8000 images (kladno_dira: ~10M points, killed at 30 GB); a batch takes a few hundred MB.
    progress(text, k, n) after every batch."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = opts.refine_extra_params = opts.refine_principal_point = False
    opts.refine_rig_from_world = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = loss_scale_px
    opts.ceres.solver_options.max_num_iterations = max_iterations
    ids = list(rec.point3D_ids())
    n = (len(ids) + batch - 1) // batch
    for k in range(0, len(ids), batch):
        config = pycolmap.BundleAdjustmentConfig()
        for point3D_id in ids[k:k + batch]:
            config.add_variable_point(point3D_id)
        ba = pycolmap.create_default_ceres_bundle_adjuster(opts, config, rec)
        solver_options = pyceres.SolverOptions(opts.ceres.create_solver_options(config, ba.problem))
        solver_options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        solver_options.num_threads = os.cpu_count() or 1  # Ceres' default is one thread
        pyceres.solve(solver_options, ba.problem, pyceres.SolverSummary())
        del ba
        if progress:
            progress("refining points: batch", k // batch + 1, n)


def triangulate(rec, db_path, image_path, out_dir, rnd, refine=False, progress=None):
    """Points from the database's matches at the registered poses (fixed). progress(text, k, n): images
    triangulated (read from COLMAP's log), then the point refinement's batches."""
    opts = pycolmap.IncrementalPipelineOptions()
    tri = opts.triangulation
    tri.merge_max_reproj_error = rnd.max_reproj_px
    tri.complete_max_reproj_error = rnd.max_reproj_px
    tri.create_max_angle_error = rnd.max_angle_error_deg
    tri.continue_max_angle_error = rnd.max_angle_error_deg
    opts.mapper.filter_max_reproj_error = rnd.max_reproj_px
    # no points-only global BA after triangulating: the poses are fixed in it, and the prior BA that
    # follows refines the points anyway (kladno_dira, 4000 images: 10.5 of 12.5 min per triangulation).
    # Track completion / merging and retriangulation still run; the filter it would have applied
    # (reprojection error, triangulation angle) is done here instead. refine: the points alone are
    # refined first (refine_points; for the final model, which no prior BA follows).
    opts.ba_global_max_refinements = 0
    os.makedirs(out_dir, exist_ok=True)
    n_reg = rec.num_reg_images()

    def line(text):
        m = _TRIANGULATING.search(text)
        if m:
            progress("triangulating: image", int(m.group(1)), n_reg)
    with jobs.stderr_lines(line) if progress else contextlib.nullcontext():
        rec = pycolmap.triangulate_points(rec, db_path, image_path, out_dir, clear_points=True,
                                          options=opts, refine_intrinsics=False)
    if refine:
        refine_points(rec, rnd.loss_scale_px, progress=progress)
    pycolmap.ObservationManager(rec).filter_all_points3D(rnd.max_reproj_px, opts.mapper.filter_min_tri_angle)
    rec.write(out_dir)
    return rec


class _Progress(pyceres.IterationCallback):
    """Forwards iterations to `fn` and ends the solve once the cameras have settled."""

    def __init__(self, fn, poses=(), cameras=(), settle=None, token=None, skip=None, skip_round=None):
        pyceres.IterationCallback.__init__(self)
        self.fn, self.costs, self.token, self.skip, self.skip_round = fn, [], token, skip, skip_round
        self.poses, self.cameras, self.settle = list(poses), list(cameras), settle
        self.history, self.settled = [], False

    def _state(self):
        P = np.array([p.params for p in self.poses]) if self.poses else np.zeros((0, 7))
        C = np.array([center_of(p) for p in P]).reshape(-1, 3)
        f = np.array([c.params[0] for c in self.cameras])
        return C, P[:, :4].copy(), f

    def __call__(self, summary):
        self.costs.append(summary.cost)
        if self.fn:
            self.fn(summary)
        if self.token is not None and self.token.cancelled:  # Stop: abort the solve now
            return pyceres.CallbackReturnType.SOLVER_ABORT
        if self.skip is not None and self.skip.is_set():  # Skip: keep the current state, end the solve
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        if self.skip_round is not None and self.skip_round.is_set():  # Skip this round: end this solve only
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        s = self.settle
        # only accepted steps count: a rejected one (e.g. "Matrix not positive definite", Ceres then damps harder)
        # leaves the cameras where they were, and a run of them is a stuck solver, not a settled one
        if s and self.poses and (summary.iteration == 0 or summary.step_is_successful):
            self.history = (self.history + [self._state()])[-(s.window + 1):]
            if len(self.history) > s.window:
                (c0, q0, f0), (c1, q1, f1) = self.history[0], self.history[-1]
                dpos = np.percentile(np.linalg.norm(c1 - c0, axis=1), s.quantile)
                drot = np.percentile(np.degrees(2 * np.arccos(np.abs(np.sum(q0 * q1, 1)).clip(0, 1))), s.quantile)
                dfoc = np.abs(f1 - f0).max() if len(f0) else 0.0
                if dpos < s.pos_m and drot < s.rot_deg and dfoc < s.focal_px:
                    self.settled = True
                    return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def prior_ba(rec, priors, model=None, loss_scale_px=1.0, prior_loss_scale=3.0,
             refine_intrinsics=True, max_iterations=500, on_iteration=None, relative=True, settle=Settle(),
             token=None, skip=None, images=None, image_weight=1.0, skip_round=None):
    """BA over all registered images with reprojection + 6-DoF prior terms.

    Absolute priors on every image (except model.abs_rejected), and, if
    model.use_relative, relative priors between time-consecutive images of one source (except
    model.rel_rejected). No gauge fixing: the priors pin position, rotation and
    metric scale in ENU. `settle` (None: run to Ceres' own convergence) ends the solve once the
    cameras stop moving.

    images (image ids, None = all): a local BA. Only these poses are free; their points are free and
    keep their observations in the other images, whose poses are held constant (COLMAP adds those
    as constant-pose residuals). Relative priors join pairs with both images inside.

    image_weight scales every reprojection term (a solve on thinned points: see local_ba).
    """
    from modelforge.sfm import robust
    model = model or robust.PriorModel(use_relative=False)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = refine_intrinsics
    ba_options.refine_extra_params = refine_intrinsics
    ba_options.refine_principal_point = False
    ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba_options.ceres.loss_function_scale = loss_scale_px
    ba_options.ceres.loss_function_weight = image_weight
    # direct sparse Schur at any size: COLMAP switches to iterative Schur above 1000 images, which on
    # kladno_dira (4000 images, 2.9M points) took 27-150 s per iteration against 11 s direct
    ba_options.ceres.max_num_images_direct_sparse_cpu_solver = 1_000_000
    ba_options.ceres.solver_options.max_num_iterations = max_iterations
    ba_config = pycolmap.BundleAdjustmentConfig()
    local = images is not None
    for image_id in (images if local else rec.reg_image_ids()):
        ba_config.add_image(image_id)
    if local:  # the window's points, with their observations outside it (constant poses)
        for image_id in images:
            for p2 in rec.image(image_id).get_observation_points2D():
                ba_config.add_variable_point(p2.point3D_id)

    ba = pycolmap.create_default_ceres_bundle_adjuster(ba_options, ba_config, rec)
    problem = ba.problem
    # Losses on whitened residuals (sigma units). Absolute: Huber, whose pull on a far-off camera
    # stays constant. Cauchy's fades, so a camera seeing few points (each in 2-view tracks that
    # slide with it) ran off 35 m in round 0 on berlin_3_1 pieces; gross prior outliers are
    # handled by explicit rejection instead. Relative: Cauchy (heavy-tailed, ~23 % rejected).
    abs_loss = pyceres.HuberLoss(prior_loss_scale)
    prior_loss = pyceres.CauchyLoss(prior_loss_scale)
    poses = {}
    for image_id in (images if local else rec.reg_image_ids()):
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
    cameras = [rec.cameras[c] for c in sorted(rec.cameras)] if refine_intrinsics else []
    progress = _Progress(on_iteration, poses.values(), cameras, settle, token, skip, skip_round)
    solver_options.callbacks = [progress]
    solver_options.update_state_every_iteration = True
    summary = pyceres.SolverSummary()
    t0 = time.time()
    pyceres.solve(solver_options, problem, summary)
    if skip is not None and skip.is_set():
        termination = "SKIPPED"
    elif skip_round is not None and skip_round.is_set():  # this solve ended by the user; the next one runs
        termination = "SKIPPED_ROUND"
        skip_round.clear()
    else:
        termination = "SETTLED" if progress.settled else str(summary.termination_type).split(".")[-1]
    return {"iterations": len(progress.costs), "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost, "usable": summary.IsSolutionUsable(),
            "termination": termination,
            "num_abs_priors": n_abs, "num_rel_priors": n_rel, "seconds": round(time.time() - t0, 1)}


@dataclass
class Local:
    """Local BA for large projects, in two levels:
    1. a global BA on a thinned copy of the points (refine.thin_points: about `points_per_image` per
       image); it sets the large-scale shape, which the priors decide by averaging over all images;
    2. windows of `window` consecutive images in capture order, each adjusted with all its points and
       everything outside held fixed (prior_ba(images=...)); each later sweep shifts the windows by
       window / sweeps, so every boundary lies inside a window once.
    Why both: the images tie neighbouring cameras together far tighter than the priors hold them
    (mm vs 0.6 m), so a window whose neighbours are still at their priors snaps onto them, and more
    sweeps barely help (block Gauss-Seidel on stiff coupling; synthetic chain: global 2.1 cm, two
    windows 6.9 cm, priors 8.6 cm). After level 1 the neighbours are right.
    One global BA over thousands of images with all points is what is slow: kladno_dira (4000 images,
    2.9M points) took 11 s per iteration (direct sparse Schur; COLMAP's default iterative Schur
    above 1000 images 27-150 s), a 300-image window 0.2 s. Intrinsics stay fixed in the windows
    (shared by all images): calibrate them first (stage_refine does, on a subset)."""
    window: int = 300
    sweeps: int = 1
    points_per_image: int = 300
    weighted: bool = True  # scale the thinned reprojection terms by full / thinned observations
    # the global pass ends when 90 % of the cameras settle: the last 1 % creep for hundreds of
    # iterations (kladno_dira: strict rule at 271), and the windows settle them in tens
    global_settle: Settle = field(default_factory=lambda: Settle(pos_m=0.001, rot_deg=0.002, quantile=90))
    min_images: int = 1000  # BA mode "auto": local from this many images on


def thin_points(rec, per_image, seed=0):
    """Delete 3D points until each image keeps about `per_image` of them: points in random order, each
    kept while one of its images still has fewer than per_image kept. Returns the number deleted.
    (No observation is lost for good: the next triangulation starts afresh.) Random, not longest
    tracks first: in drone footage the longest tracks are distant features, which pin rotation but
    triangulate depth poorly, so a solve on them leaned on the priors for position."""
    ids = list(rec.point3D_ids())
    tracks = [[e.image_id for e in rec.point3D(p).track.elements] for p in ids]
    count = {}
    deleted = 0
    for k in np.random.default_rng(seed).permutation(len(ids)):
        tr = tracks[k]
        if any(count.get(i, 0) < per_image for i in tr):
            for i in tr:
                count[i] = count.get(i, 0) + 1
        else:
            rec.delete_point3D(ids[k])
            deleted += 1
    return deleted


def calibration_subset(priors, names, blocks=5, size=60, cameras=None):
    """Image names for calibrating the intrinsics and the prior model before a local BA: `blocks`
    runs of `size` consecutive images spread evenly over the capture order (consecutive, so they
    share matches; spread, so they see different parts and attitudes of the flight). cameras ({name: camera id},
    several cameras): a camera none of the runs covers gets a run of its own, from the middle of its images."""
    from modelforge.sfm.priors import by_order
    order = by_order(priors, names)
    if len(order) <= blocks * size:
        return order
    starts = np.linspace(0, len(order) - size, blocks).round().astype(int)
    out = [order[s + k] for s in starts for k in range(size)]
    if cameras:
        have = {cameras.get(n) for n in out}
        for c in sorted({cameras.get(n) for n in order} - have, key=str):
            own = [n for n in order if cameras.get(n) == c]
            a = max(0, (len(own) - size) // 2)
            out += own[a:a + size]
    return out


def windows(order, local):
    """[(sweep, ids)] for image ids in capture order (none for 0 sweeps: the global pass only)."""
    n, w = len(order), max(2, local.window)
    out = []
    for s in range(local.sweeps):
        start = (s * w) // local.sweeps
        cuts = ([0] if start else []) + list(range(start, n, w)) + [n]
        out += [(s, order[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return out


def local_ba(rec, priors, model, local, on_window=None, token=None, skip=None, refine_intrinsics=False,
             global_pass=True, **kw):
    """Two-level local BA (see Local). on_window(k, n, sweep, rec) before each step, k = -1 for the
    global pass on the thinned copy (rec = that copy, whose poses Ceres moves). kw go to prior_ba;
    refine_intrinsics applies to the global pass only. global_pass=False: the windows alone, for
    poses already consistent at large scale (chunked: after the pose graph). Returns the stats."""
    t0 = time.time()
    names = {rec.image(i).name: i for i in rec.reg_image_ids()}
    from modelforge.sfm.priors import by_order
    order = [names[n] for n in by_order(priors, [n for n in names if n in priors])]
    wins = windows(order, local)
    g, deleted, n_points, weight, points_seconds = {"iterations": 0, "num_abs_priors": 0, "num_rel_priors": 0,
                                                    "usable": True, "termination": "NONE"}, 0, rec.num_points3D(), 1.0, 0.0
    if global_pass:
        thin = pycolmap.Reconstruction(rec)
        n_points, n_obs = thin.num_points3D(), thin.compute_num_observations()
        deleted = thin_points(thin, local.points_per_image)
        weight = n_obs / max(1, thin.compute_num_observations()) if local.weighted else 1.0
        if on_window:
            on_window(-1, len(wins), 0, thin)
        g = prior_ba(thin, priors, model, token=token, skip=skip, refine_intrinsics=refine_intrinsics,
                     image_weight=weight, **{**kw, "settle": local.global_settle})
        for frame_id in rec.reg_frame_ids():
            rec.frame(frame_id).rig_from_world = thin.frame(frame_id).rig_from_world
        for cid, cam in thin.cameras.items():
            rec.camera(cid).params = cam.params
        del thin
        # the full points were triangulated at the old poses (kladno_dira: median 93 px off after the
        # global pass, 0.5 px after this); a window starting from them first swings its cameras away
        tp = time.time()
        refine_points(rec, kw.get("loss_scale_px", 1.0), max_iterations=5)
        points_seconds = round(time.time() - tp, 1)
    stats = {"global": {**g, "points": n_points - deleted, "of_points": n_points, "image_weight": round(weight, 2)},
             "refine_points_seconds": points_seconds,
             "iterations": g["iterations"], "usable": bool(g["usable"]), "windows": len(wins), "window": local.window,
             "sweeps": local.sweeps, "num_abs_priors": g["num_abs_priors"], "num_rel_priors": g["num_rel_priors"],
             "window_iterations": 0, "settled": 0, "max_window_iterations": 0}
    tw = time.time()
    for k, (sweep, ids) in enumerate(wins):
        if (token is not None and token.cancelled) or (skip is not None and skip.is_set()):
            break
        if on_window:
            on_window(k, len(wins), sweep, rec)
        r = prior_ba(rec, priors, model, token=token, skip=skip, images=ids, refine_intrinsics=False, **kw)
        stats["iterations"] += r["iterations"]
        stats["window_iterations"] += r["iterations"]
        stats["max_window_iterations"] = max(stats["max_window_iterations"], r["iterations"])
        stats["settled"] += r["termination"] == "SETTLED"
        stats["usable"] &= bool(r["usable"])
    stats["termination"] = ("SKIPPED" if skip is not None and skip.is_set() else
                            f"LOCAL (global {g['termination']}, {stats['settled']}/{len(wins)} windows settled)")
    stats["window_seconds"] = round(time.time() - tw, 1)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


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
        **({"cameras": {str(c): {"model": rec.cameras[c].model.name, "size": [rec.cameras[c].width,
                                                                              rec.cameras[c].height],
                                 "params": [round(float(v), 4) for v in rec.cameras[c].params]}
                        for c in sorted(rec.cameras)}} if len(rec.cameras) > 1 else {}),
        "prior_deviation": {"position_m": q(dpos), "horizontal_m": q(dhoriz), "vertical_m": q(dvert),
                            "rotation_deg": q(drot)},
    }

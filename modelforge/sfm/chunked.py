"""Chunked refinement for large projects (thousands to tens of thousands of images).

The flight in capture order is cut into overlapping chunks of consecutive images. Each chunk is
triangulated and bundle-adjusted on its own (its own images, its matches only, anchored by its own
priors), so memory and time per chunk do not grow with the project. A pose graph then solves all
cameras at once without any 3D points: the relative poses each chunk measured (stiff: images bind
neighbouring cameras to mm) plus the absolute priors (loose: GNSS, ~0.7 m), so the chunks' shapes
are kept and their placement averages the priors over the whole flight. Cameras shared by
overlapping chunks tie the chunks together. With 6 parameters per camera and no points, the pose
graph solves 10k+ cameras in seconds.

pipeline.stage_refine then triangulates all images at the pose-graph poses and polishes them with
one sweep of windows (refine.local_ba without its global pass): the neighbours a window holds fixed
are consistent now, and the windows see the observations between chunks (revisits included), which
the chunks themselves do not.

Revisits (loop closures, `revisit_edges`): chunks and chunk edges only relate images close in capture
order, so two passes over the same ground follow their own (drifting) priors and triangulate as offset
copies; the windows cannot pull them together either (kladno_dira, 2883 images: passes > 60 s apart split
0.39 m median, p90 1.57 m, after 3 drift passes). After the pose graph, all images are triangulated, and
every image pair far apart in capture order whose inlier matches landed on different 3D points gives an
edge: its current relative pose with the second camera moved by the median offset between the copies
(X_a - X_b). The pose graph is solved again with these edges, the chunk edges and the priors, and the
loop repeats until the copies merge (`revisit_done_m`). The loop-closure edges are Huber, not Cauchy like the chunk
edges: a 1.5 m split at sigma 3 cm is 50 sigma, where Cauchy (3) keeps ~0.4 % of the weight, so on kladno_dira the
passes stayed apart (1.53 -> 0.73 -> 0.76 m); Huber keeps a constant pull, bounded per edge.

What the pose graph cannot do, a global BA does (dira2k: revisits 0.05 m after global refine, 0.24 m chunked): after
the drift passes, `global_passes` global BAs over all cameras on a thinned copy of the points
(`global_points_per_image`, refine.thin_points; a full one does not fit in memory at ~6000 images), each after a
fresh triangulation, which merges the copies that came closer.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pyceres
import pycolmap

from modelforge.sfm import refine
from modelforge.sfm.refine import center_of


@dataclass
class Chunked:
    size: int = 500  # images per chunk
    overlap: int = 100  # images shared by consecutive chunks
    # BA mode "auto" without match statistics: chunked from this many images on (with them: pipeline.auto_ba_mode,
    # by the memory a global refine needs)
    min_images: int = 2500
    edge_steps: tuple = (1, 4, 16, 64)  # pose-graph edges i -> i + step within a chunk
    # pose-graph edge sigmas: a + b * baseline (position), c (rotation); far below the priors'
    # 0.7 m / 0.7 deg, so the chunks' shapes win, but finite, so overlapping chunks can agree
    rel_pos_m: float = 0.005
    rel_pos_per_m: float = 0.002
    rel_rot_deg: float = 0.02
    min_points: int = 30  # a chunk's camera with fewer points gives no pose-graph edges
    polish_sweeps: int = 1  # window sweeps after the pose graph (0: none)
    polish_window: int = 300
    revisit_passes: int = 3  # triangulate all + loop-closure edges + pose graph again (0: none)
    revisit_min_split: int = 20  # split matches an image pair needs for an edge
    revisit_pos_m: float = 0.03  # edge sigmas: position floor (the offset's own uncertainty when larger), rotation
    revisit_rot_deg: float = 0.1
    revisit_done_m: float = 0.05  # stop when the median split of the far pairs is below this
    revisit_loss: float = 3.0  # Huber scale of the loop-closure edges (whitened)
    # height: the barometer is smooth over seconds but off between flights (berlin_all: 5-10 m), so in the pose graph
    # a flight's priors set its height only up to a free offset (pulled weakly to 0), and consecutive photos keep
    # their barometric height difference (never rejected). Without it, cross-flight links lifted pieces of a flight
    # (6-7 m steps between consecutive photos) and the pieces drew stacked copies of the buildings.
    flight_z: bool = True
    global_passes: int = 2  # triangulate all + global BA on thinned points, before the polish (0: none)
    global_points_per_image: int = 100
    # pose-graph linear solver: "cgnr" (conjugate gradients, Jacobi) or "cholesky" (sparse normal Cholesky, CHOLMOD;
    # loop edges fill its factor in: dira2k loop graphs 48-65 s vs 1-3 s, same revisit split; docs/todo_posegraph_speed.md)
    posegraph_solver: str = "cgnr"


def solver_options(name):
    """pose_graph `solver` options for Chunked.posegraph_solver."""
    if name == "cgnr":
        return {"linear_solver_type": pyceres.LinearSolverType.CGNR}
    if name != "cholesky":
        raise ValueError(f"unknown pose-graph solver {name!r} (cholesky / cgnr)")
    return {}


def chunks(order, size, overlap):
    """Overlapping runs of `size` consecutive items (the last one may be shorter, but not tiny)."""
    n = len(order)
    if n <= size:
        return [list(order)]
    step = max(1, size - overlap)
    starts = list(range(0, n - overlap, step))
    if n - starts[-1] < size // 2 and len(starts) > 1:  # fold a short tail into the chunk before
        starts.pop()
    return [list(order[a:(b + overlap if k + 1 < len(starts) else n)])
            for k, (a, b) in enumerate(zip(starts, starts[1:] + [n]))]


def _rigid(params):
    r = pycolmap.Rigid3d()
    r.params = np.asarray(params, float)
    return r


def relative(pa, pb):
    """i_from_j = T_a T_b^-1 for cam_from_world params (x, y, z, w, tx, ty, tz)."""
    return (_rigid(pa) * _rigid(pb).inverse()).params


def edge_covariance(cfg, pa, pb):
    ca, cb = (center_of(np.asarray(p, float)) for p in (pa, pb))
    s_pos = cfg.rel_pos_m + cfg.rel_pos_per_m * float(np.linalg.norm(ca - cb))
    s_rot = np.radians(cfg.rel_rot_deg)
    return np.diag([s_rot ** 2] * 3 + [s_pos ** 2] * 3)


def chunk_edges(cfg, names, poses):
    """Pose-graph edges (a, b, i_from_j, cov) from one chunk's solution: each image to the ones
    edge_steps later in the chunk (several steps, so the chunk stays rigid, not a chain)."""
    out = []
    for k, a in enumerate(names):
        for s in cfg.edge_steps:
            if k + s < len(names):
                b = names[k + s]
                if a in poses and b in poses:
                    out.append((a, b, relative(poses[a], poses[b]), edge_covariance(cfg, poses[a], poses[b])))
    return out


FLIGHT_Z_SIGMA_M = 20.0  # flight_z: the weak pull of a flight's height offset to its barometric height
Z_STEP_SIGMA_M = 0.1  # flight_z: consecutive photos' height difference vs the barometer's, sigma = this + per s * dt
Z_STEP_PER_S = 0.05
Z_STEP_MAX_S = 10.0  # photos further apart in time are not tied this way


def _cz(p):
    """Camera centre height of cam_from_world params p (qx qy qz qw tx ty tz) and its gradient (7,): c = -R^T t."""
    x, y, z, w = p[:4]
    t = p[4:7]
    col = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])  # third column of R
    g = np.empty(7)
    g[0] = -(2 * z * t[0] - 2 * w * t[1] - 4 * x * t[2])
    g[1] = -(2 * w * t[0] + 2 * z * t[1] - 4 * y * t[2])
    g[2] = -(2 * x * t[0] + 2 * y * t[1])
    g[3] = -(2 * y * t[0] - 2 * x * t[1])
    g[4:7] = -col
    return -float(col @ t), g


class _ZPrior(pyceres.CostFunction):
    """(camera height - (prior height + the flight's offset)) / sigma; blocks [pose (7), offset (1)]."""

    def __init__(self, z0, sigma):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([7, 1])
        self.z0, self.s = z0, sigma

    def Evaluate(self, params, residuals, jacobians):
        cz, g = _cz(params[0])
        residuals[0] = (cz - self.z0 - params[1][0]) / self.s
        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = g / self.s
            if jacobians[1] is not None:
                jacobians[1][0] = -1.0 / self.s
        return True


class _ZStep(pyceres.CostFunction):
    """((height b - height a) - the barometer's difference) / sigma; blocks [pose a, pose b]."""

    def __init__(self, dz, sigma):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([7, 7])
        self.dz, self.s = dz, sigma

    def Evaluate(self, params, residuals, jacobians):
        za, ga = _cz(params[0])
        zb, gb = _cz(params[1])
        residuals[0] = (zb - za - self.dz) / self.s
        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = -ga / self.s
            if jacobians[1] is not None:
                jacobians[1][:] = gb / self.s
        return True


class _Scalar(pyceres.CostFunction):
    """x / sigma for a 1-parameter block."""

    def __init__(self, sigma):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])
        self.s = sigma

    def Evaluate(self, params, residuals, jacobians):
        residuals[0] = params[0][0] / self.s
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = 1.0 / self.s
        return True


def pose_graph(init, edges, priors, model, rejected=(), max_iterations=200, edge_loss=None, loops=(),
               loop_loss=3.0, solver=None, on_iteration=None, token=None, flight_z=False):
    """Solve all poses: chunk edges (RelativePosePriorCost, Cauchy) + loop-closure edges (`loops`, the same
    form, Huber at `loop_loss`) + absolute priors (AbsolutePosePriorCost, Huber at 3 sigma, as in
    refine.prior_ba; `rejected` names get none). init: name -> params, mutated in place. solver: extra
    pyceres.SolverOptions attributes; on_iteration(summary) per iteration; token: Stop aborts. flight_z
    (Chunked.flight_z): heights from the priors only up to a free offset per flight (source), and consecutive photos
    keep the barometer's height difference; the offsets are returned in "flight_z_m" (add them to the flight's
    priors). Returns the solver stats."""
    t0 = time.time()
    params = {n: np.array(p, float) for n, p in init.items()}
    problem = pyceres.Problem()
    edge_loss = edge_loss or pyceres.CauchyLoss(3.0)
    for a, b, i_from_j, cov in edges:
        cost = pycolmap.cost_functions.RelativePosePriorCost(cov, _rigid(i_from_j))
        problem.add_residual_block(cost, edge_loss, [params[a], params[b]])
    for a, b, i_from_j, cov in loops:
        cost = pycolmap.cost_functions.RelativePosePriorCost(cov, _rigid(i_from_j))
        problem.add_residual_block(cost, pyceres.HuberLoss(loop_loss), [params[a], params[b]])
    n_abs = 0
    abs_loss = pyceres.HuberLoss(3.0)
    sig = model.abs_sigmas
    if flight_z:  # the priors' heights go in below, relative to a free offset per flight
        import copy
        sig = copy.copy(model.abs_sigmas)
        sig.pos_vertical_m = 1e4
    offsets, n_steps = {}, 0
    for n, p in params.items():
        if n in rejected or n not in priors:
            continue
        cost = pycolmap.cost_functions.AbsolutePosePriorCost(priors[n].covariance(sig), priors[n].rigid())
        problem.add_residual_block(cost, abs_loss, [p])
        n_abs += 1
        if flight_z:
            src = priors[n].source
            if src not in offsets:
                offsets[src] = np.zeros(1)
                problem.add_residual_block(_Scalar(FLIGHT_Z_SIGMA_M), None, [offsets[src]])
            problem.add_residual_block(_ZPrior(float(priors[n].center[2]), model.abs_sigmas.pos_vertical_m), abs_loss,
                                       [p, offsets[src]])
    if flight_z:  # consecutive photos of a flight: the barometer's height difference, never rejected
        seq = sorted((n for n in params if n in priors), key=lambda n: (priors[n].source, priors[n].time))
        for a, b in zip(seq, seq[1:]):
            pa, pb = priors[a], priors[b]
            dt = abs(pb.time - pa.time)
            if pa.source != pb.source or dt > Z_STEP_MAX_S:
                continue
            problem.add_residual_block(_ZStep(float(pb.center[2] - pa.center[2]), Z_STEP_SIGMA_M + Z_STEP_PER_S * dt),
                                       pyceres.HuberLoss(3.0), [params[a], params[b]])
            n_steps += 1
    for p in params.values():
        if problem.has_parameter_block(p):
            problem.set_manifold(p, pyceres.ProductManifold(pyceres.EigenQuaternionManifold(), pyceres.EuclideanManifold(3)))
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = max_iterations
    options.num_threads = os.cpu_count() or 1
    for k, v in (solver or {}).items():
        setattr(options, k, v)
    if on_iteration or token is not None:
        options.callbacks = [refine._Progress(on_iteration, token=token)]
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    for n, p in params.items():
        init[n] = p
    z = {"flight_z_m": {int(k): round(float(v[0]), 3) for k, v in sorted(offsets.items())}, "z_steps": n_steps} \
        if flight_z else {}
    return {**z, "cameras": len(params), "edges": len(edges), "loops": len(loops), "abs_priors": n_abs, "iterations": summary.num_successful_steps + summary.num_unsuccessful_steps,
            "initial_cost": summary.initial_cost, "final_cost": summary.final_cost,
            "termination": str(summary.termination_type).split(".")[-1], "seconds": round(time.time() - t0, 1)}


def revisit_edges(rec, priors, db_path, cfg):
    """Loop-closure edges (a, b, i_from_j, cov) between images far apart in capture order (beyond the chunk
    edges' reach) whose inlier matches triangulated as different 3D points: the current relative pose with
    camera b moved by median(X_a - X_b), so its copy lands on a's. Also the split statistics of those pairs."""
    from modelforge.sfm import robust
    reach = max(cfg.edge_steps)
    out, split = [], []
    for same in (True, False):  # within a source and across sources: both are revisits
        for im1, im2, d, merged in robust._split_pairs(rec, priors, db_path, same_source=same):
            a, b = im1.name, im2.name
            if len(d) < cfg.revisit_min_split or a not in priors or b not in priors or \
                    abs(priors[a].order - priors[b].order) <= reach:
                continue
            off = np.median(d, axis=0)
            spread = 1.4826 * np.median(np.linalg.norm(d - off, axis=1)) / np.sqrt(len(d))
            split.append(float(np.linalg.norm(off)))
            pa = np.array(rec.frame(im1.frame_id).rig_from_world.params)
            pb = np.array(rec.frame(im2.frame_id).rig_from_world.params)
            moved = pb.copy()
            moved[4:] = pb[4:] - _rigid(pb).rotation.matrix() @ off  # centre c -> c + off
            s_pos = max(cfg.revisit_pos_m, float(spread))
            cov = np.diag([np.radians(cfg.revisit_rot_deg) ** 2] * 3 + [s_pos ** 2] * 3)
            out.append((a, b, relative(pa, moved), cov))
    st = {"pairs": len(out), "split_median_m": round(float(np.median(split)), 3) if split else 0.0,
          "split_p90_m": round(float(np.percentile(split, 90)), 3) if split else 0.0}
    return out, st


def global_ba(rec, priors, model, cfg, **kw):
    """One BA over all cameras on about cfg.global_points_per_image points per image (refine.thin_points, deleted
    from rec itself: the next triangulation starts afresh), the reprojection terms weighted by full / thinned
    observations so images and priors balance as in a full BA; it ends when 90 % of the cameras settle
    (refine.Local.global_settle). Intrinsics fixed (calibrated on the subset). kw go to refine.prior_ba."""
    n_points, n_obs = rec.num_points3D(), rec.compute_num_observations()
    deleted = refine.thin_points(rec, cfg.global_points_per_image)
    weight = n_obs / max(1, rec.compute_num_observations())
    g = refine.prior_ba(rec, priors, model, refine_intrinsics=False, image_weight=weight,
                        settle=refine.Local().global_settle, **kw)
    return {**g, "points": n_points - deleted, "of_points": n_points, "image_weight": round(weight, 2)}

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
loop repeats until the copies merge (`revisit_done_m`).
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import pyceres
import pycolmap

from modelforge.sfm.refine import center_of


@dataclass
class Chunked:
    size: int = 500  # images per chunk
    overlap: int = 100  # images shared by consecutive chunks
    # BA mode "auto": chunked from this many images on. Chunks + pose graph + windowed polish never pull two passes
    # over the same ground together the way one global BA does (dira2k, 2000 images, revisit split median /
    # p90: chunked 0.24 / 0.88 m, global 0.05 / 0.15 m), so global runs as long as it is affordable (dira2k:
    # 67 min, 7.2 GB vs chunked 24 min, 3.9 GB; 4000 images: > 30 min for round 0 alone, ~20 GB)
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


def pose_graph(init, edges, priors, model, rejected=(), max_iterations=200, edge_loss=None):
    """Solve all poses: chunk edges (RelativePosePriorCost) + absolute priors (AbsolutePosePriorCost,
    Huber at 3 sigma, as in refine.prior_ba; `rejected` names get none). init: name -> params,
    mutated in place. Returns the solver stats."""
    t0 = time.time()
    params = {n: np.array(p, float) for n, p in init.items()}
    problem = pyceres.Problem()
    edge_loss = edge_loss or pyceres.CauchyLoss(3.0)
    for a, b, i_from_j, cov in edges:
        cost = pycolmap.cost_functions.RelativePosePriorCost(cov, _rigid(i_from_j))
        problem.add_residual_block(cost, edge_loss, [params[a], params[b]])
    n_abs = 0
    abs_loss = pyceres.HuberLoss(3.0)
    for n, p in params.items():
        if n in rejected or n not in priors:
            continue
        cost = pycolmap.cost_functions.AbsolutePosePriorCost(priors[n].covariance(model.abs_sigmas), priors[n].rigid())
        problem.add_residual_block(cost, abs_loss, [p])
        n_abs += 1
    for p in params.values():
        if problem.has_parameter_block(p):
            problem.set_manifold(p, pyceres.ProductManifold(pyceres.EigenQuaternionManifold(), pyceres.EuclideanManifold(3)))
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = max_iterations
    options.num_threads = os.cpu_count() or 1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    for n, p in params.items():
        init[n] = p
    return {"cameras": len(params), "edges": len(edges), "abs_priors": n_abs, "iterations": summary.num_successful_steps + summary.num_unsuccessful_steps,
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

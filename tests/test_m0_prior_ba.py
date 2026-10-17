"""M0: pycolmap BA problem + pyceres + 6-DoF covariance-weighted pose priors.

Synthetic scene, poses perturbed, priors = ground truth shifted by a known world
offset. Reprojection alone cannot see a global shift, so the solution only lands
on the shifted poses if the priors are really in the problem and fix the gauge.
"""

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

SHIFT = np.array([2.0, -1.0, 0.5])  # world offset applied to the priors
ROT_NOISE_RAD = np.radians(3.0)
TRANS_NOISE = 0.3


def quat_mul(a, b):  # (x, y, z, w)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def center(params):
    """Camera center in world from cam_from_world params [qx qy qz qw tx ty tz]."""
    return -quat_to_rot(params[:4]).T @ params[4:]


def rot_angle(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return 2 * np.arccos(min(1.0, d))


class CountingCallback(pyceres.IterationCallback):
    def __init__(self):
        pyceres.IterationCallback.__init__(self)
        self.costs = []

    def __call__(self, summary):
        self.costs.append(summary.cost)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def make_scene(seed=0):
    pycolmap.set_random_seed(seed)
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_cameras_per_rig = 1
    opts.num_frames_per_rig = 20
    opts.num_points3D = 300
    rec = pycolmap.synthesize_dataset(opts)
    noise = pycolmap.SyntheticNoiseOptions()
    noise.point2D_stddev = 0.5  # px
    pycolmap.synthesize_noise(noise, rec)
    return rec


def run(with_priors):
    rec = make_scene()
    rng = np.random.default_rng(1)

    frames = {fid: rec.frame(fid) for fid in rec.reg_frame_ids()}
    gt = {fid: np.array(f.rig_from_world.params) for fid, f in frames.items()}

    # priors: same orientation, centers moved by SHIFT  ->  t' = t - R @ SHIFT
    priors = {}
    for fid, p in gt.items():
        q = p[:4]
        prior = pycolmap.Rigid3d()
        prior.params = np.r_[q, p[4:] - quat_to_rot(q) @ SHIFT]
        priors[fid] = prior

    # perturb the current estimate
    for fid, f in frames.items():
        p = gt[fid].copy()
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.normal() * ROT_NOISE_RAD
        dq = np.r_[axis * np.sin(ang / 2), np.cos(ang / 2)]
        p[:4] = quat_mul(dq, p[:4])
        p[4:] += rng.normal(size=3) * TRANS_NOISE
        pose = f.rig_from_world
        pose.params = p
        f.rig_from_world = pose

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        ba_config.add_image(image_id)
    # no gauge fixing: the priors must provide it

    ba = pycolmap.create_default_ceres_bundle_adjuster(ba_options, ba_config, rec)
    problem = ba.problem
    assert isinstance(problem, pyceres.Problem)
    n_reproj = problem.num_residual_blocks()

    if with_priors:
        cov = np.diag([np.radians(0.5) ** 2] * 3 + [0.05 ** 2] * 3)  # rotation first, then translation
        for fid, f in frames.items():
            cost = pycolmap.cost_functions.AbsolutePosePriorCost(cov, priors[fid])
            problem.add_residual_block(cost, pyceres.TrivialLoss(), [f.rig_from_world.params])
        assert problem.num_residual_blocks() == n_reproj + len(frames)
    else:
        # reprojection-only: fix the gauge the usual way so the problem is well posed
        problem.set_parameter_block_constant(frames[min(frames)].rig_from_world.params)

    solver_options = pyceres.SolverOptions(ba_options.ceres.create_solver_options(ba_config, problem))
    callback = CountingCallback()
    solver_options.callbacks = [callback]
    solver_options.update_state_every_iteration = True
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, problem, summary)
    print(summary.BriefReport())

    assert summary.IsSolutionUsable()
    assert len(callback.costs) > 1, "iteration callback did not fire"
    assert callback.costs[-1] < callback.costs[0]

    pos_err, rot_err = [], []
    for fid, f in frames.items():
        est = np.array(f.rig_from_world.params)
        target = np.array(priors[fid].params)
        pos_err.append(np.linalg.norm(center(est) - center(target)))
        rot_err.append(rot_angle(est[:4], target[:4]))
    pos_err, rot_err = np.array(pos_err), np.degrees(rot_err)
    print(f"[priors={with_priors}] center error vs priors: mean {pos_err.mean():.4f} max {pos_err.max():.4f} "
          f"| rotation error: mean {rot_err.mean():.4f} max {rot_err.max():.4f} deg")
    return pos_err, rot_err


def test_prior_ba_follows_shifted_priors():
    pos_err, rot_err = run(with_priors=True)
    # initial perturbation was ~0.3 (+ a 2.3 global shift) and ~3 deg
    assert pos_err.max() < 0.05
    assert rot_err.max() < 0.5


def test_without_priors_does_not_reach_shifted_poses():
    # negative control: reprojection alone cannot recover the world shift
    pos_err, _ = run(with_priors=False)
    assert pos_err.mean() > 0.5 * np.linalg.norm(SHIFT)

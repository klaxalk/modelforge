"""M3: prior self-calibration, outlier rejection and systematic-error recovery on a
synthetic scene with known ground truth (cameras ~5 units from the points)."""

import numpy as np
import pycolmap
import pytest

from modelforge.sfm import refine, robust
from modelforge.sfm.priors import Prior, PriorSigmas


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def make_scene(seed=0, n=40, points=500):
    pycolmap.set_random_seed(seed)
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs, o.num_cameras_per_rig, o.num_frames_per_rig, o.num_points3D = 1, 1, n, points
    rec = pycolmap.synthesize_dataset(o)
    noise = pycolmap.SyntheticNoiseOptions()
    noise.point2D_stddev = 0.5
    pycolmap.synthesize_noise(noise, rec)
    gt = {}
    for k, image_id in enumerate(sorted(rec.reg_image_ids())):
        img = rec.image(image_id)
        p = np.array(rec.frame(img.frame_id).rig_from_world.params)
        R = quat_to_rot(p[:4])
        gt[img.name] = (k, R, -R.T @ p[4:])
    return rec, gt


def make_priors(gt, rng, pos_sigma=0.0, rot_sigma_deg=0.0, boresight=None, dt=0.0, outlier=None):
    """Priors = ground truth corrupted by iid noise, a camera-frame boresight, a time offset
    (C_prior = C_true - v dt) and optionally one gross outlier (name, shift, angle_deg)."""
    priors = {}
    B = robust.exp_so3(boresight) if boresight is not None else np.eye(3)
    for name, (k, R, C) in gt.items():
        v = rng.normal(size=3) * 2.0
        dR = robust.exp_so3(rng.normal(size=3) * np.radians(rot_sigma_deg))
        Rp = B.T @ dR @ R  # true R = B * R_prior  =>  R_prior = B^T R
        Cp = C + rng.normal(size=3) * pos_sigma - v * dt
        if outlier and name == outlier[0]:
            Cp = Cp + outlier[1]
            Rp = robust.exp_so3(np.radians(outlier[2]) * np.array([0, 0, 1.0])) @ Rp
        priors[name] = Prior.from_pose(name, k, float(k), Rp, Cp, v)
    return priors


def set_poses(rec, priors):
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        frame = rec.frame(img.frame_id)
        frame.rig_from_world = priors[img.name].rigid()


def run_rounds(rec, priors, model, rounds=4):
    reps = []
    for _ in range(rounds):
        corrected = robust.corrected_priors(priors, model)
        ba = refine.prior_ba(rec, corrected, model, refine_intrinsics=False)
        assert ba["usable"]
        reps.append((ba, robust.update(model, rec, priors)))
    return reps


def pose_errors(rec, gt):
    pos, rot = {}, {}
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        p = np.array(rec.frame(img.frame_id).rig_from_world.params)
        R = quat_to_rot(p[:4])
        _, Rg, Cg = gt[img.name]
        pos[img.name] = np.linalg.norm(-R.T @ p[4:] - Cg)
        rot[img.name] = np.degrees(np.linalg.norm(robust.log_so3(R @ Rg.T)))
    return pos, rot


def test_sigma_calibration_recovers_true_noise():
    rec, gt = make_scene(seed=1)
    rng = np.random.default_rng(1)
    priors = make_priors(gt, rng, pos_sigma=0.2, rot_sigma_deg=0.5)
    set_poses(rec, priors)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.6, 0.6, 1.5, 1.5),  # 3x too loose
                              use_relative=False, estimate_systematic=False)
    run_rounds(rec, priors, model)
    a = model.abs_sigmas
    print(model.summary())
    assert 0.13 < a.pos_horizontal_m < 0.3 and 0.13 < a.pos_vertical_m < 0.3
    assert 0.33 < a.roll_pitch_deg < 0.75 and 0.33 < a.yaw_deg < 0.75


def test_gross_outlier_prior_is_rejected():
    rec, gt = make_scene(seed=2)
    rng = np.random.default_rng(2)
    bad = sorted(gt)[17]
    priors = make_priors(gt, rng, pos_sigma=0.1, rot_sigma_deg=0.3, outlier=(bad, np.array([2.0, -1.5, 1.0]), 10.0))
    set_poses(rec, priors)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.1, 0.1, 0.3, 0.3), use_relative=False,
                              estimate_systematic=False)
    reps = run_rounds(rec, priors, model)
    assert bad in model.abs_rejected
    assert len(model.abs_rejected) <= 2  # the outlier, at most one false alarm
    pos, rot = pose_errors(rec, gt)
    print(f"outlier camera error: {pos[bad]:.4f} units, {rot[bad]:.3f} deg; rejected {sorted(model.abs_rejected)}")
    assert pos[bad] < 0.05 and rot[bad] < 0.3  # recovered by the images, not dragged by its prior


def test_boresight_is_recovered():
    rec, gt = make_scene(seed=3)
    rng = np.random.default_rng(3)
    b_true = np.radians(2.0) * np.array([0.5, 0.5, 0.7]) / np.linalg.norm([0.5, 0.5, 0.7])
    priors = make_priors(gt, rng, pos_sigma=0.05, rot_sigma_deg=0.2, boresight=b_true)
    set_poses(rec, priors)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.05, 0.05, 0.2, 0.2), use_relative=False, calibrate=False)
    run_rounds(rec, priors, model)
    err = np.degrees(np.linalg.norm(model.boresight - b_true))
    print(f"boresight est {np.degrees(model.boresight).round(3)} true {np.degrees(b_true).round(3)} err {err:.3f} deg")
    assert err < 0.2
    _, rot = pose_errors(rec, gt)
    assert np.median(list(rot.values())) < 0.2


def test_time_offset_is_recovered():
    rec, gt = make_scene(seed=4)
    rng = np.random.default_rng(4)
    priors = make_priors(gt, rng, pos_sigma=0.05, rot_sigma_deg=0.2, dt=0.3)  # 0.6 units at |v| ~ 3.5
    set_poses(rec, priors)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.05, 0.05, 0.2, 0.2), use_relative=False, calibrate=False)
    run_rounds(rec, priors, model)
    print(f"dt est {model.dt * 1000:.1f} ms (true 300)")
    assert abs(model.dt - 0.3) < 0.03


def test_relative_priors_enter_the_problem():
    rec, gt = make_scene(seed=5)
    priors = make_priors(gt, np.random.default_rng(5), pos_sigma=0.05, rot_sigma_deg=0.2)
    set_poses(rec, priors)
    model = robust.PriorModel()
    ba = refine.prior_ba(rec, priors, model, refine_intrinsics=False)
    assert ba["num_abs_priors"] == len(gt) and ba["num_rel_priors"] == len(gt) - 1


def test_settle_rule():
    """BA ends once no camera moved more than Settle.pos_m across the window; a slow creep that is
    tiny per step but large across the window keeps it going."""
    import pyceres

    class Pose:
        def __init__(self):
            self.params = np.array([0, 0, 0, 1.0, 0, 0, 0])

    class Summary:
        cost = 1.0

    poses = [Pose(), Pose()]
    s = refine.Settle(pos_m=0.0005, window=10)
    cb = refine._Progress(None, poses, (), s)
    go = pyceres.CallbackReturnType.SOLVER_CONTINUE
    for _ in range(30):  # creep 0.1 mm per iteration: 1 mm across the window
        poses[1].params = poses[1].params + [0, 0, 0, 0, 1e-4, 0, 0]
        assert cb(Summary()) == go
    still = 0
    while cb(Summary()) == go:  # still: the window's displacement drops below pos_m within one window
        still += 1
        assert still <= s.window
    assert cb.settled and still >= 4  # not before 0.5 mm of creep has left the window
    cb = refine._Progress(None, poses, (), None)  # settle=None never stops early
    assert all(cb(Summary()) == go for _ in range(30))

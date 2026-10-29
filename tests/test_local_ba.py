"""Local prior BA for large projects (refine.Local / local_ba: a global pass on thinned points, then
windows) on a synthetic scene: it reaches the full global BA's poses, while windows alone stay near
the priors. Also the windows, the thinning, the calibration subset and the relative-prior pairs
across gaps."""

import json
import os
import re
import shutil
import time

import numpy as np
import pytest

from modelforge.sfm import refine, robust
from modelforge.sfm.priors import Prior
from tests.test_m3_robust import make_priors, make_scene, pose_errors, set_poses


def scene(seed=3, n=60, reach=4):
    """make_scene with local visibility, as along a flight: each point keeps its observations in the
    images within `reach` of a random image in capture order (pycolmap's synthetic cameras all see
    every point, the worst case for windows)."""
    rec, gt = make_scene(seed, n, points=2000)
    rng = np.random.default_rng(seed + 100)
    ids = sorted(rec.reg_image_ids())
    for pid in list(rec.point3D_ids()):
        c = rng.integers(n)
        obs = [(el.image_id, el.point2D_idx) for el in rec.point3D(pid).track.elements]
        drop = [o for o in obs if abs(ids.index(o[0]) - c) > reach]
        if len(obs) - len(drop) < 2:
            rec.delete_point3D(pid)
            continue
        for image_id, idx in drop:
            rec.delete_observation(image_id, idx)
    priors = make_priors(gt, np.random.default_rng(seed), pos_sigma=0.05, rot_sigma_deg=0.5)
    set_poses(rec, priors)
    return rec, gt, priors


def test_local_ba_matches_global():
    model = robust.PriorModel(use_relative=False)
    rec_g, gt, priors = scene()
    refine.prior_ba(rec_g, priors, model, refine_intrinsics=False, settle=None)
    rec_l, _, _ = scene()
    stats = refine.local_ba(rec_l, priors, model, refine.Local(window=15, sweeps=2, points_per_image=40), settle=None)
    assert stats["windows"] == 4 + 5 and stats["usable"] and stats["num_abs_priors"] == 60
    assert stats["global"]["points"] < 0.5 * stats["global"]["of_points"]
    prior_err = np.median([np.linalg.norm(priors[n].center - gt[n][2]) for n in gt])
    pg, rg = pose_errors(rec_g, gt)
    pl, rl = pose_errors(rec_l, gt)
    names = sorted(pg)
    pg, pl = np.array([pg[n] for n in names]), np.array([pl[n] for n in names])
    assert np.median(pg) < 0.5 * prior_err  # the images do improve on the priors ...
    assert np.median(pl) < 0.5 * prior_err  # ... in windows as well
    # and end close to the global solution (measured: 0.1 mm-1.5 mm against 8.6 cm prior error)
    assert np.median(np.abs(pl - pg)) < 0.05 * prior_err, (np.median(pl), np.median(pg), prior_err)


def test_windows_alone_snap_to_their_neighbours():
    """Why the global pass: with the neighbours fixed at their priors, a window follows them (the images
    tie cameras together far tighter than the priors), so it barely improves on the priors."""
    model = robust.PriorModel(use_relative=False)
    rec, gt, priors = scene()
    order = sorted(rec.reg_image_ids(), key=lambda i: priors[rec.image(i).name].order)
    refine.prior_ba(rec, priors, model, refine_intrinsics=False, settle=None, images=order[:30])
    refine.prior_ba(rec, priors, model, refine_intrinsics=False, settle=None, images=order[30:])
    prior_err = np.median([np.linalg.norm(priors[n].center - gt[n][2]) for n in gt])
    assert np.median(list(pose_errors(rec, gt)[0].values())) > 0.5 * prior_err  # measured 6.9 vs 8.6 cm


def test_thin_points_keeps_each_image_covered():
    rec, _, _ = scene()
    before = {i: rec.image(i).num_points3D for i in rec.reg_image_ids()}
    deleted = refine.thin_points(rec, 50)
    after = {i: rec.image(i).num_points3D for i in rec.reg_image_ids()}
    assert deleted > 0 and rec.num_points3D() > 0
    assert all(after[i] >= min(50, before[i]) for i in before)
    assert np.median(list(after.values())) < 0.6 * np.median(list(before.values()))


def test_windows_cover_every_boundary():
    order = list(range(10))
    w = refine.windows(order, refine.Local(window=4, sweeps=2))
    assert [ids for s, ids in w if s == 0] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [ids for s, ids in w if s == 1] == [[0, 1], [2, 3, 4, 5], [6, 7, 8, 9]]
    assert refine.windows(order, refine.Local(window=20, sweeps=2)) == [(0, order), (1, order)]


def test_calibration_subset_and_gap_pairs():
    priors = {f"i{k:04d}": Prior.from_pose(f"i{k:04d}", k, float(k), np.eye(3), np.zeros(3), np.zeros(3), order=k)
              for k in range(1000)}
    sub = refine.calibration_subset(priors, list(priors), blocks=5, size=60)
    assert len(sub) == 300 and sub[:2] == ["i0000", "i0001"] and sub[-1] == "i0999"
    pairs = robust.consecutive_pairs(priors, sub)
    assert len(pairs) == 5 * 59  # consecutive within each block, none across the gaps
    assert ("i0000", "i0001") in pairs
    # a missing image or two is still bridged
    assert robust.consecutive_pairs(priors, ["i0000", "i0002", "i0003"]) == [("i0000", "i0002"), ("i0002", "i0003")]
    assert refine.calibration_subset(priors, sub[:100]) == sub[:100]


REAL = os.environ.get("MODELFORGE_TEST_PROJECT",  # any project with matches (database.db)
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


def chunk_solution(rec, priors, model, names):
    """One chunk on its own, as chunked refinement runs it: only its images, its observations."""
    import pycolmap
    c = pycolmap.Reconstruction(rec)
    keep = set(names)
    for i in list(c.reg_image_ids()):
        if c.image(i).name not in keep:
            c.deregister_frame(c.image(i).frame_id)
    refine.prior_ba(c, priors, model, refine_intrinsics=False, settle=None)
    return {c.image(i).name: np.array(c.frame(c.image(i).frame_id).rig_from_world.params) for i in c.reg_image_ids()}


def test_chunks_and_pose_graph_match_global():
    """Chunks of 20 (overlap 6) on their own, then the pose graph: as close to the global BA as the
    global BA is to the truth's scatter, while each chunk alone sits nearer its own priors."""
    from modelforge.sfm import chunked
    model = robust.PriorModel(use_relative=False)
    rec_g, gt, priors = scene()
    refine.prior_ba(rec_g, priors, model, refine_intrinsics=False, settle=None)
    rec, _, _ = scene()
    order = sorted(gt, key=lambda n: priors[n].order)
    cfg = chunked.Chunked(size=20, overlap=6)
    parts = chunked.chunks(order, cfg.size, cfg.overlap)
    assert len(parts) == 4 and all(a[-cfg.overlap:] == b[:cfg.overlap] for a, b in zip(parts, parts[1:]))
    init, edges = {}, []
    for names in parts:
        sol = chunk_solution(rec, priors, model, names)
        edges += chunked.chunk_edges(cfg, names, sol)
        for n in names:
            init.setdefault(n, sol[n])
    st = chunked.pose_graph(init, edges, priors, model)
    assert st["cameras"] == 60 and st["abs_priors"] == 60 and st["termination"] == "CONVERGENCE"
    prior_err = np.median([np.linalg.norm(priors[n].center - gt[n][2]) for n in gt])
    pg = pose_errors(rec_g, gt)[0]
    pc = {n: np.linalg.norm(refine.center_of(init[n]) - gt[n][2]) for n in gt}
    assert np.median(list(pc.values())) < 0.5 * prior_err
    assert np.median([abs(pc[n] - pg[n]) for n in gt]) < 0.1 * prior_err, (np.median(list(pc.values())), np.median(list(pg.values())))


def test_chunks_cover_everything():
    from modelforge.sfm import chunked
    order = list(range(1234))
    parts = chunked.chunks(order, 500, 100)
    assert parts[0][0] == 0 and parts[-1][-1] == 1233 and sorted({x for p in parts for x in p}) == order
    assert all(len(p) >= 250 for p in parts) and all(a[-100:] == b[:100] for a, b in zip(parts, parts[1:]))
    assert chunked.chunks(order[:300], 500, 100) == [order[:300]]


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")),
                    reason="needs a matched project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_chunked_refine_from_gui(tmp_path):
    """Sparse tab, BA mode chunked: calibration rounds on the subset, chunks, the pose graph and the
    polish; the state line names them; snapshots calib_N and posegraph appear."""
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path)
    for name in ("poses.json", "images", "database.db"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    for name in ("ingest.json", "camera.json"):
        if os.path.exists(os.path.join(REAL, name)):
            shutil.copyfile(os.path.join(REAL, name), os.path.join(p, name))
    app = App(p, port=free_port())
    try:
        app.g_ba_mode.value = "chunked"
        app.g_chunk_size.value = 30
        app.g_chunk_overlap.value = 10
        app.run_stage("refine")
        seen, t0 = set(), time.time()
        while app.running and time.time() - t0 < 900:
            seen.add(app._job.get("detail", ""))
            time.sleep(0.05)
        assert "finished" in app.g_state.content, app.g_state.content
        for word in ("calibration round", "chunk 1/", "polish"):  # the pose graph takes ms here: see the report
            assert any(word in d for d in seen), (word, seen)
        r = json.load(open(os.path.join(p, "report.json")))["refine"]
        assert r["ba"]["mode"] == "chunked" and {"calib_0", "calib_1", "chunked"} <= set(r)
        c = r["chunked"]
        assert len(c["chunks"]) >= 3 and c["pose_graph"]["cameras"] == r["final"]["reg_images"]
        assert r["final"]["point_error_px"]["median"] < 1.0
        names = app.project.snapshot_names()
        assert names.index("calib_1") < names.index("posegraph") < names.index("refined")
    finally:
        app.server.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")),
                    reason="needs a matched project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_refine_shows_iterations_of_the_cap(tmp_path):
    """The state line shows "iteration k / cap" during BA, each round's final count ("BA k / cap iterations
    (SETTLED)"), and the History line lists every round's iterations."""
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path)
    for name in ("poses.json", "images", "database.db"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    for name in ("ingest.json", "camera.json"):
        if os.path.exists(os.path.join(REAL, name)):
            shutil.copyfile(os.path.join(REAL, name), os.path.join(p, name))
    app = App(p, port=free_port())
    try:
        app.g_ba_mode.value = "global"
        app.g_ba_iters.value = 300
        app.run_stage("refine")
        seen, t0 = set(), time.time()
        while app.running and time.time() - t0 < 900:
            seen.add(app._job.get("detail", ""))
            time.sleep(0.05)
        assert "finished" in app.g_state.content, app.g_state.content
        assert any(re.search(r"iteration \d+ / 300", d) for d in seen), seen
        assert any(re.search(r"BA \d+ / 300 iterations \((SETTLED|CONVERGENCE|NO_CONVERGENCE)\)", d)
                   for d in seen), seen
        assert re.search(r"BA iterations \d+, \d+, \d+ / 300", app.g_history.content), app.g_history.content
    finally:
        app.server.stop()


@pytest.mark.parametrize("amp, sigma, loss", [(1.0, 0.05, "huber"), (3.0, 0.3, "huber"), (3.0, 0.3, "cauchy")])
def test_revisit_edges_join_two_passes(tmp_path, amp, sigma, loss):
    """Two passes over the same scene, solved as separate chunks with drifting priors (0.8 units over the
    flight, `amp` times that): the pose graph alone leaves them as offset copies; loop-closure edges from the
    copies' offsets join them, rigidly right up to the gauge. As Huber terms (the default); with the chunk edges'
    Cauchy a large split carries almost no weight and the passes stay apart."""
    from modelforge.sfm import chunked
    from modelforge.sfm.priors import PriorSigmas
    from tests.test_multi_source import make_drift_scene

    def drift(t):
        return amp * np.array([0.3 * np.sin(t / 119 * np.pi), 0.0, 0.8 * t / 119])
    db_path, image_path, pri, gt = make_drift_scene(tmp_path, drift)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(sigma, sigma, 0.2, 0.2), calibrate=False,
                              estimate_systematic=False, use_relative=False, estimate_drift=False)
    cfg = chunked.Chunked(edge_steps=(1, 4))
    order = sorted(pri, key=lambda n: pri[n].order)
    rnd = refine.Round(4.0, 1.0, 2.0)
    init, edges = {}, []
    for k, names in enumerate((order[:20], order[20:])):  # one chunk per pass: nothing ties them but the priors
        c = refine.init_from_database(db_path, pri, names=names)
        c = refine.triangulate(c, db_path, image_path, str(tmp_path / f"chunk_{k}"), rnd)
        refine.prior_ba(c, pri, model, refine_intrinsics=False)
        sol = {c.image(i).name: np.array(c.frame(c.image(i).frame_id).rig_from_world.params) for i in c.reg_image_ids()}
        edges += chunked.chunk_edges(cfg, names, sol)
        init.update(sol)
    chunked.pose_graph(init, edges, pri, model)
    rec = refine.init_from_database(db_path, pri)
    ids = {rec.image(i).name: i for i in rec.reg_image_ids()}

    def apply():
        for n, i in ids.items():
            rec.frame(rec.image(i).frame_id).rig_from_world = chunked._rigid(init[n])

    def spread():  # camera centre errors after removing the common offset (the gauge)
        pos = np.array([refine.center_of(init[n]) - gt[n] for n in ids])
        return float(np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1)))
    apply()
    before = spread()
    splits = []
    for k in range(3):
        rec = refine.triangulate(rec, db_path, image_path, str(tmp_path / f"all_{k}"), rnd)
        loops, st = chunked.revisit_edges(rec, pri, db_path, cfg)
        splits.append(st["split_median_m"])
        if not loops or st["split_median_m"] < cfg.revisit_done_m:
            break
        init = {n: np.array(rec.frame(rec.image(i).frame_id).rig_from_world.params) for n, i in ids.items()}
        if loss == "huber":
            chunked.pose_graph(init, edges, pri, model, loops=loops)
        else:
            chunked.pose_graph(init, edges + loops, pri, model)
        apply()
    after = spread()
    print("split per pass", splits, "camera spread before", round(before, 3), "after", round(after, 3))
    if loss == "cauchy":  # measured: split 0.31 -> 1.0 -> 1.98, spread 1.03 -> 1.03
        assert splits[-1] > 0.2 and after > 0.5 * before
        return
    assert splits[0] > 0.2 and splits[-1] < cfg.revisit_done_m  # offset copies first, then merged
    # measured 0.34 -> 0.048 (amp 1, sigma 0.05; a global BA with these tight priors stays split: 0.34) and
    # 1.03 -> 0.11 (amp 3, sigma 0.3); the rest is the drift within each pass, which its chunk follows
    assert before > 0.2 and after < 0.2 * before


def test_auto_ba_mode_by_memory():
    """auto: global while its estimated peak fits in the available RAM, else chunked; without match statistics by
    the image count."""
    from modelforge import pipeline
    big = {"match": {"inlier_matches": 82_679_112}}  # kladno_dira, 5681 images: OOM-killed at 30 GB
    small = {"match": {"inlier_matches": 14_712_904}}  # dira2k: 7.2 GB measured
    assert pipeline.auto_ba_mode(big, 5681, available_gb=27)[0] == "chunked"
    assert pipeline.auto_ba_mode(small, 2000, available_gb=27)[0] == "global"
    assert pipeline.auto_ba_mode(small, 2000, available_gb=8)[0] == "chunked"
    assert pipeline.auto_ba_mode({}, 3000, available_gb=27)[0] == "chunked"
    assert pipeline.auto_ba_mode({}, 300, available_gb=27)[0] == "global"
    assert 7.2 < pipeline.global_ba_need_gb(14_712_904) < 15


@pytest.mark.parametrize("flight_z", [False, True])
def test_flight_height_offset_moves_whole_flights(flight_z):
    """Two flights over the same ground, flight 1's barometer 5 m high; loop edges tie only a stretch of flight 1 to
    flight 0 (where the matcher linked them). Without flight_z that stretch is pulled down alone and tears off its
    neighbours (a step, stacked copies; berlin_all 2026-10-29); with it the whole flight moves and its heights stay
    smooth."""
    from modelforge.sfm import chunked
    from modelforge.sfm.priors import PriorSigmas, rot_to_quat_xyzw
    R = np.diag([1.0, -1.0, -1.0])  # looking straight down
    q = rot_to_quat_xyzw(R)

    def params(C):
        return np.r_[q, -R @ C]
    truth, pri, order = {}, {}, 0
    for src in (0, 1):
        for k in range(40):
            n = f"f{src}_{k:03d}"
            C = np.array([2.0 * k, 3.0 * src, 30.0])
            truth[n] = params(C)
            Cp = C + np.array([0.0, 0.0, 5.0 if src == 1 else 0.0])  # the barometer's error
            pri[n] = Prior.from_pose(n, k, k, R, Cp, None, src, order, 1000.0 * src + 1.0 * k)
            order += 1
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.75, 0.5, 0.5, 1.5), calibrate=False, estimate_systematic=False,
                              use_relative=False, estimate_drift=False)
    cfg = chunked.Chunked(edge_steps=(1, 4))
    edges = []
    f0, f1 = [n for n in truth if n.startswith("f0")], [n for n in truth if n.startswith("f1")]
    for names in (f0, f1[:15], f1[15:25], f1[25:]):  # flight 1's chain broken where photos gave no chunk edges
        edges += chunked.chunk_edges(cfg, names, {n: truth[n] for n in names})
    cov = np.diag([np.radians(0.1) ** 2] * 3 + [0.03 ** 2] * 3)
    loops = [(f"f0_{k:03d}", f"f1_{k:03d}", chunked.relative(truth[f"f0_{k:03d}"], truth[f"f1_{k:03d}"]), cov)
             for k in range(15, 25)]  # only this stretch was matched across the flights
    init = {n: np.array(pri[n].params) for n in truth}
    pg = chunked.pose_graph(init, edges, pri, model, loops=loops, flight_z=flight_z)
    z0 = np.array([refine.center_of(init[f"f0_{k:03d}"])[2] for k in range(40)])
    z1 = np.array([refine.center_of(init[f"f1_{k:03d}"])[2] for k in range(40)])
    rel = z1 - z0  # truth 0 (the absolute height is a gauge)
    steps = np.abs(np.diff(z1))
    print("flight 1 - flight 0 heights", np.round(rel[[0, 14, 15, 24, 25, 39]], 2), "largest step",
          round(float(steps.max()), 2))
    if not flight_z:  # measured: the linked stretch at 0.01 m, the rest of flight 1 at +4.63 m
        assert steps.max() > 2.0
        return
    assert steps.max() < 0.1 and np.all(np.abs(rel) < 0.3)  # the whole flight came down, smoothly
    assert abs(pg["flight_z_m"][1] - pg["flight_z_m"][0] + 5.0) < 0.3

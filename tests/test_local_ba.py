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

"""Multi-video projects: per-source offset recovery on a synthetic scene, and ingest of two
pieces of the sample clip into one project (skipped without the sample clip in data/)."""

import json
import os

import numpy as np
import pycolmap
import pytest

from modelforge.sfm import pairs, priors as priors_mod, refine, robust
from modelforge.sfm.priors import Prior, PriorSigmas
from tests.test_m3_robust import quat_to_rot
from tests.sample_clip import sample_video

VIDEO = sample_video()


def make_two_source_scene(tmp_path, offset, seed=0, n=40):
    """Synthetic database + priors: images of the first half are source 0 (exact priors up to
    noise), the second half source 1 with all prior centers shifted by `offset`."""
    pycolmap.set_random_seed(seed)
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs, o.num_cameras_per_rig, o.num_frames_per_rig, o.num_points3D = 1, 1, n, 400
    db_path = str(tmp_path / "database.db")
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.synthesize_dataset(o, db)
    db.close()
    rng = np.random.default_rng(seed)
    priors, gt = {}, {}
    for k, image_id in enumerate(sorted(rec.reg_image_ids())):
        img = rec.image(image_id)
        p = np.array(rec.frame(img.frame_id).rig_from_world.params)
        R = quat_to_rot(p[:4])
        C = -R.T @ p[4:]
        src = int(k >= n // 2)
        Cp = C + rng.normal(size=3) * 0.01 + (offset if src else 0.0)
        priors[img.name] = Prior.from_pose(img.name, k, float(k), R, Cp, np.zeros(3), source=src, order=k)
        gt[img.name] = C
    os.makedirs(tmp_path / "images", exist_ok=True)
    return db_path, str(tmp_path / "images"), priors, gt


def test_source_offset_is_recovered(tmp_path):
    offset = np.array([0.6, -0.4, 0.3])  # cameras are ~5 units from the points
    db_path, image_path, pri, gt = make_two_source_scene(tmp_path, offset)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.05, 0.05, 0.2, 0.2), calibrate=False,
                              estimate_systematic=False)
    rec = refine.init_from_database(db_path, pri)
    methods = []
    for i in range(3):
        rec = refine.triangulate(rec, db_path, image_path, str(tmp_path / f"round_{i}"),
                                 refine.Round(4.0, 1.0, 2.0))
        ba = refine.prior_ba(rec, robust.corrected_priors(pri, model), model, refine_intrinsics=False)
        assert ba["usable"]
        rep = robust.update_source_offsets(model, rec, pri, db_path)
        methods.append(rep["method"])
        print(i, rep["method"], rep["offsets_m"], {k: (e["n_split"], e["merged"]) for k, e in rep["edges"].items()})
    est = model.source_offsets[1]
    print(f"offset est {np.round(est, 3)} true {-offset}")
    assert methods[0] == "split_points"  # the videos first triangulate separate copies of the scene
    assert np.linalg.norm(est + offset) < 0.03
    edges = rep["edges"]["0-1"]
    assert edges["merged"] > 10 * edges["n_split"]  # the cameras moved with the offset, so the copies merged
    assert ba["num_rel_priors"] <= len(pri) - 2  # never across the two sources
    pos = [np.linalg.norm(refine.center_of(np.array(rec.frame(rec.image(i).frame_id).rig_from_world.params))
                          - gt[rec.image(i).name]) for i in rec.reg_image_ids()]
    assert np.median(pos) < 0.03


def test_single_source_is_a_no_op(tmp_path):
    db_path, image_path, pri, _ = make_two_source_scene(tmp_path, np.zeros(3))
    one = {n: Prior.from_pose(p.name, p.frame, p.pts, p.R_cw, p.center, p.velocity) for n, p in pri.items()}
    model = robust.PriorModel()
    rec = refine.init_from_database(db_path, one)
    assert robust.update_source_offsets(model, rec, one, db_path) is None and not model.source_offsets


def test_pairs_and_relative_links_stay_within_sources():
    R = np.diag([1.0, -1.0, -1.0])  # looking down
    pri = {}
    for k in range(20):
        src = k // 10
        pri[f"v{src}_{k}"] = Prior.from_pose(f"v{src}_{k}", k % 10, float(k % 10), R, np.array([k * 1.0, 0, 30]),
                                             source=src, order=k)
    seq = pairs.select(pri, ground_z=0.0, seq_neighbors=3, spatial_neighbors=0)
    assert all(pri[a].source == pri[b].source for a, b in seq)
    both = pairs.select(pri, ground_z=0.0, seq_neighbors=3, spatial_neighbors=5)
    assert any(pri[a].source != pri[b].source for a, b in both)  # overlap still pairs the videos
    rel = robust.consecutive_pairs(pri, list(pri))
    assert len(rel) == 18 and all(pri[a].source == pri[b].source for a, b in rel)


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="needs the sample clip berlin_3_1.MP4 in data/")
def test_ingest_two_pieces(tmp_path):
    from modelforge.dji import ingest
    sources = [ingest.Source(os.path.relpath(VIDEO), 60.0, 90.0),
               ingest.Source(VIDEO, 10.0, 40.0)]
    settings = ingest.Settings(num_images=8)
    plan = ingest.make_plan(sources, settings)
    assert [sp.source.start for sp in plan.sources] == [10.0, 60.0]  # chronological
    assert plan.num_images == 8 and all(len(sp.windows) for sp in plan.sources)
    assert {sp.prefix for sp in plan.sources} == {"berlin_3_1"} and all(sp.alt_offset == 0 for sp in plan.sources)
    prev = ingest.preview(plan)
    assert len(prev["sources"]) == 2 and len(prev["sources"][0]["samples"]) == len(plan.sources[0].windows)

    project = str(tmp_path / "proj")
    out = ingest.run(project, sources, settings, log=lambda m: None)
    assert len(out["images"]) == 8 and len(out["sources"]) == 2
    names = sorted(os.listdir(os.path.join(project, "images")))
    assert len(names) == 8 and all(n.startswith("berlin_3_1_") for n in names)
    pri, meta = priors_mod.load(project)
    assert {p.source for p in pri.values()} == {0, 1} and "ground_z" in meta
    t = [pri[n].time for n in priors_mod.by_order(pri)]
    assert t == sorted(t)
    assert json.load(open(os.path.join(project, "ingest.json")))["settings"]["num_images"] == 8

    with pytest.raises(FileExistsError):
        ingest.run(project, sources, settings, log=lambda m: None)
    ingest.run(project, sources[:1], ingest.Settings(num_images=3), overwrite=True, log=lambda m: None)
    assert len(os.listdir(os.path.join(project, "images"))) == 3
    prev_dirs = os.listdir(os.path.join(project, "previous"))
    assert len(prev_dirs) == 1 and len(os.listdir(os.path.join(project, "previous", prev_dirs[0], "images"))) == 8


def make_drift_scene(tmp_path, drift, seed=0, n=40, gap=80.0):
    """Synthetic database + one source whose prior centers drift with time: C_prior = C + drift(t). Two passes
    over the same scene: images k < n/2 at t = k, the others `gap` seconds later (a revisit)."""
    pycolmap.set_random_seed(seed)
    o = pycolmap.SyntheticDatasetOptions()
    o.num_rigs, o.num_cameras_per_rig, o.num_frames_per_rig, o.num_points3D = 1, 1, n, 400
    db_path = str(tmp_path / "database.db")
    db = pycolmap.Database.open(db_path)
    rec = pycolmap.synthesize_dataset(o, db)
    db.close()
    rng = np.random.default_rng(seed)
    priors, gt = {}, {}
    for k, image_id in enumerate(sorted(rec.reg_image_ids())):
        img = rec.image(image_id)
        p = np.array(rec.frame(img.frame_id).rig_from_world.params)
        R = quat_to_rot(p[:4])
        C = -R.T @ p[4:]
        t = k + (gap if k >= n // 2 else 0.0)
        Cp = C + rng.normal(size=3) * 0.01 + drift(t)
        priors[img.name] = Prior.from_pose(img.name, k, t, R, Cp, np.zeros(3), source=0, order=k, time=t)
        gt[img.name] = C
    os.makedirs(tmp_path / "images", exist_ok=True)
    return db_path, str(tmp_path / "images"), priors, gt


def test_drift_is_recovered(tmp_path):
    """A slow drift of the priors (0.8 units in z over the flight, cameras ~5 units from the points) splits
    the revisit into an offset copy of the scene; update_drift measures it from the split points and the
    copies merge. The drift is known up to a constant (the mean stays at the priors)."""
    def drift(t):
        return np.array([0.3 * np.sin(t / 119 * np.pi), 0.0, 0.8 * t / 119])
    db_path, image_path, pri, gt = make_drift_scene(tmp_path, drift)
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.05, 0.05, 0.2, 0.2), calibrate=False,
                              estimate_systematic=False, drift_knot_s=10.0)
    rec = refine.init_from_database(db_path, pri)
    reps = []
    for i in range(4):
        rec = refine.triangulate(rec, db_path, image_path, str(tmp_path / f"round_{i}"), refine.Round(4.0, 1.0, 2.0))
        ba = refine.prior_ba(rec, robust.corrected_priors(pri, model), model, refine_intrinsics=False)
        assert ba["usable"]
        reps.append(robust.update_drift(model, rec, pri, db_path)["sources"]["0"])
        print(i, reps[-1])
    assert reps[0]["far_split_median_m"] > 0.2  # far-apart images first triangulate offset copies
    assert reps[-1]["increment_max_m"] < 0.02  # settled
    names = sorted(pri, key=lambda n: pri[n].order)
    resid = np.array([robust.drift_at(model, 0, pri[n].time) + drift(pri[n].time) for n in names])
    resid -= resid.mean(0)
    print("drift error after removing the mean:", np.abs(resid).max(0).round(3))
    assert np.abs(resid).max() < 0.1  # 0.8 units of drift; only the difference between the passes is measured
    pos = np.array([refine.center_of(np.array(rec.frame(rec.image(i).frame_id).rig_from_world.params))
                    - gt[rec.image(i).name] for i in rec.reg_image_ids()])
    assert np.median(np.linalg.norm(pos - np.median(pos, 0), axis=1)) < 0.03  # rigidly right, up to the gauge


def test_drift_without_revisits_stays_zero(tmp_path):
    db_path, image_path, pri, _ = make_drift_scene(tmp_path, lambda k: np.zeros(3))
    model = robust.PriorModel(abs_sigmas=PriorSigmas(0.05, 0.05, 0.2, 0.2), calibrate=False,
                              estimate_systematic=False, drift_knot_s=10.0)
    rec = refine.init_from_database(db_path, pri)
    rec = refine.triangulate(rec, db_path, image_path, str(tmp_path / "round_0"), refine.Round(4.0, 1.0, 2.0))
    refine.prior_ba(rec, robust.corrected_priors(pri, model), model, refine_intrinsics=False)
    robust.update_drift(model, rec, pri, db_path)
    assert np.abs(model.drift[0][1]).max() < 0.02
    model = robust.PriorModel(estimate_drift=False)
    assert robust.update_drift(model, rec, pri, db_path) is None and not model.drift

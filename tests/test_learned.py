"""Learned features + LightGlue, RoMa for hard pairs, covisible pairs between passes (sfm/learned.py, sfm/pairs.py;
the user asked, 2026-10-28)."""
import json
import os
import shutil
import sqlite3

import numpy as np
import pytest

from modelforge.sfm import learned, pairs

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


def test_snap_reuses_keypoints_and_shares_new_ones():
    existing = np.array([[10.0, 10.0], [100.0, 50.0], [300.2, 400.7]])
    new = np.array([[10.4, 9.8],     # within 1.5 px of keypoint 0
                    [100.0, 51.4],   # of keypoint 1
                    [500.1, 500.2],  # new
                    [500.2, 500.1],  # the same 0.5 px cell: the same new keypoint
                    [300.2, 403.0],  # 2.3 px from keypoint 2: new
                    [11.0, 11.0]])   # 1.41 px from keypoint 0
    idx, add = learned.snap(existing, new)
    assert list(idx[:2]) == [0, 1] and idx[5] == 0
    assert idx[2] == idx[3] >= 3 and idx[4] >= 3 and idx[4] != idx[2]
    assert len(add) == 2 and np.allclose(add[idx[2] - 3], [500.0, 500.0])
    idx, add = learned.snap(np.zeros((0, 2)), new[:2])
    assert list(idx) == [0, 1] and len(add) == 2


def _db(path, names):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, "
                "data BLOB, config INTEGER)")
    con.executemany("INSERT INTO images VALUES (?, ?)", [(k + 1, n) for k, n in enumerate(names)])
    return con


def test_dense_pairs_between_passes_with_a_budget(tmp_path):
    """Unlinked pairs between passes / flights only, best covisibility first, at most per_photo per photo."""
    names = [f"p{k}.jpg" for k in range(6)]
    times = [0, 10, 100, 110, 200, 205]
    json.dump({"images": [{"image": n, "time_s": t, "source": 0} for n, t in zip(names, times)]},
              open(tmp_path / "poses.json", "w"))
    con = _db(str(tmp_path / "database.db"), names)
    P = learned.PAIR_ID
    con.execute("INSERT INTO two_view_geometries VALUES (?, 100, 2, NULL, 3)", (1 * P + 3,))  # p0-p2 linked
    con.commit()
    con.close()
    pairs.write(str(tmp_path), [("p0.jpg", "p1.jpg"), ("p0.jpg", "p2.jpg"), ("p0.jpg", "p3.jpg"),
                                ("p0.jpg", "p4.jpg"), ("p1.jpg", "p4.jpg"), ("p4.jpg", "p5.jpg")])
    json.dump({"pairs": [["p0.jpg", "p4.jpg", 0.9], ["p1.jpg", "p4.jpg", 0.5]]},
              open(tmp_path / "pairs_covis.json", "w"))
    got, n = learned.dense_pairs(str(tmp_path), 15, per_photo=1)
    # p0-p1 / p4-p5 are close in time, p0-p2 is linked: candidates p0-p3, p0-p4, p1-p4
    assert n == 3 and got == [("p0.jpg", "p4.jpg")]  # best covisibility, then p0 and p4 are used up
    got, _ = learned.dense_pairs(str(tmp_path), 15, per_photo=2)
    assert set(got) == {("p0.jpg", "p4.jpg"), ("p1.jpg", "p4.jpg"), ("p0.jpg", "p3.jpg")}


def test_dense_blocks_are_reused_per_pair(tmp_path):
    """RoMa's finished pairs survive a killed or repeated run (blocks of several runs, a half-written one ignored);
    another match count starts over."""
    out = tmp_path / "roma"
    (out / "run_2").mkdir(parents=True)

    def block(path, ab):
        k = np.zeros((2 * len(ab), 2), np.float32)
        np.savez(path, a=np.array([a for a, _ in ab]), b=np.array([b for _, b in ab]),
                 offsets=np.arange(0, 2 * len(ab) + 1, 2), ka=k, kb=k, certainty=k[:, 0])
    block(out / "roma_00000.npz", [("p0", "p4"), ("p1", "p4")])
    block(out / "run_2" / "roma_00000.npz", [("p0", "p3")])
    (out / "run_2" / "roma_00001.npz.part.npz").write_bytes(b"half")
    (out / "meta.json").write_text('{"num": 2000}')
    assert learned._dense_done(str(out), 2000) == {("p0", "p4"), ("p1", "p4"), ("p0", "p3")}
    assert sum(len(a) for a, *_ in learned._dense_blocks(str(out))) == 3
    assert learned._dense_done(str(out), 500) == set() and not out.exists()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined")), reason="needs a refined project")
def test_ground_check_keeps_true_matches_only():
    """RoMa matches any two photos, so its matches must land on the same ground in the earlier model: the model's own
    observations of shared points do, the same points paired at random do not."""
    import pycolmap
    rec = pycolmap.Reconstruction(os.path.join(REAL, "sparse", "refined"))
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    a, b = ims[10], ims[12]
    obs = lambda im: {p.point3D_id: p.xy for p in im.points2D if p.has_point3D()}
    oa, ob = obs(a), obs(b)
    shared = sorted(set(oa) & set(ob))
    assert len(shared) > 100
    ka, kb = np.array([oa[k] for k in shared]), np.array([ob[k] for k in shared])
    g = learned.Ground(REAL, names={a.name, b.name})
    assert g.ok and set(g.photos) == {a.name, b.name}
    assert g.agree(a.name, b.name, ka, kb).mean() > 0.9
    wrong = np.random.default_rng(0).permutation(len(kb))
    assert g.agree(a.name, b.name, ka, kb[wrong], within=1.0).mean() < 0.2
    assert not learned.Ground(os.path.dirname(REAL)).ok  # no model there


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined")), reason="needs a refined project")
def test_covisible_pairs_from_the_model(tmp_path):
    from modelforge.sfm import priors
    pri, _ = priors.load(REAL)
    cv = pairs.covisible(REAL, pri, per_photo=3, min_gap_s=5.0)
    assert cv and all(o >= pairs.COVIS_MIN_OVERLAP for _, _, o in cv)
    assert all(abs(pri[a].time - pri[b].time) > 5.0 for a, b, _ in cv)
    assert pairs.covisible(REAL, pri, per_photo=0) == []
    assert pairs.covisible(str(tmp_path), pri) == []  # no model


@pytest.mark.skipif(learned.available() is not None, reason="scripts/install_matchers not run")
@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="needs a project")
def test_learned_features_through_refine(tmp_path):
    """ALIKED + LightGlue: features.db + database.db COLMAP reads, RoMa on forced pairs, refine, and photos located
    with the same features."""
    from modelforge import evaluate, pipeline
    p = tmp_path / "p"
    p.mkdir()
    shutil.copy(os.path.join(REAL, "poses.json"), p / "poses.json")
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    names = sorted(r["image"] for r in json.load(open(p / "poses.json"))["images"])[:30]
    poses = json.load(open(p / "poses.json"))
    poses["images"] = [r for r in poses["images"] if r["image"] in set(names)]
    json.dump(poses, open(p / "poses.json", "w"))
    rep = pipeline.Report(str(p))
    pipeline.stage_features(str(p), rep, extractor="aliked-lightglue")
    assert learned.method_of(str(p)) == "aliked-lightglue" and rep.data["features"]["images"] == 30
    from modelforge.sfm import priors
    pri, meta = priors.load(str(p))
    pipeline.stage_pairs(str(p), rep, pri, meta)
    pipeline.stage_match(str(p), rep)
    assert rep.data["match"]["verified_pairs"] > 0 and rep.data["match"]["method"] == "aliked-lightglue"
    learned.FAR_S, far = 0.0, learned.FAR_S
    real, calls = learned.verified, [0]

    def first_all_unlinked(db):  # dense_pairs of both densify calls below sees every pair unlinked
        calls[0] += 1
        return {} if calls[0] <= 2 else real(db)
    learned.verified = first_all_unlinked
    try:
        assert learned.densify(str(p), "roma", 15, per_photo=1)["skipped"]  # nothing to check RoMa's matches against
        os.makedirs(p / "sparse")
        os.symlink(os.path.abspath(os.path.join(REAL, "sparse", "refined")), p / "sparse" / "refined")  # earlier model
        d = learned.densify(str(p), "roma", 15, per_photo=1, num=500)
    finally:
        learned.FAR_S, learned.verified = far, real
    assert d["pairs"] >= 10 and d["linked"] == d["pairs"] - d["dropped"] >= 10 and d["keypoints_added"] > 0
    os.unlink(p / "sparse" / "refined")
    pipeline.stage_refine(str(p), rep, pri)
    final = rep.data["refine"]["final"]
    assert final["reg_images"] == 30 and final["point_error_px"]["median"] < 1.0
    loc = evaluate.locate(str(p), str(p / "images"), [{"image": names[5], "time_s": pri[names[5]].time + 0.01,
                                                     "source": 0}], str(tmp_path / "ws"))
    assert loc[names[5]]["inliers"] > 100
    fdir = learned.feature_dir(str(p), "aliked-lightglue")
    q = os.path.join(fdir, names[5] + ".npz")
    assert os.path.exists(q)  # the located photo went into the cache too
    # the cache: a second extraction takes every photo from it, except one whose stored stamp is not its photo's
    f = os.path.join(fdir, names[0] + ".npz")
    d = dict(np.load(f))
    d["stamp"] = d["stamp"] + 1
    np.savez(f[:-4], **d)
    pipeline.stage_features(str(p), rep, extractor="aliked-lightglue")
    assert rep.data["features"]["cached"] == 29 and learned.method_of(str(p)) == "aliked-lightglue"
    pipeline.stage_features(str(p), rep)  # SIFT again: the learned features are no longer features.db's
    assert learned.method_of(str(p)) is None


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="needs a project")
def test_auto_picks_learned_for_several_passes(tmp_path, monkeypatch):
    """Extractor / dense "auto": SIFT for one pass, ALIKED + RoMa when look-at pairs lie > 60 s apart (the same
    photos with every other one moved 1000 s later on the timeline), SIFT again without the matchers."""
    from modelforge import pipeline
    shutil.copy(os.path.join(REAL, "poses.json"), tmp_path / "poses.json")
    a = pipeline.auto_matching(str(tmp_path))
    assert not a["learned"] and a["far_pairs"] == 0
    poses = json.load(open(tmp_path / "poses.json"))
    for k, r in enumerate(poses["images"]):
        if k % 2:
            r["time_s"] += 1000.0
    json.dump(poses, open(tmp_path / "poses.json", "w"))
    monkeypatch.setattr(learned, "available", lambda: None)
    a = pipeline.auto_matching(str(tmp_path))
    assert a["learned"] and a["far_pairs"] >= pipeline.AUTO_MIN_FAR_PAIRS and "ALIKED" in a["reason"]
    monkeypatch.setattr(learned, "available", lambda: "not installed")
    a = pipeline.auto_matching(str(tmp_path))
    assert not a["learned"] and "not installed" in a["reason"]

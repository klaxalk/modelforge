"""Register more photos into the model (sfm/register.py): on fullrun_test with half its photos "in use" and all of
them in the pool, registering pool photos the refined model also holds must give back the model's own poses; the
export keeps them in transforms.json; Remove takes them out."""
import json
import os
import shutil

import numpy as np
import pytest

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "features.db")), reason="needs a refined project with features")
def test_registered_poses_match_the_model(tmp_path):
    import pycolmap
    from modelforge.sfm import register
    p = tmp_path / "proj"
    (p / "images").mkdir(parents=True)
    (p / "extracted" / "images").mkdir(parents=True)
    poses = json.load(open(os.path.join(REAL, "poses.json")))
    names = [r["image"] for r in poses["images"]]
    for k, n in enumerate(names):
        src = os.path.realpath(os.path.join(REAL, "images", n))
        os.symlink(src, p / "extracted" / "images" / n)  # (tmp is another file system: no hard links)
        if k % 2 == 0:
            os.symlink(src, p / "images" / n)
    json.dump(poses, open(p / "extracted" / "poses.json", "w"))
    sub = dict(poses, images=[r for k, r in enumerate(poses["images"]) if k % 2 == 0])
    json.dump(sub, open(p / "poses.json", "w"))
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("features.db", "report.json", "transforms.json"):
        shutil.copy2(os.path.join(REAL, f), p / f)
    t0 = json.load(open(p / "transforms.json"))
    st = register.run(str(p), every=4, on_line=lambda m: None)
    assert st["tried"] == 10 and st["registered"] >= 9 and st["error_px_median"] < 1.5
    rec = pycolmap.Reconstruction(str(p / "sparse" / "refined"))
    by_name = {rec.image(i).name: rec.image(i) for i in rec.reg_image_ids()}
    reg = register.load(str(p))["images"]
    for n, v in reg.items():  # the model holds these photos too: their own poses
        m = by_name[n].cam_from_world().matrix()
        R, C = m[:, :3], -m[:, :3].T @ m[:, 3]
        Rr = np.array(v["cam_from_world"])[:, :3]
        Cr = -Rr.T @ np.array(v["cam_from_world"])[:, 3]
        ang = np.degrees(np.arccos(np.clip((np.trace(R @ Rr.T) - 1) / 2, -1, 1)))
        assert np.linalg.norm(C - Cr) < 0.03 and ang < 0.05, (n, np.linalg.norm(C - Cr), ang)
        assert v["from_trajectory"]["m"] < register.MAX_SHIFT_M
    t = json.load(open(p / "transforms.json"))
    added = [f for f in t["frames"] if f.get("mf_registered")]
    assert len(added) == len(reg) and len(t["frames"]) == len(t0["frames"]) + len(reg)
    assert all(os.path.exists(p / f["file_path"]) for f in added)
    register.add_frames(str(p))  # again: not twice
    assert len(json.load(open(p / "transforms.json"))["frames"]) == len(t["frames"])
    register.remove(str(p))
    assert len(json.load(open(p / "transforms.json"))["frames"]) == len(t0["frames"])
    assert not (p / register.DIR).exists()


def test_candidates_from_a_selection(tmp_path):
    """The Images used choices over the whole pool (the user asked, 2026-10-25), minus the photos in use."""
    from modelforge.sfm import register
    p = tmp_path / "proj"
    (p / "images").mkdir(parents=True)
    (p / "extracted").mkdir()
    recs = [{"image": f"a_{k:03d}.jpg", "source": 0, "time_s": float(k)} for k in range(20)]
    json.dump({"images": recs}, open(p / "extracted" / "poses.json", "w"))
    for k in range(0, 20, 2):  # every second photo in use
        (p / "images" / recs[k]["image"]).touch()
    names = lambda rs: [r["image"] for r in rs]  # noqa: E731
    assert len(register.candidates(str(p))) == 10  # all unused
    assert names(register.candidates(str(p), selection={"every": 1})) == [f"a_{k:03d}.jpg" for k in range(1, 20, 2)]
    assert register.candidates(str(p), selection={"every": 2}) == []  # the same photos as in use
    assert names(register.candidates(str(p), selection={"every": 3, "offset": 1, "start": 0, "end": 10})) == \
        ["a_001.jpg", "a_007.jpg"]  # 1, 4, 7, 10 chosen; 4 and 10 in use
    assert names(register.candidates(str(p), selection={"names": ["a_002.jpg", "a_003.jpg", "zz.jpg"]})) == \
        ["a_003.jpg"]

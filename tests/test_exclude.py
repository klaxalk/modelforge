"""Photos the user excludes from the splat on (modelforge/exclude.py; the user asked, 2026-10-28)."""
import json
import os
import shutil

import numpy as np
import pytest

from modelforge import exclude

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


def test_names_by_pick_and_range(tmp_path):
    recs = [{"image": f"a_{k}.jpg", "source": 0, "time_s": float(k)} for k in range(10)] + \
           [{"image": f"b_{k}.jpg", "source": 1, "time_s": 100.0 + k} for k in range(10)]
    json.dump({"images": recs}, open(tmp_path / "poses.json", "w"))
    p = str(tmp_path)
    assert exclude.names(p) == set()
    assert exclude.toggle(p, "a_1.jpg") is True
    exclude.add_range(p, 5, 3, source=0)  # given backwards
    exclude.add_range(p, 103.0, 104.0)  # every source
    assert exclude.names(p) == {"a_1.jpg", "a_3.jpg", "a_4.jpg", "a_5.jpg", "b_3.jpg", "b_4.jpg"}
    exclude.add_range(p, 0, 200, source=1)
    assert len(exclude.names(p)) == 4 + 10
    assert exclude.undo(p)["source"] == 1
    assert exclude.toggle(p, "a_1.jpg") is False  # picked again: back in
    assert "a_1.jpg" not in exclude.names(p)
    exclude.clear(p)
    assert not (tmp_path / exclude.FILE).exists() and exclude.names(p) == set()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")), reason="needs a matched, refined project")
def test_export_and_registered_photos_leave_them_out(tmp_path):
    """The export leaves the excluded photos out of transforms.json and triangulates without them; registered photos
    inside an excluded range stay out too, and their neighbours are not taken for an isolated stretch's."""
    import pycolmap
    from modelforge import pipeline
    from modelforge.sfm import register
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("database.db", "poses.json"):
        shutil.copy(os.path.join(REAL, f), p / f)
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    recs = sorted(json.load(open(p / "poses.json"))["images"], key=lambda r: (r.get("source", 0), r["time_s"]))
    picked, ranged = recs[3]["image"], recs[20:26]
    exclude.toggle(str(p), picked)
    exclude.add_range(str(p), ranged[0]["time_s"], ranged[-1]["time_s"], source=ranged[0].get("source", 0))
    drop = {picked} | {r["image"] for r in ranged}
    assert exclude.names(str(p)) == drop
    pipeline.stage_export(str(p), pipeline.Report(str(p)), drop_isolated=False)
    t = json.load(open(p / "transforms.json"))
    assert len(t["frames"]) == len(recs) - len(drop)
    assert not {os.path.basename(f["file_path"]) for f in t["frames"]} & drop
    rec = pycolmap.Reconstruction(str(p / "sparse" / "export"))
    seen = {rec.image(e.image_id).name for pt in rec.points3D.values() for e in pt.track.elements}
    assert rec.num_points3D() > 1000 and not seen & drop
    rep = json.load(open(p / "report.json"))["export"]["excluded"]
    assert rep["photos"] == len(drop) and rep["triangulated"] and rep["entries"] == exclude.load(str(p))

    # registered photos: one inside the range (excluded), its neighbours outside it (kept)
    cam = next(iter(rec.cameras))
    mid = 0.5 * (ranged[2]["time_s"] + ranged[3]["time_s"])
    imgs = {f"reg_{k}.jpg": {"cam_from_world": np.eye(4)[:3].tolist(), "camera_id": cam, "inliers": 100,
                             "source": ranged[0].get("source", 0), "frame": 0, "time_s": ts}
            for k, ts in enumerate((mid, 0.5 * (recs[30]["time_s"] + recs[31]["time_s"])))}
    (p / register.DIR).mkdir()
    json.dump({"model": register._model_key(str(p)), "stats": {}, "images": imgs},
              open(p / register.DIR / "registered.json", "w"))
    register.add_frames(str(p))
    out = json.load(open(p / "transforms.json"))
    added = {os.path.basename(f["file_path"]) for f in out["frames"] if f.get("mf_registered")}
    assert added == {"reg_1.jpg"} and out["modelforge"]["registered_excluded"] == ["reg_0.jpg"]
    assert out["modelforge"]["registered_near_dropped"] == []  # user exclusions are no ghost copy

"""Per-photo exposure from the djmd camera records (modelforge/exposure.py)."""
import json
import os
import shutil

import numpy as np
import pytest

from modelforge import exposure

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", "data/fullrun_test")


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="no test project")
def test_gains_from_the_metadata(tmp_path):
    p = str(tmp_path / "proj")
    os.makedirs(p)
    shutil.copy(os.path.join(REAL, "poses.json"), p)
    e = exposure.per_image(p)
    poses = json.load(open(os.path.join(p, "poses.json")))
    assert set(e) == {r["image"] for r in poses["images"]}
    g = np.array([x["gain"] for x in e.values()])
    assert abs(np.median(g) - 1) < 1e-3 and (g > 0).all()
    x = next(iter(e.values()))
    rel = x["shutter_s"] * x["iso"] / x["fnum"] ** 2
    ref = np.median([v["shutter_s"] * v["iso"] / v["fnum"] ** 2 for v in e.values()])
    assert abs(x["gain"] - rel / ref) < 1e-4
    assert os.path.exists(os.path.join(p, exposure.CACHE)) and exposure.per_image(p) == e  # cached


def test_no_poses(tmp_path):
    assert exposure.per_image(str(tmp_path)) == {}

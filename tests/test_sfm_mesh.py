"""sfm_mesh: Poisson over the SfM points, trimmed to them; listed as View mesh "sfm"."""
import json
import os
import shutil

import numpy as np
import plyfile
import pytest

from modelforge import sfm_mesh
from modelforge.gui import data as gdata

REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "points3D.bin")),
                    reason="needs a refined project")
def test_sfm_mesh_lies_on_the_points(tmp_path):
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), tmp_path / "sparse" / "refined")
    st = sfm_mesh.run(str(tmp_path), on_line=lambda m: None)
    assert st["faces"] > 1000 and st["faces"] < st["faces_untrimmed"]  # Poisson's closure beyond the points is cut
    assert st["cell"] < 4 * st["point_spacing"]  # no stray points stretched the octree
    v = plyfile.PlyData.read(tmp_path / "sfmmesh" / "mesh.ply")["vertex"].data
    p = plyfile.PlyData.read(tmp_path / "sfmmesh" / "points.ply")["vertex"].data
    vx, px = np.c_[v["x"], v["y"], v["z"]], np.c_[p["x"], p["y"], p["z"]]
    import cv2
    c = px.mean(0)
    _, d2 = cv2.flann_Index((px - c).astype(np.float32), dict(algorithm=1, trees=4)).knnSearch(
        (vx - c).astype(np.float32), 1, params={"checks": 64})
    assert np.percentile(np.sqrt(d2[:, 0]), 99) <= st["trim_dist"] * 1.01  # nothing far from the SfM points
    assert "red" in v.dtype.names  # vertex colours
    assert (p["nz"] > 0.5).mean() > 0.5  # a drone survey: mostly ground, normals up towards the cameras
    assert gdata.dense_artifacts(str(tmp_path))["meshes"]["sfm"].endswith(".ply")
    assert json.load(open(tmp_path / "report.json"))["sfm_mesh"]["faces"] == st["faces"]

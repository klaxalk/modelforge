"""sfm_mesh: Poisson over the SfM points, trimmed to them; listed as View mesh "sfm"."""
import json
import os
import shutil

import numpy as np
import plyfile
import pytest

from modelforge import sfm_mesh, splat
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
    # patching holes: enclosed pieces of the cut Poisson surface come back, the outside stays cut
    r = sfm_mesh.fill(str(tmp_path), on_line=lambda m: None)
    assert r["holes"] > 0 and r["faces"] == st["faces"] + r["faces_added"] < st["faces_untrimmed"]
    assert r["largest_patched"] <= r["max_hole"]
    patched = np.load(tmp_path / "sfmmesh" / "patched.npy")
    assert patched.sum() == r["faces_added"] and len(patched) == r["faces"]
    assert (tmp_path / "sfmmesh" / "mesh_trimmed.ply").exists()
    sfm_mesh.run(str(tmp_path), on_line=lambda m: None)  # a new mesh drops the old patch
    assert not (tmp_path / "sfmmesh" / "patched.npy").exists()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "points3D.bin")),
                    reason="needs a refined project")
def test_delaunay_sfm_mesh(tmp_path):
    """Mesh type delaunay: COLMAP's sparse Delaunay mesher, its vertices the SfM points (coloured like them), no
    patching."""
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), tmp_path / "sparse" / "refined")
    st = sfm_mesh.run(str(tmp_path), method="delaunay", on_line=lambda m: None)
    assert st["method"] == "delaunay" == sfm_mesh.method(str(tmp_path)) and st["faces"] > 1000
    v = plyfile.PlyData.read(tmp_path / "sfmmesh" / "mesh.ply")["vertex"].data
    assert "red" in v.dtype.names and np.ptp(v["red"]) > 0  # the points' colours
    assert not (tmp_path / "sfmmesh" / "kept.npy").exists() and not sfm_mesh.can_fill(str(tmp_path))
    with pytest.raises(RuntimeError, match="Delaunay"):
        sfm_mesh.fill(str(tmp_path), on_line=lambda m: None)
    assert sfm_mesh.run(str(tmp_path), on_line=lambda m: None)["method"] == "poisson"  # back: patchable again
    assert sfm_mesh.can_fill(str(tmp_path)) and not (tmp_path / "sfmmesh" / "delaunay.ply").exists()


def test_components_split_by_shared_edges():
    # two triangles sharing an edge, a third touching them only at a vertex, a fourth masked out
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [5, 6, 7]])
    lab = sfm_mesh.components(faces, np.array([True, True, True, False]))
    assert lab[0] == lab[1] != lab[2] and lab[3] == -1


def test_mesh_prior_in_the_train_command():
    from modelforge import splat
    cmd = splat.train_command("/p", lidar_depth=0.5, depth_cloud="/p/sfmmesh/mesh.ply")
    k = cmd.index("lidar-data")
    assert cmd[0].endswith("ns-train") and cmd[1] == "splatfacto-lidar"
    assert cmd[k:].count("--depth-ply") == 1 and cmd[cmd.index("--depth-ply") + 1] == "/p/sfmmesh/mesh.ply"
    assert "--depth-ply" not in splat.train_command("/p", depth_cloud="/p/m.ply")  # no depth loss, no prior
    cmd = splat.train_command("/p", lidar_depth=1.0, depth_cloud="/p/m.ply", depth_maps="/p/sfmmesh/depth_x2")
    assert "--depth-ply" not in cmd and cmd[cmd.index("--depth-maps") + 1] == "/p/sfmmesh/depth_x2"
    assert cmd.index("--depth-maps") > cmd.index("lidar-data")  # a dataparser argument


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")), reason="needs an exported project")
def test_gui_ticks_the_mesh_prior_when_a_mesh_appears(tmp_path):
    from modelforge.gui.app import App
    from tests.test_projects import free_port
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("poses.json", "transforms.json", "sparse_pc.ply"):
        shutil.copy(os.path.join(REAL, f), p / f)
    if os.path.exists(os.path.join(REAL, "ingest.json")):
        shutil.copy(os.path.join(REAL, "ingest.json"), p / "ingest.json")
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    app = App(str(p), port=free_port(), root=str(tmp_path))
    try:
        assert not app.g_splat_mesh_prior.value and app.g_splat_mesh_prior.disabled
        assert app.g_splat_depth.value == 0.0  # no mesh: no depth loss by default
        sfm_mesh.run(str(p), on_line=lambda m: None)
        app._refresh_dense()
        assert app.g_splat_mesh_prior.value and not app.g_splat_mesh_prior.disabled
        assert app.g_splat_depth.value == splat.DEFAULT_LIDAR_DEPTH  # the prior's default weight (0.1)
        assert "sfm" in app.g_mesh.options
        app.g_splat_mesh_prior.value = False  # the user's choice holds while the mesh stays
        app._refresh_dense()
        assert not app.g_splat_mesh_prior.value
    finally:
        app.server.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")) or
                    not os.path.exists("/opt/nerfstudio/bin/python"), reason="needs an exported project + nerfstudio")
def test_mesh_depth_maps_line_up_with_nerfstudios_cameras(tmp_path):
    """The SfM mesh rendered at every training camera (export/mesh_depth.py) matches the photo's own SfM points
    projected with nerfstudio's undistorted, cropped camera (×2: the plan's default on 4K) to a fraction of a percent."""
    import subprocess

    import pycolmap

    from modelforge import splat
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("transforms.json", "sparse_pc.ply"):
        shutil.copy(os.path.join(REAL, f), p / f)
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    sfm_mesh.run(str(p), on_line=lambda m: None)
    t = json.load(open(p / "transforms.json"))
    t["frames"] = t["frames"][::10]  # a few cameras are enough
    json.dump(t, open(p / "transforms.json", "w"))
    splat.prepare_downscaled(str(p), 2, on_line=lambda m: None)
    maps = splat.mesh_depth_maps(str(p), str(p / "sfmmesh" / "mesh.ply"), 2, on_line=lambda m: None)
    rec = pycolmap.Reconstruction(str(p / "sparse" / "refined"))
    obs = {}
    for i in rec.reg_image_ids():
        im = rec.image(i)
        obs[im.name] = np.array([rec.point3D(q.point3D_id).xyz for q in im.points2D if q.has_point3D()])
    np.savez(tmp_path / "observed.npz", **obs)
    code = f"""
import numpy as np
from pathlib import Path
from mf_lidar_dataparser import LidarDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import undistort_view
out = LidarDataParserConfig(data=Path("{p}"), depth_maps=Path("{maps}"), downscale_factor=2).setup() \\
    .get_dataparser_outputs("train")
ds, OBS, rel = InputDataset(out), np.load("{tmp_path / 'observed.npz'}"), []
T = out.dataparser_transform.numpy()
for i in range(len(ds)):
    cam, data = undistort_view(i, ds)
    cam, (H, W) = cam[0], data["image"].shape[:2]
    K = cam.get_intrinsics_matrices().numpy()
    c2w = np.eye(4); c2w[:3] = cam.camera_to_worlds.numpy(); c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    X = (OBS[Path(out.image_filenames[i]).name] @ T[:, :3].T + T[:, 3]) * out.dataparser_scale
    X = X @ w2c[:3, :3].T + w2c[:3, 3]
    X = X[X[:, 2] > 1e-3]
    u, v = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.load(out.metadata["depth_files"][i]).astype(np.float32) * out.metadata["depth_scale"]
    assert m.shape == (H, W), (m.shape, H, W)
    z = m[v[ok].astype(int), u[ok].astype(int)]
    rel.append(np.abs(z[z > 0] - X[ok][z > 0, 2]) / X[ok][z > 0, 2])
r = np.concatenate(rel)
print("median", np.median(r), "p90", np.percentile(r, 90))
assert np.median(r) < 0.005 and np.percentile(r, 90) < 0.03
print("ok")
"""
    r = subprocess.run([os.path.join(splat.NS_BIN, "python"), "-c", code], env=splat._env(),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "ok" in r.stdout, (r.stdout + r.stderr)[-2000:]


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")) or
                    not os.path.exists("/opt/nerfstudio/bin/python"), reason="needs an exported project + nerfstudio")
def test_colour_from_the_photos(tmp_path):
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    shutil.copy(os.path.join(REAL, "transforms.json"), p / "transforms.json")
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    sfm_mesh.run(str(p), on_line=lambda m: None)
    r = sfm_mesh.colour(str(p), width=960, on_line=lambda m: None)
    assert r["coloured"] > 0.8 * r["vertices"]
    v0 = plyfile.PlyData.read(p / "sfmmesh" / "mesh.ply")["vertex"].data
    v1 = plyfile.PlyData.read(p / "sfmmesh" / "mesh_coloured.ply")["vertex"].data
    assert len(v0) == len(v1) and np.allclose(v0["x"], v1["x"])  # the same mesh, other colours
    assert np.mean(v0["red"] != v1["red"]) > 0.5
    assert "coloured" in gdata.dense_artifacts(str(p))["meshes"]["sfm"]
    sfm_mesh.fill(str(p), on_line=lambda m: None)  # a patch drops the colours (patched faces have none)
    assert "coloured" not in gdata.dense_artifacts(str(p))["meshes"]["sfm"]


def test_mesh_uses_the_export_model_only_while_it_is_current(tmp_path):
    for name in ("refined", "export"):
        (tmp_path / "sparse" / name).mkdir(parents=True)
    ref, exp = tmp_path / "sparse" / "refined" / "points3D.bin", tmp_path / "sparse" / "export" / "points3D.bin"
    ref.write_bytes(b"")
    exp.write_bytes(b"")
    os.utime(ref, (1000, 1000))
    os.utime(exp, (2000, 2000))
    assert sfm_mesh.model_dir(str(tmp_path)).endswith("export")  # the export after the refine
    os.utime(ref, (3000, 3000))  # refine ran again since
    assert sfm_mesh.model_dir(str(tmp_path)).endswith("refined")


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")), reason="needs an exported project")
def test_view_opens_on_the_export_model_while_current(tmp_path):
    from modelforge.gui.app import App
    from tests.test_projects import free_port
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "export")
    for f in ("poses.json", "transforms.json", "sparse_pc.ply"):
        shutil.copy(os.path.join(REAL, f), p / f)
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    for f in os.listdir(p / "sparse" / "refined"):
        os.utime(p / "sparse" / "refined" / f, (1000, 1000))
    app = App(str(p), port=free_port(), root=str(tmp_path))
    try:
        assert app.snap.name == "export"
        for f in os.listdir(p / "sparse" / "refined"):  # refine ran again since: refined is the current model
            os.utime(p / "sparse" / "refined" / f, None)
        os.utime(p / "sparse" / "export" / "points3D.bin", (1000, 1000))
        app.open_project(str(p))
        assert app.snap.name == "refined"
    finally:
        app.server.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")) or
                    not os.path.exists("/opt/nerfstudio/bin/python"), reason="needs an exported project + nerfstudio")
def test_colour_live_and_kept_when_stopped(tmp_path):
    from modelforge import jobs
    from modelforge.gui.app import App
    from tests.test_projects import free_port
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("poses.json", "transforms.json", "sparse_pc.ply"):
        shutil.copy(os.path.join(REAL, f), p / f)
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    sfm_mesh.run(str(p), on_line=lambda m: None)
    n_v = plyfile.PlyData.read(p / "sfmmesh" / "mesh.ply")["vertex"].count
    app = App(str(p), port=free_port(), root=str(tmp_path))
    try:
        app.g_mesh.value = "sfm"
        app._colour_live_mesh, app._colour_live_t = None, 0.0
        token, lives = jobs.Token(), []

        def on_live(path, k, n):
            rgb = np.load(path)
            assert rgb.shape == (n_v, 3) and rgb.dtype == np.uint8
            lives.append(k)
            app._on_colour_live(path, k, n, every_s=0)  # the GUI recolours the shown mesh
            if len(lives) == 2:
                token.cancel()  # Stop after the second snapshot
        with pytest.raises(jobs.Cancelled):
            sfm_mesh.colour(str(p), width=480, token=token, on_live=on_live, every=20, on_line=lambda m: None)
        assert lives[:2] == [20, 40]  # a snapshot already on its way when Stop came may follow
        assert f"colouring, {lives[-1]} of" in app._dense_lines["mesh"]
        rep = json.load(open(p / "sfmmesh" / "mesh_coloured.ply.json"))
        assert rep["partial"] and rep["photos_done"] == lives[-1]  # the colours saved last were kept
        assert plyfile.PlyData.read(p / "sfmmesh" / "mesh_coloured.ply")["vertex"].count == n_v
        assert not (p / "sfmmesh" / sfm_mesh.LIVE).exists()
    finally:
        app.server.stop()


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "points3D.bin")),
                    reason="needs a refined project")
def test_preview_keeps_the_colours(tmp_path):
    """COLMAP's mesh_simplifier writes positions only: the preview takes the colours of the nearest mesh vertex."""
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), tmp_path / "sparse" / "refined")
    st = sfm_mesh.run(str(tmp_path), on_line=lambda m: None)
    out = str(tmp_path / "sfmmesh")
    sfm_mesh.write_preview(out, os.path.join(out, "mesh.ply"), st["faces"], None, name="small.ply", faces=5000)
    v = plyfile.PlyData.read(os.path.join(out, "small.ply"))["vertex"].data
    src = plyfile.PlyData.read(os.path.join(out, "mesh.ply"))["vertex"].data
    assert "red" in v.dtype.names and v["red"].std() > 5  # colours, not one grey
    assert abs(float(v["green"].mean()) - float(src["green"].mean())) < 15


def test_seed_points_fill_where_sfm_points_are_thin(tmp_path):
    """Mesh seeds: on the surface, only where no SfM point is near, spread at the spacing, normals along the
    surface's, colours from the mesh."""
    import plyfile
    from modelforge import sfm_mesh
    n = 21  # a 20 x 20 m flat ground at z = 2, two triangles per 1 m cell
    gx, gy = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    v = np.zeros(n * n, [(c, "f4") for c in "xyz"] + [(c, "u1") for c in ("red", "green", "blue")])
    v["x"], v["y"], v["z"], v["red"], v["green"], v["blue"] = gx.ravel(), gy.ravel(), 2.0, 200, 100, 50
    faces = [(i * n + j, (i + 1) * n + j, i * n + j + 1) for i in range(n - 1) for j in range(n - 1)] + \
        [(i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1) for i in range(n - 1) for j in range(n - 1)]
    os.makedirs(tmp_path / "sfmmesh")
    sfm_mesh.write_mesh(str(tmp_path / "sfmmesh" / "mesh.ply"), v, np.array(faces))
    json.dump({"point_spacing": 0.25}, open(tmp_path / "sfmmesh" / "stats.json", "w"))
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(0, 10, 4000), rng.uniform(0, 20, 4000), np.full(4000, 2.0)]  # on the half x < 10 only
    pc = np.zeros(len(pts), [(c, "f4") for c in "xyz"])
    pc["x"], pc["y"], pc["z"] = pts.T
    edge = np.zeros(100, pc.dtype)  # a sparse row along the far edge: the box covers the whole ground
    edge["x"], edge["y"], edge["z"] = 20.0, np.linspace(0, 20, 100), 2.0
    pc = np.concatenate([pc, edge])
    plyfile.PlyData([plyfile.PlyElement.describe(pc, "vertex")]).write(str(tmp_path / "sparse_pc.ply"))
    out = str(tmp_path / "seeds.ply")
    lines = []
    m = sfm_mesh.seed_points(str(tmp_path / "sfmmesh" / "mesh.ply"), str(tmp_path / "sparse_pc.ply"), out,
                             on_line=lines.append)
    s = plyfile.PlyData.read(out)["vertex"].data
    assert m == len(s) and "starting Gaussians" in lines[0]
    assert s["x"].min() > 10 - 1e-6 + 2.5 * 0.25 - 0.3  # none where the SfM points are (gap 2.5 spacings)
    free = (20 - 10 - 2 * 2.5 * 0.25) * 20  # between the covered half and the edge row, minus the gaps
    assert abs(len(s) - free / 0.25 ** 2) < 0.15 * free / 0.25 ** 2  # ~the free area / spacing^2
    assert np.allclose(np.abs(s["nz"]), 1) and np.allclose(s["z"], 2.0, atol=1e-5)
    assert (s["red"] == 200).all() and (s["blue"] == 50).all()

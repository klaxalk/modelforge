"""Export tab back end: mesh (glb / obj / ply) and splat (ply / .splat) exports, splat training options."""

import json
import os

import numpy as np
import plyfile
import pytest
import trimesh
from PIL import Image

from modelforge import dense, splat
from modelforge.export import products
from tests.test_dense_chain import project  # noqa: F401  (synthetic dense workspace fixture)
from tests.test_splat_to_enu import make_splat


@pytest.fixture(scope="module")
def textured(project):  # noqa: F811
    dense.fusion(project)
    dense.mesh(project, "delaunay")
    dense.texture(project, "delaunay")
    with open(os.path.join(project, "poses.json"), "w") as f:
        json.dump({"enu_origin_lla": [52.6, 13.15, 12.4]}, f)
    return project


def test_available(textured):
    have = products.available(textured)
    assert have["meshes"]["delaunay"] == {"textured": True, "plain": True} and not have["splat"]


@pytest.mark.parametrize("fmt", ["glb", "obj"])
def test_mesh_yup_with_uvs(textured, fmt):
    paths = products.export_mesh(textured, "delaunay", fmt)
    src = products.load_textured(os.path.join(dense.workspace(textured), "textured-delaunay", "mesh.ply"))
    out = trimesh.load(paths[0], force="mesh")
    assert len(out.faces) == len(src.faces)
    assert out.visual.uv is not None and len(out.visual.uv) == len(out.vertices)
    # Y-up: (east, north, up) -> (east, up, -north)
    assert np.allclose(np.sort(out.vertices[:, 1]), np.sort(src.vertices[:, 2]), atol=1e-4)
    assert np.allclose(np.sort(out.vertices[:, 2]), np.sort(-src.vertices[:, 1]), atol=1e-4)
    if fmt == "obj":
        names = sorted(os.path.basename(p) for p in paths)
        assert any(n.endswith(".png") and "mesh-delaunay" in n for n in names)  # renamed, no material_0.png
        assert "Kd 1.0" in open(paths[0].replace(".obj", ".mtl")).read()
    g = json.load(open(os.path.join(products.out_dir(textured), "georef.json")))
    assert g["enu_origin_lla"] == [52.6, 13.15, 12.4]
    assert g["files"][os.path.basename(paths[0])]["axes"] == "Y-up"


def test_mesh_ply_keeps_colmap_files(textured):
    paths = products.export_mesh(textured, "delaunay", "ply")
    assert paths[0].endswith("mesh.ply") and any(p.endswith(".png") for p in paths)
    paths = products.export_mesh(textured, "delaunay", "ply", textured=False)
    assert paths == [os.path.join(products.out_dir(textured), f"{os.path.basename(textured)}_mesh-delaunay.ply")]


def test_splat_bytes_roundtrip():
    v, _ = make_splat(np.random.default_rng(1), n=300)
    rec = np.frombuffer(products.to_splat_bytes(v), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4),
                                                            ("r", "u1", 4)])
    assert len(rec) == 300
    alpha = 1 / (1 + np.exp(-v["opacity"]))
    scale = np.exp(np.c_[v["scale_0"], v["scale_1"], v["scale_2"]])
    order = np.argsort(-(alpha * scale.prod(1)))
    assert np.allclose(rec["p"], np.c_[v["x"], v["y"], v["z"]][order], atol=1e-6)
    assert np.allclose(rec["s"], scale[order], rtol=1e-5)
    assert np.allclose(rec["c"][:, 3], alpha[order] * 255, atol=1)
    q = np.c_[v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]][order]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    assert np.allclose((rec["r"].astype(float) - 128) / 128, q, atol=1 / 64)


def test_export_splat(tmp_path):
    p = str(tmp_path)
    v, _ = make_splat(np.random.default_rng(2), n=50)
    os.makedirs(os.path.join(p, "splat"))
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(os.path.join(p, "splat", "splat_enu.ply"))
    ply = products.export_splat(p, "ply")[0]
    assert plyfile.PlyData.read(ply)["vertex"].count == 50
    assert os.path.getsize(products.export_splat(p, "splat")[0]) == 50 * 32


def test_train_command_cache_and_downscale():
    cmd = splat.train_command("/p", iterations=100)
    assert cmd[cmd.index("--pipeline.datamanager.cache-images") + 1] == "disk"  # default
    assert cmd[-3:] == ["nerfstudio-data", "--downscale-factor", "1"]  # explicit even at 1
    big = splat.train_command("/p", method="splatfacto-big")
    assert big[big.index("--method-name") + 1] == "splatfacto-big"  # its own run folder
    cmd = splat.train_command("/p", cache="gpu", downscale=2, extra=["--foo", "1"])
    assert cmd[cmd.index("--pipeline.datamanager.cache-images") + 1] == "gpu"
    assert cmd[-5:] == ["--foo", "1", "nerfstudio-data", "--downscale-factor", "2"]  # dataparser last
    with pytest.raises(ValueError):
        splat.train_command("/p", cache="ram")


def test_prepare_downscaled(tmp_path):
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "images"))
    for i in range(2):
        Image.new("RGB", (103, 61), (i * 100, 50, 0)).save(os.path.join(p, "images", f"f{i}.jpg"))
    with open(os.path.join(p, "transforms.json"), "w") as f:
        json.dump({"frames": [{"file_path": f"images/f{i}.jpg"} for i in range(2)]}, f)
    lines = []
    assert splat.prepare_downscaled(p, 1) is None
    out = splat.prepare_downscaled(p, 2, lines.append)
    assert sorted(os.listdir(out)) == ["f0.jpg", "f1.jpg"]
    assert Image.open(os.path.join(out, "f0.jpg")).size == (51, 30)  # floor, as nerfstudio rescales
    splat.prepare_downscaled(p, 2, lines.append)
    assert "0 of 2" in lines[-1]  # up to date: nothing rewritten


def test_prepare_downscaled_progress_and_stop(tmp_path):
    from modelforge import jobs
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "images"))
    for i in range(6):
        Image.new("RGB", (160, 96), (i * 40, 50, 0)).save(os.path.join(p, "images", f"f{i}.jpg"))
    with open(os.path.join(p, "transforms.json"), "w") as f:
        json.dump({"frames": [{"file_path": f"images/f{i}.jpg"} for i in range(6)]}, f)
    counts = []
    out = splat.prepare_downscaled(p, 4, lambda m: None, on_count=lambda k, n: counts.append((k, n)))
    assert counts[-1] == (6, 6) and Image.open(os.path.join(out, "f3.jpg")).size == (40, 24)
    assert not [n for n in os.listdir(out) if ".part" in n]
    token = jobs.Token()
    token.cancel()
    with pytest.raises(jobs.Cancelled):
        splat.prepare_downscaled(p, 2, lambda m: None, job=token)

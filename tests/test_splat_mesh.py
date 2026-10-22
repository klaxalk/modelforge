"""Mesh from the splat (modelforge.splat_mesh + export/splat_tsdf.py): a synthetic splat of flat, opaque
Gaussians tiling the ground and a box, seen by cameras looking down; the fused mesh must lie on those
surfaces with their colour. Needs the CUDA image (gsplat + Open3D in nerfstudio's venv)."""

import json
import os

import numpy as np
import pytest

from modelforge import splat, splat_mesh
from modelforge.export import products
from modelforge.gui import data as gdata

SH_C0 = 0.28209479177387814


def write_splat(path, pts, normals, rgb, size):
    """Disc-like Gaussians (flat along their normal) at pts, as splat_to_enu writes them (degree 0)."""
    from plyfile import PlyData, PlyElement
    n = len(pts)
    z = np.array([0.0, 0.0, 1.0])
    quats = []
    for nv in normals:  # rotation taking z to the normal (wxyz)
        axis = np.cross(z, nv)
        s, c = np.linalg.norm(axis), float(np.dot(z, nv))
        if s < 1e-9:
            quats.append((1.0, 0, 0, 0) if c > 0 else (0.0, 1, 0, 0))
            continue
        ang = np.arctan2(s, c)
        a = axis / s * np.sin(ang / 2)
        quats.append((np.cos(ang / 2), *a))
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    v = np.zeros(n, dtype=[(k, "f4") for k in names])
    v["x"], v["y"], v["z"] = pts.T
    v["scale_0"] = v["scale_1"] = np.log(size)
    v["scale_2"] = np.log(size * 0.05)
    for i in range(4):
        v[f"rot_{i}"] = np.array(quats)[:, i]
    for i in range(3):
        v[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / SH_C0
    v["opacity"] = 8.0  # sigmoid ~ 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def scene(project, step=0.05):
    g = np.arange(-6, 6 + 1e-9, step)
    X, Y = np.meshgrid(g, g)
    ground = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    keep = ~((np.abs(ground[:, 0]) < 1) & (np.abs(ground[:, 1]) < 1))  # under the box
    ground = ground[keep]
    b = np.arange(-1, 1 + 1e-9, step)
    BX, BY = np.meshgrid(b, b)
    top = np.c_[BX.ravel(), BY.ravel(), np.full(BX.size, 1.0)]
    h = np.arange(step / 2, 1.0, step)
    sides, side_n = [], []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        U, Hh = np.meshgrid(b, h)
        p = np.zeros((U.size, 3))
        p[:, axis], p[:, 1 - axis], p[:, 2] = sign * 1.0, U.ravel(), Hh.ravel()
        sides.append(p)
        nv = np.zeros(3)
        nv[axis] = sign
        side_n.append(np.tile(nv, (U.size, 1)))
    pts = np.concatenate([ground, top] + sides)
    normals = np.concatenate([np.tile([0, 0, 1.0], (len(ground) + len(top), 1))] + side_n)
    rgb = np.concatenate([np.tile([0.2, 0.6, 0.2], (len(ground), 1)), np.tile([0.9, 0.1, 0.1], (len(pts) - len(ground), 1))])
    write_splat(splat_mesh.splat_path(project), pts, normals, rgb, size=step * 0.9)
    frames = []  # 12 cameras on a ring at 6 m height, looking at the centre (OpenGL camera axes)
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        c = np.array([5 * np.cos(a), 5 * np.sin(a), 6.0])
        f = -c / np.linalg.norm(c)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, c
        frames.append({"file_path": "images/x.jpg", "transform_matrix": c2w.tolist()})
    json.dump({"camera_model": "OPENCV", "fl_x": 300.0, "fl_y": 300.0, "cx": 320.0, "cy": 240.0, "w": 640, "h": 480,
               "k1": 0, "k2": 0, "p1": 0, "p2": 0, "frames": frames}, open(os.path.join(project, "transforms.json"), "w"))


def test_load_splat_and_cameras(tmp_path):
    p = str(tmp_path)
    scene(p)
    g = splat_mesh.load_splat(splat_mesh.splat_path(p))
    assert g["sh"].shape[1:] == (1, 3) and np.allclose(g["sh"][0, 0] * SH_C0 + 0.5, [0.2, 0.6, 0.2], atol=1e-5)
    views, Ks, W, H = splat_mesh.cameras(p, width=320)
    assert (W, H) == (320, 240) and np.isclose(Ks[0, 0, 0], 150)
    c = -views[0][:3, :3].T @ views[0][:3, 3]  # camera centre, and the scene centre ahead (+z in OpenCV)
    assert np.allclose(c, [5, 0, 6], atol=1e-5) and (views[0] @ [0, 0, 0, 1.0])[2] > 0
    assert splat_mesh.parse_progress("[tsdf] view 3 / 12") == (3, 12)


@pytest.mark.skipif(splat.available() is not None, reason="needs the CUDA image with nerfstudio")
def test_mesh_from_splat(tmp_path):
    import trimesh
    p = str(tmp_path)
    scene(p)
    stats = splat_mesh.run(p, "fine", width=640, on_line=lambda m: None)
    assert stats["faces"] > 1000 and stats["views"] == 12
    m = trimesh.load(os.path.join(p, splat_mesh.OUT, "mesh.ply"), process=False)
    v = np.asarray(m.vertices)
    inbox = (np.abs(v[:, 0]) < 0.9) & (np.abs(v[:, 1]) < 0.9)
    ground = (np.abs(v[:, 0]) > 1.3) | (np.abs(v[:, 1]) > 1.3)
    vox = stats["voxel_m"]
    # within 1.5 voxels: rendered depth is the alpha-blended Gaussian *centre* depth, which oblique views pull
    # towards the camera by about the Gaussian size (here 4.5 cm discs)
    assert np.median(np.abs(v[ground, 2])) < 1.5 * vox and np.median(np.abs(v[inbox, 2] - 1.0)) < 1.5 * vox
    cols = np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
    assert np.allclose(np.median(cols[ground], 0), [0.2, 0.6, 0.2], atol=0.08)
    assert np.allclose(np.median(cols[inbox], 0), [0.9, 0.1, 0.1], atol=0.08)
    assert products.available(p)["meshes"]["splat"] == {"textured": False, "plain": True}
    assert gdata.dense_artifacts(p)["meshes"]["splat"].endswith("preview.ply")
    glb = products.export_mesh(p, "splat", "glb")[0]
    assert os.path.getsize(glb) > 0


def test_marching_cubes_in_pieces(tmp_path):
    """Open3D's marching cubes crashes from 2^18 blocks on, so large grids go in pieces: the same mesh."""
    p = str(tmp_path)
    scene(p)
    whole = splat_mesh.run(p, "fine", width=640, on_line=lambda m: None)
    lines = []
    pieces = splat_mesh.run(p, "fine", width=640, on_line=lines.append, mc_max_blocks=40)
    assert any("piece 2 /" in s for s in lines)
    assert abs(pieces["faces_raw"] - whole["faces_raw"]) <= 0.001 * whole["faces_raw"]
    assert abs(pieces["vertices"] - whole["vertices"]) <= 0.001 * whole["vertices"]


def test_patch_holes(tmp_path):
    """Holes up to the size are closed (also two touching at a vertex); larger ones, the outer border and a separate
    piece's border stay open."""
    from modelforge import sfm_mesh
    out = tmp_path / splat_mesh.OUT
    out.mkdir()
    n = 41  # a 4 x 4 m ground of 0.1 m cells
    gx, gy = np.meshgrid(np.arange(n) * 0.1, np.arange(n) * 0.1, indexing="ij")
    xyz = np.c_[gx.ravel(), gy.ravel(), np.zeros(n * n)]
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            small = 5 <= i < 8 and 5 <= j < 8 or (i, j) == (8, 8)  # 0.3 m hole + a 0.1 m one touching its corner
            large = 20 <= i < 35 and 20 <= j < 35  # 1.5 m hole
            if not (small or large):
                a = i * n + j
                faces += [(a, a + n, a + 1), (a + 1, a + n, a + n + 1)]
    island = len(xyz) + np.array([[0, 2, 1], [1, 2, 3]])  # a separate 0.2 m square, 10 m away
    xyz = np.r_[xyz, [[10, 10, 0], [10.2, 10, 0], [10, 10.2, 0], [10.2, 10.2, 0]]]
    faces = np.r_[np.array(faces), island]
    v = np.zeros(len(xyz), [(c, "f8") for c in ("x", "y", "z", "nx", "ny", "nz")] + [(c, "u1") for c in ("red", "green", "blue")])
    v["x"], v["y"], v["z"], v["nz"], v["green"] = *xyz.T, 1.0, 200
    sfm_mesh.write_mesh(str(out / "mesh.ply"), v, faces)
    json.dump({"voxel_m": 0.1}, open(out / "stats.json", "w"))
    rep = splat_mesh.fill(str(tmp_path), max_hole=1.0, on_line=lambda m: None)
    assert rep["holes"] == 2 and rep["faces_added"] == 16 and rep["largest_patched"] == pytest.approx(0.3)
    v2, f2 = sfm_mesh.read_mesh(str(out / "mesh.ply"))
    assert len(f2) == len(faces) + 16 and (out / splat_mesh.UNPATCHED).exists()
    new = f2[len(faces):]
    p = np.c_[v2["x"], v2["y"], v2["z"]]
    nrm = np.cross(p[new[:, 1]] - p[new[:, 0]], p[new[:, 2]] - p[new[:, 0]])
    old = np.cross(p[f2[0, 1]] - p[f2[0, 0]], p[f2[0, 2]] - p[f2[0, 0]])
    assert (np.sign(nrm[:, 2]) == np.sign(old[2])).all()  # the same side up as the ground
    assert v2["green"][-1] == 200 and v2["nz"][-1] == pytest.approx(1.0)
    rep = splat_mesh.fill(str(tmp_path), max_hole=2.0, on_line=lambda m: None)  # again, from the unpatched mesh
    assert rep["holes"] == 3 and len(sfm_mesh.read_mesh(str(out / "mesh.ply"))[1]) == len(faces) + 16 + 60

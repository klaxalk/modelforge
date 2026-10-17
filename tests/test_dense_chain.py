"""Dense chain after PatchMatch (fusion -> Delaunay / Poisson meshing) on a synthetic
workspace with exact depth/normal maps of a bumpy ground z = h(x, y). (A perfectly
planar scene makes CGAL's 3D Delaunay degenerate and crashes the graph cut.)

PatchMatch itself needs CUDA; this checks everything downstream of it on CPU, including
COLMAP's workspace layout and depth-map format (header "W&H&C&", then float32 planes).
"""

import os

import cv2
import numpy as np
import plyfile
import pycolmap
import pytest

from modelforge import dense
from modelforge.sfm.priors import rot_to_quat_xyzw

W, H, F = 160, 120, 110.0


def h(x, y):
    return 0.4 * np.sin(1.3 * x) * np.cos(1.1 * y)


def h_grad(x, y):
    return 0.52 * np.cos(1.3 * x) * np.cos(1.1 * y), -0.44 * np.sin(1.3 * x) * np.sin(1.1 * y)


def write_map(path, arr):
    arr = np.atleast_3d(arr).astype(np.float32)
    h, w, c = arr.shape
    with open(path, "wb") as f:
        f.write(f"{w}&{h}&{c}&".encode())
        f.write(np.ascontiguousarray(arr.transpose(2, 0, 1)).astype("<f4").tobytes())  # planar


def make_workspace(project):
    ws = os.path.join(project, "dense")
    for d in ("images", "sparse", "stereo/depth_maps", "stereo/normal_maps"):
        os.makedirs(os.path.join(ws, d), exist_ok=True)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[F, F, W / 2, H / 2], camera_id=1)
    rec.add_camera_with_trivial_rig(cam)
    R_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # cam x = E, cam y = -N, cam z = down
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1)
    # sparse points on the plane: fusion takes image overlap from shared sparse tracks
    gx, gy = np.meshgrid(np.linspace(-3, 3, 9), np.linspace(-2.5, 2.5, 7))
    sparse = np.stack([gx.ravel(), gy.ravel(), h(gx.ravel(), gy.ravel())], -1)
    tracks = [[] for _ in sparse]
    names = []
    k = 0
    for cx in (-2.0, 0.0, 2.0):
        for cy in (-1.5, 1.5):
            k += 1
            tilt = np.radians(5.0 * cx / 2)
            Rt = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
            R_cw = Rt @ R_down  # cameras 10 m up, slightly tilted, on a 3 x 2 grid
            C = np.array([cx, cy, 10.0])
            rays_w = rays @ R_cw  # = R_cw^T @ ray per pixel
            depth = -C[2] / rays_w[..., 2]  # camera-z depth (ray z = 1); start on z = 0
            for _ in range(30):  # fixed-point ray / surface intersection (relief << height)
                X = C + depth[..., None] * rays_w
                depth = (h(X[..., 0], X[..., 1]) - C[2]) / rays_w[..., 2]
            X = C + depth[..., None] * rays_w
            tex = (((np.floor(X[..., 0] * 2) + np.floor(X[..., 1] * 2)) % 2) * 160 + 40).astype(np.int16)
            tex += (np.sin(X[..., 0] * 7.3) * np.cos(X[..., 1] * 5.1) * 40).astype(np.int16)
            name = f"img_{k:02d}.jpg"
            cv2.imwrite(os.path.join(ws, "images", name), np.stack([np.clip(tex, 0, 255).astype(np.uint8)] * 3, -1))
            gxh, gyh = h_grad(X[..., 0], X[..., 1])
            n_w = np.stack([-gxh, -gyh, np.ones_like(gxh)], -1)
            n_w /= np.linalg.norm(n_w, axis=-1, keepdims=True)
            normal = n_w @ R_cw.T  # camera frame; up-facing normals face these downward cameras
            write_map(os.path.join(ws, "stereo/depth_maps", f"{name}.geometric.bin"), depth)
            write_map(os.path.join(ws, "stereo/normal_maps", f"{name}.geometric.bin"), normal)
            Xc = (sparse - C) @ R_cw.T
            uv = Xc[:, :2] / Xc[:, 2:] * F + np.array([W / 2, H / 2])
            vis = np.flatnonzero((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H))
            for kp, j in enumerate(vis):
                tracks[j].append((k, kp))
            pose = pycolmap.Rigid3d()
            pose.params = np.r_[rot_to_quat_xyzw(R_cw), -R_cw @ C]
            rec.add_image_with_trivial_frame(pycolmap.Image(name=name, keypoints=uv[vis], camera_id=1, image_id=k), pose)
            names.append(name)
    for xyz, tr in zip(sparse, tracks):
        if len(tr) >= 2:
            track = pycolmap.Track()
            for image_id, kp in tr:
                track.add_element(image_id, kp)
            rec.add_point3D(xyz, track)
    rec.write(os.path.join(ws, "sparse"))
    with open(os.path.join(ws, "stereo", "fusion.cfg"), "w") as f:
        f.write("\n".join(names) + "\n")
    with open(os.path.join(ws, "stereo", "patch-match.cfg"), "w") as f:
        f.write("".join(f"{n}\n__auto__, 5\n" for n in names))
    return ws


def ply_vertices(path):
    v = plyfile.PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], -1)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("dense_project"))
    make_workspace(p)
    return p


def test_fusion_recovers_the_plane(project):
    st = dense.fusion(project)
    pts = ply_vertices(st["path"])
    err = np.abs(pts[:, 2] - h(pts[:, 0], pts[:, 1]))
    print(st, "surface error max", err.max())
    assert st["points"] > 2000
    assert err.max() < 0.05
    assert os.path.exists(st["path"] + ".vis")


@pytest.mark.parametrize("method", dense.MESH_METHODS)
def test_meshing(project, method):
    if not os.path.exists(os.path.join(project, "dense", "fused.ply")):
        dense.fusion(project)
    # trim=0: with only ~5k synthetic points Poisson density trimming would remove everything
    st = dense.mesh(project, method, poisson_depth=7, poisson_trim=0.0)
    ply = plyfile.PlyData.read(st["path"])
    v = ply_vertices(st["path"])
    nf = ply["face"].count
    err = np.abs(v[:, 2] - h(v[:, 0], v[:, 1]))
    print(method, st, "vertices", len(v), "faces", nf, "surface error median", np.median(err))
    assert nf > 100
    assert np.median(err) < 0.1


def test_texture_and_preview(project):
    if not os.path.exists(os.path.join(project, "dense", "meshed-delaunay.ply")):
        dense.fusion(project)
        dense.mesh(project, "delaunay")
    st = dense.texture(project, "delaunay")
    print(st)
    assert "texture.png" in st["files"] and any(f.endswith(".ply") for f in st["files"])
    tex = cv2.imread(os.path.join(st["path"], "texture.png"))
    assert tex is not None and tex.std() > 5  # not a blank atlas
    pv = dense.preview(project, "delaunay", max_faces=1000)
    print(pv)
    assert pv["faces_out"] <= 1.2 * 1000 < pv["faces_in"]


def test_stereo_without_cuda_fails_clearly(project):
    if dense.has_cuda():
        pytest.skip("CUDA available")
    with pytest.raises(RuntimeError, match="needs CUDA"):
        dense.stereo(project)

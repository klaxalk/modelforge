"""Depth maps of a mesh at every training camera, as splatfacto sees the photo: the dense SfM mesh prior.

Runs in nerfstudio's venv (Open3D, OpenCV; CPU raycasting). For each frame of DATASET/transforms.json: the camera
at the training downscale (intrinsics x 1/N, size floor(w / N), as nerfstudio's dataparser), undistorted and cropped
exactly as nerfstudio's datamanager does (`_undistort_image`: K - 0.5, cv2.getOptimalNewCameraMatrix(alpha 0), roi
crop, + 0.5), so a map lines up pixel for pixel with the training image; a ray through every pixel centre, the
z-depth of the first mesh hit (0 = no hit) -> OUT/<image name>.npy (float16, metres in the transforms' world).
Maps that exist for the same mesh, downscale and camera are kept (OUT/manifest.json), so tiles share them.

python export/mesh_depth.py DATASET MESH OUT --downscale N
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


def undistorted_camera(t, downscale):
    """(K, width, height) of the undistorted, cropped training image (nerfstudio's datamanager)."""
    s = 1.0 / downscale
    w, h = int(t["w"] * s), int(t["h"] * s)
    K = np.array([[t["fl_x"] * s, 0, t["cx"] * s], [0, t["fl_y"] * s, t["cy"] * s], [0, 0, 1.0]])
    dist = np.array([t.get("k1", 0.0), t.get("k2", 0.0), t.get("p1", 0.0), t.get("p2", 0.0), t.get("k3", 0.0), 0, 0, 0])
    if not np.any(dist):
        return K, w, h
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    x, y, rw, rh = roi
    newK[0, 2] += 0.5 - x
    newK[1, 2] += 0.5 - y
    return newK, rw, rh


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=1)
    a = ap.parse_args()
    t = json.load(open(os.path.join(a.dataset, "transforms.json")))
    if t.get("camera_model", "OPENCV") != "OPENCV":
        raise SystemExit(f"mesh depth: camera model {t.get('camera_model')} not supported (OPENCV only)")
    K, w, h = undistorted_camera(t, a.downscale)
    st = os.stat(a.mesh)
    key = {"mesh": os.path.realpath(a.mesh), "size": st.st_size,  # one cache whatever path reaches the mesh "mtime": st.st_mtime, "downscale": a.downscale,
           "K": np.round(K, 6).tolist(), "w": w, "h": h}
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.json")
    if os.path.exists(man) and json.load(open(man)) != key:  # another mesh or camera: start over
        for f in os.listdir(a.out):
            if f.endswith(".npy"):
                os.remove(os.path.join(a.out, f))
    json.dump(key, open(man, "w"), indent=1)
    todo = [f for f in t["frames"]
            if not os.path.exists(os.path.join(a.out, os.path.basename(f["file_path"]) + ".npy"))]
    print(f"mesh depth: {len(t['frames'])} photos: {len(t['frames']) - len(todo)} maps exist already (from other "
          f"tiles or runs), {len(todo)} to render, {w} x {h} px", flush=True)
    if not todo:
        return
    t0 = time.time()
    mesh = o3d.t.io.read_triangle_mesh(a.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(mesh.vertex.positions.to(o3d.core.float32),
                                                    mesh.triangle.indices.to(o3d.core.int32)))
    print(f"mesh depth: {len(mesh.triangle.indices):,} triangles loaded in {time.time() - t0:.0f} s", flush=True)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    for k, f in enumerate(todo):
        c2w = np.array(f["transform_matrix"], float)
        R = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenGL -> OpenCV camera axes
        d = d_cam @ R.T  # unnormalized: t_hit is then the z-depth
        rays = np.concatenate([np.broadcast_to(c2w[:3, 3], d.shape), d], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(h, w)
        hit[~np.isfinite(hit)] = 0
        name = os.path.join(a.out, os.path.basename(f["file_path"]) + ".npy")
        np.save(name + ".part.npy", hit.astype(np.float16))
        os.replace(name + ".part.npy", name)
        if (k + 1) % 20 == 0 or k + 1 == len(todo):
            print(f"mesh depth: {k + 1} / {len(todo)} missing maps of {len(t['frames'])} photos "
                  f"({(time.time() - t0) / (k + 1):.2f} s per map)", flush=True)


if __name__ == "__main__":
    main()

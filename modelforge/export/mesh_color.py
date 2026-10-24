"""Colour a mesh's vertices from the photos (the SfM mesh, Sparse tab "Colour SfM mesh from the photos").

Runs in nerfstudio's venv (Open3D, OpenCV). For every frame of PROJECT/transforms.json: the vertices are projected
with the camera's OPENCV distortion into the photo (decoded at <= --width px); a vertex counts when it is in front,
inside the frame, faces the camera (cos > 0.1) and is not hidden: a ray from the camera centre to it hits the mesh
no earlier than just before it (Open3D raycasting). Its colour is the weighted mean over those photos, weight
(cos / distance)^4: the near, head-on views dominate, so detail stays sharp where one photo saw it best, and far or
grazing views only fill in. Vertices no photo saw keep their colour (Poisson's, from the SfM points).

With --live PATH, the colours so far (uint8 RGB per vertex, in the mesh file's vertex order) are saved to PATH
every --every photos ("mesh colour: live k / n"): the GUI shows them while it runs, and a stopped run keeps them.

python export/mesh_color.py PROJECT MESH OUT [--width 1920] [--live PATH --every 100]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame_camera  # noqa: E402

POWER = 4  # weight (cos / distance) ** POWER


def project(Xc, K, dist):
    """Pixel coordinates (OpenCV OPENCV model) of camera-frame points with z > 0."""
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2 = dist
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


def bilinear(img, u, v):
    """Colours (N, 3) at pixel coordinates (pixel centres at integer + 0.5, as COLMAP and nerfstudio)."""
    x, y = u - 0.5, v - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    h, w = img.shape[:2]
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    a, b = img[y0, x0].astype(np.float32), img[y0, x0 + 1].astype(np.float32)
    c, d = img[y0 + 1, x0].astype(np.float32), img[y0 + 1, x0 + 1].astype(np.float32)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1920, help="photos are decoded at most this wide")
    ap.add_argument("--live", help="save the colours so far here (.npy) every --every photos")
    ap.add_argument("--every", type=int, default=100)
    a = ap.parse_args()
    t = json.load(open(os.path.join(a.project, "transforms.json")))
    cams, index = frame_camera.cameras(t)
    if any(c.get("camera_model", "OPENCV") != "OPENCV" for c in cams):
        raise SystemExit("mesh colour: only OPENCV cameras are supported")
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    print(f"mesh colour: {len(V):,} vertices, {len(mesh.triangles):,} triangles loaded in {time.time() - t0:.0f} s",
          flush=True)
    def decoding(c):
        """(imread flag, K, dist) of a camera: JPEG decoding at 1/2, 1/4, 1/8 (DCT scaling) down to --width."""
        reduce = 1
        while c["w"] / (reduce * 2) >= a.width and reduce < 8:
            reduce *= 2
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
        K = np.array([[c["fl_x"], 0, c["cx"]], [0, c["fl_y"], c["cy"]], [0, 0, 1.0]]) / reduce
        K[2, 2] = 1.0
        return flag, K, (c.get("k1", 0.0), c.get("k2", 0.0), c.get("p1", 0.0), c.get("p2", 0.0))
    per_cam = [decoding(c) for c in cams]
    acc, wsum = np.zeros((len(V), 3)), np.zeros(len(V))
    old = (np.asarray(mesh.vertex_colors) * 255.0) if mesh.has_vertex_colors() else np.full((len(V), 3), 128.0)

    def current():
        rgb = old.copy()
        seen = wsum > 0
        rgb[seen] = acc[seen] / wsum[seen, None]
        return np.clip(rgb, 0, 255), seen
    V32 = V.astype(np.float32)
    frames = t["frames"]
    for k, f in enumerate(frames):
        if (k + 1) % 10 == 0 or k + 1 == len(frames):
            print(f"mesh colour: {k + 1} / {len(frames)} ({(time.time() - t0) / (k + 1):.2f} s per photo)", flush=True)
        if a.live and k and k % a.every == 0:
            np.save(a.live + ".part.npy", np.round(current()[0]).astype(np.uint8))
            os.replace(a.live + ".part.npy", a.live)
            print(f"mesh colour: live {k} / {len(frames)}", flush=True)
        flag, K, dist = per_cam[index[k]]
        img = cv2.imread(os.path.join(a.project, f["file_path"]), flag)
        if img is None:
            continue
        h, w = img.shape[:2]
        c2w = np.array(f["transform_matrix"], float)
        R, C = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0]), c2w[:3, 3]  # OpenGL -> OpenCV camera axes
        Xc = (V32 - C.astype(np.float32)) @ R.astype(np.float32)
        idx = np.flatnonzero(Xc[:, 2] > 0.05)
        u, v = project(Xc[idx].astype(np.float64), K, dist)
        ok = (u >= 0.5) & (u < w - 0.5) & (v >= 0.5) & (v < h - 0.5)
        idx, u, v = idx[ok], u[ok], v[ok]
        ray = V[idx] - C
        d = np.linalg.norm(ray, axis=1)
        cos = np.einsum("ij,ij->i", N[idx], -ray) / d
        ok = cos > 0.1
        idx, u, v, ray, d, cos = idx[ok], u[ok], v[ok], ray[ok], d[ok], cos[ok]
        if not len(idx):
            continue
        rays = np.concatenate([np.broadcast_to(C, ray.shape), ray], 1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()  # 1 = the vertex itself
        seen = hit >= 1 - np.maximum(0.002, 0.05 / d)  # nothing in between (5 cm / 0.2 % slack)
        idx, u, v, d, cos = idx[seen], u[seen], v[seen], d[seen], cos[seen]
        wgt = (cos / d) ** POWER  # float64: (1 / 30 m) ** 4 ~ 1e-6 is no problem
        col = bilinear(img, u, v)[:, ::-1]  # BGR -> RGB
        acc[idx] += col * wgt[:, None]  # idx is unique within a photo
        wsum[idx] += wgt
    rgb, seen = current()
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb / 255.0)
    mesh.vertex_normals = o3d.utility.Vector3dVector()  # keep the file as the input's: positions, colours, faces
    o3d.io.write_triangle_mesh(a.out + ".part.ply", mesh, write_ascii=False, compressed=False,
                               write_vertex_normals=False, write_vertex_colors=True, write_triangle_uvs=False)
    os.replace(a.out + ".part.ply", a.out)
    print(f"mesh colour: {seen.mean():.1%} of the vertices coloured from {len(frames)} photos in "
          f"{time.time() - t0:.0f} s", flush=True)
    json.dump({"vertices": int(len(V)), "coloured": int(seen.sum()), "photos": len(frames), "width": a.width,
               "seconds": round(time.time() - t0, 1)}, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()

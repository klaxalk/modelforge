"""Share of each camera's image that shows each tile: the SfM mesh raycast from every camera, every hit looked up in
a label map on the ground. modelforge.splat_tiles uses it to decide which photos train a tile.

Runs in nerfstudio's venv (Open3D, CPU raycasting). For each frame of DATASET/transforms.json a pinhole camera of
its intrinsics scaled to --width pixels (the small lens distortion is ignored: this counts pixels per tile, it does
not align them); rays through every pixel centre; the first mesh hit's (x, y) -> label (LABELS.npz: lo, grid,
labels[row = y, col = x], positions beyond the grid take the nearest edge cell, as tile_regions.Regions). A pixel
without a hit (sky, holes) counts for no tile, so the shares of a view add up to its hit fraction.

python export/mesh_share.py DATASET MESH LABELS.npz OUT.npz [--width 240]
OUT.npz: share (frames x tiles, float32), hit (frames).
"""
import argparse
import json
import os
import sys
import time

import numpy as np



def _load_frame_camera():
    """export/frame_camera.py by its path: this folder on sys.path would shadow nerfstudio (export/nerfstudio.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mf_frame_camera", os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_camera.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frame_camera = _load_frame_camera()


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("mesh")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=240)
    a = ap.parse_args()
    t = json.load(open(os.path.join(a.dataset, "transforms.json")))
    d = np.load(a.labels)
    lo, grid, labels = d["lo"].astype(float), float(d["grid"]), d["labels"]
    tiles = int(labels.max()) + 1
    mesh = o3d.io.read_triangle_mesh(a.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cams, index = frame_camera.cameras(t)

    def pixel_dirs(c):
        s = a.width / float(c["w"])
        W, H = a.width, max(1, int(round(c["h"] * s)))
        fx, fy, cx, cy = c["fl_x"] * s, c["fl_y"] * s, c["cx"] * s, c["cy"] * s
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
    per_cam = [pixel_dirs(c) for c in cams]
    n = len(t["frames"])
    share, hit = np.zeros((n, tiles), np.float32), np.zeros(n, np.float32)
    t0 = time.time()
    for i, f in enumerate(t["frames"]):
        dirs = per_cam[index[i]]
        c2w = np.array(f["transform_matrix"], float)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
        rays = np.concatenate([np.repeat(c2w[None, :3, 3], len(dirs), 0), dirs @ c2w[:3, :3].T], 1).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        ok = np.isfinite(th)
        hit[i] = ok.mean()
        if ok.any():
            p = rays[ok, :3] + rays[ok, 3:] * th[ok, None]
            col = np.clip(np.floor((p[:, 0] - lo[0]) / grid).astype(int), 0, labels.shape[1] - 1)
            row = np.clip(np.floor((p[:, 1] - lo[1]) / grid).astype(int), 0, labels.shape[0] - 1)
            share[i] = np.bincount(labels[row, col], minlength=tiles)[:tiles] / len(th)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"mesh share: {i + 1} / {n} ({time.time() - t0:.0f} s)", flush=True)
    np.savez(a.out, share=share, hit=hit)


if __name__ == "__main__":
    main()

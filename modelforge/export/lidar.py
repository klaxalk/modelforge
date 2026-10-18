"""LiDAR points for the outputs (project kind "rosbag"): colour them from the registered images and
merge them into the initial point cloud for splatfacto, so Gaussians start on the true surfaces even
where the images lack texture.

Colour: each point takes the colour of the nearest camera that sees it inside the image (front of the
camera, inside the frame). Occlusion is not modelled, so a point hidden behind another surface may
take that surface's colour; splatfacto re-optimizes colours anyway, positions are what matter.
"""

import os

import numpy as np
from PIL import Image

CLOUD = os.path.join("lidar", "cloud.ply")


def load(project):
    """(N, 3) LiDAR points in the world frame, or None."""
    path = os.path.join(project, CLOUD)
    if not os.path.exists(path):
        return None
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float64)


def colorize(rec, image_dir, xyz, max_depth=80.0):
    """(N, 3) uint8 colours and a seen mask for world points, from the registered images."""
    n = len(xyz)
    best = np.full(n, np.inf)
    rgb = np.zeros((n, 3), np.uint8)
    for image_id in rec.reg_image_ids():
        img = rec.image(image_id)
        cam = rec.cameras[img.camera_id]
        T = img.cam_from_world().matrix()
        Xc = xyz @ T[:, :3].T + T[:, 3]
        front = (Xc[:, 2] > 0.1) & (Xc[:, 2] < max_depth) & (Xc[:, 2] < best)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        uv = cam.img_from_cam(Xc[idx])
        inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam.width - 1) & (uv[:, 1] < cam.height - 1)
        if not inside.any():
            continue
        idx, uv = idx[inside], uv[inside]
        pix = np.asarray(Image.open(os.path.join(image_dir, img.name)).convert("RGB"))
        rgb[idx] = pix[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        best[idx] = Xc[idx, 2]
    return rgb, np.isfinite(best)


def merged_cloud(rec, project, image_dir="images", max_points=300_000, seed=0):
    """SfM points + coloured LiDAR points (only those seen by a camera), for sparse_pc.ply."""
    sfm = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    sfm_rgb = np.array([p.color for p in rec.points3D.values()], np.uint8).reshape(-1, 3)
    xyz = load(project)
    if xyz is None or not len(xyz):
        return sfm, sfm_rgb, 0
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    rgb, seen = colorize(rec, os.path.join(project, image_dir), xyz)
    return np.concatenate([sfm, xyz[seen]]), np.concatenate([sfm_rgb, rgb[seen]]), int(seen.sum())


def write_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                                  ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    v["red"], v["green"], v["blue"] = rgb.T
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def nearest_distance(query, cloud, max_cloud=500_000, seed=0):
    """Distance from each query point to its nearest cloud point (OpenCV FLANN k-d trees)."""
    import cv2
    if len(cloud) > max_cloud:
        cloud = cloud[np.random.default_rng(seed).choice(len(cloud), max_cloud, replace=False)]
    index = cv2.flann_Index(cloud.astype(np.float32), dict(algorithm=1, trees=4))
    _, d2 = index.knnSearch(query.astype(np.float32), 1, params=dict(checks=64))
    return np.sqrt(d2[:, 0].astype(np.float64))


def sfm_check(rec, project, covered_m=2.0):
    """How well the refined SfM points agree with the LiDAR, over the points the LiDAR covers (nearest
    return within covered_m; far background beyond the LiDAR range is left out). None without LiDAR.
    The distance is bounded below by the cloud's own point spacing."""
    cloud = load(project)
    if cloud is None or not len(cloud) or not rec.num_points3D():
        return None
    pts = np.array([p.xyz for p in rec.points3D.values()])
    d = nearest_distance(pts, cloud)
    c = d[d < covered_m]
    if not len(c):
        return {"points": len(pts), "covered": 0.0}
    return {"points": len(pts), "covered": round(float(len(c) / len(d)), 3), "median_m": round(float(np.median(c)), 4),
            "p90_m": round(float(np.percentile(c, 90)), 4), "within_10cm": round(float(np.mean(c < 0.10)), 3)}

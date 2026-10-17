"""Project data for the GUI as plain numpy: priors, reconstruction snapshots, thumbnails.

All poses are world_from_cam (ENU world, OpenCV camera), which is also viser's
frustum convention: wxyz = scalar-first quaternion of R_wc, position = center.
"""

import json
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pycolmap

from modelforge.sfm import priors as priors_mod


def rot_to_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def xyzw_to_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Cameras:
    """Per-image poses, aligned with Project.names (NaN rows = not registered)."""
    centers: np.ndarray  # (N, 3)
    R_wc: np.ndarray  # (N, 3, 3)

    @classmethod
    def empty(cls, n):
        return cls(np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan))

    def wxyz(self, i):
        return rot_to_wxyz(self.R_wc[i])

    @property
    def valid(self):
        return np.isfinite(self.centers[:, 0])


@dataclass
class Snapshot:
    name: str
    cams: Cameras
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    point_error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reproj_error: np.ndarray = None  # (N,) mean per image [px]
    num_obs: np.ndarray = None  # (N,)
    intrinsics: dict = None  # width, height, params, model
    rec: object = None  # pycolmap.Reconstruction (None for the priors snapshot)


def _snapshot_order(name):
    if name == "priors":
        return (-1, 0)
    if name == "refined":
        return (10 ** 6, 0)
    m = re.match(r"round_(\d+)(_tri)?$", name)
    return (int(m.group(1)), 0 if m.group(2) else 1) if m else (10 ** 5, 0)


class Project:
    def __init__(self, path, thumb_width=192):
        self.path = path
        self.priors, self.meta = priors_mod.load(path)
        self.names = priors_mod.by_order(self.priors)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.pts = np.array([self.priors[n].pts for n in self.names])  # per video
        self.time = np.array([self.priors[n].time for n in self.names])  # project timeline
        self.source = np.array([self.priors[n].source for n in self.names])
        self.source_names = [s.get("prefix", f"source {k}") for k, s in enumerate(self.meta.get("sources", []))] \
            or [os.path.splitext(os.path.basename(self.meta.get("source", "video")))[0]]
        self.thumb_width = thumb_width
        self._thumbs = {}
        n = len(self.names)
        cams = Cameras.empty(n)
        for i, name in enumerate(self.names):
            p = self.priors[name]
            cams.centers[i] = p.center
            cams.R_wc[i] = p.R_cw.T
        self.prior_cams = cams

    def report(self):
        path = os.path.join(self.path, "report.json")
        return json.load(open(path)) if os.path.exists(path) else {}

    def snapshot_names(self):
        sparse = os.path.join(self.path, "sparse")
        names = ["priors"]
        if os.path.isdir(sparse):
            names += [d for d in os.listdir(sparse)
                      if os.path.exists(os.path.join(sparse, d, "images.bin"))]
        return sorted(names, key=_snapshot_order)

    def load_snapshot(self, name):
        if name == "priors":
            return Snapshot("priors", self.prior_cams)
        rec = pycolmap.Reconstruction(os.path.join(self.path, "sparse", name))
        return self.snapshot_from_reconstruction(name, rec)

    def poses_from_reconstruction(self, rec):
        """Registered poses only (cheap enough to call on every BA iteration)."""
        cams = Cameras.empty(len(self.names))
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            i = self.index.get(img.name)
            if i is None:
                continue
            params = np.array(rec.frame(img.frame_id).rig_from_world.params)
            R_cw = xyzw_to_rot(params[:4])
            cams.R_wc[i] = R_cw.T
            cams.centers[i] = -R_cw.T @ params[4:]
        return cams

    def snapshot_from_reconstruction(self, name, rec):
        n = len(self.names)
        cams = self.poses_from_reconstruction(rec)
        reproj = np.full(n, np.nan)
        nobs = np.zeros(n, int)
        cam_by_id = rec.cameras
        for image_id in rec.reg_image_ids():
            img = rec.image(image_id)
            i = self.index.get(img.name)
            if i is None:
                continue
            R_cw = cams.R_wc[i].T
            t = -R_cw @ cams.centers[i]
            obs = [(p.xy, p.point3D_id) for p in img.points2D if p.has_point3D()]
            nobs[i] = len(obs)
            if obs:
                xy = np.array([o[0] for o in obs])
                X = np.array([rec.point3D(o[1]).xyz for o in obs])
                Xc = X @ R_cw.T + t
                proj = np.asarray(cam_by_id[img.camera_id].img_from_cam(Xc))
                err = np.linalg.norm(proj - xy, axis=1)
                err = err[np.isfinite(err)]  # points behind the camera project to NaN before filtering
                reproj[i] = float(err.mean()) if len(err) else np.nan
        pts = rec.points3D
        points = np.array([p.xyz for p in pts.values()]) if len(pts) else np.zeros((0, 3))
        colors = np.array([p.color for p in pts.values()], np.uint8) if len(pts) else np.zeros((0, 3), np.uint8)
        rec.update_point_3d_errors()
        perr = np.array([p.error for p in pts.values()]) if len(pts) else np.zeros(0)
        cam = next(iter(rec.cameras.values()))
        intr = {"model": cam.model.name, "width": int(cam.width), "height": int(cam.height),
                "params": [float(v) for v in cam.params]}
        return Snapshot(name, cams, points, colors, perr, reproj, nobs, intr, rec)

    def thumbnail(self, i):
        if i not in self._thumbs:
            img = cv2.imread(os.path.join(self.path, "images", self.names[i]))
            h, w = img.shape[:2]
            img = cv2.resize(img, (self.thumb_width, round(h * self.thumb_width / w)), interpolation=cv2.INTER_AREA)
            self._thumbs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._thumbs[i]

    def inspect_image(self, snap, i, width=960, residual_gain=10.0):
        """Image with observed keypoints (green), magnified reprojection residuals (red)
        and keypoints without a 3D point (grey)."""
        img = cv2.cvtColor(cv2.imread(os.path.join(self.path, "images", self.names[i])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = width / w
        img = cv2.resize(img, (width, round(h * s)), interpolation=cv2.INTER_AREA)
        stats = {}
        rec = snap.rec
        if rec is not None:
            image = rec.find_image_with_name(self.names[i])
            if image is not None and snap.cams.valid[i]:
                R_cw = snap.cams.R_wc[i].T
                t = -R_cw @ snap.cams.centers[i]
                cam = rec.camera(image.camera_id)
                errs = []
                for p in image.points2D:
                    x, y = p.xy * s
                    if not p.has_point3D():
                        cv2.circle(img, (int(x), int(y)), 1, (150, 150, 150), -1)
                        continue
                    X = rec.point3D(p.point3D_id).xyz
                    u = np.asarray(cam.img_from_cam((R_cw @ X + t)[None]))[0]
                    errs.append(np.linalg.norm(u - p.xy))
                    u = u * s
                    end = (int(x + residual_gain * (u[0] - x)), int(y + residual_gain * (u[1] - y)))
                    cv2.line(img, (int(x), int(y)), end, (255, 60, 60), 1, cv2.LINE_AA)
                    cv2.circle(img, (int(x), int(y)), 2, (60, 220, 60), -1)
                if errs:
                    errs = np.array(errs)
                    stats = {"observations": len(errs), "keypoints": len(image.points2D),
                             "median_err_px": float(np.median(errs)), "p90_err_px": float(np.percentile(errs, 90))}
        return img, stats


# ---------------------------------------------------------------- dense / splat
SH_C0 = 0.28209479177387814


def dense_artifacts(path):
    """Which dense / splat outputs exist in the project (paths, or None)."""
    ws = os.path.join(path, "dense")
    found = {"fused": None, "splat": None, "meshes": {}}
    if os.path.exists(os.path.join(ws, "fused.ply")):
        found["fused"] = os.path.join(ws, "fused.ply")
    for method in ("delaunay", "poisson"):
        for name in (f"preview-{method}.ply", f"meshed-{method}.ply"):  # prefer the simplified copy
            p = os.path.join(ws, name)
            if os.path.exists(p):
                found["meshes"][method] = p
                break
    p = os.path.join(path, "splat", "splat_enu.ply")
    if os.path.exists(p):
        found["splat"] = p
    return found


def load_fused(ply_path, max_points=1_500_000, seed=0):
    import plyfile
    v = plyfile.PlyData.read(ply_path)["vertex"].data
    idx = np.arange(len(v))
    if len(v) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(v), max_points, replace=False))
    pts = np.stack([v["x"][idx], v["y"][idx], v["z"][idx]], -1).astype(np.float32)
    names = v.dtype.names
    if {"red", "green", "blue"} <= set(names):
        cols = np.stack([v["red"][idx], v["green"][idx], v["blue"][idx]], -1).astype(np.uint8)
    else:
        cols = np.full((len(idx), 3), 200, np.uint8)
    return pts, cols, len(v)


def load_mesh(ply_path):
    import trimesh
    return trimesh.load(ply_path, process=False)


def load_splat(ply_path, max_gaussians=1_000_000, seed=0):
    """ENU splat (modelforge.export.splat_to_enu output) as viser gaussian-splat arrays."""
    import plyfile
    return splat_arrays(plyfile.PlyData.read(ply_path)["vertex"].data, max_gaussians, seed)


def load_live_splat(npz_path, transforms_path, max_gaussians=500_000, seed=0):
    """A training snapshot (modelforge.export.ckpt_dump) mapped to ENU, as viser arrays; (step, arrays).
    Band 0 only: the preview shows base colours, so no SH rotation is needed."""
    from modelforge.export import splat_to_enu
    d = np.load(npz_path)
    n = len(d["means"])
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    v = np.zeros(n, dtype=[(k, "f4") for k in names])
    cols = np.c_[d["means"], d["scales"], d["quats"], d["features_dc"].reshape(n, -1)[:, :3],
                 d["opacities"].reshape(n)]
    for i, k in enumerate(names):
        v[k] = cols[:, i]
    if n > max_gaussians:  # subsample before the transform: cheaper, same result
        v = v[np.sort(np.random.default_rng(seed).choice(n, max_gaussians, replace=False))]
    R, t, s = splat_to_enu.load_transform(transforms_path)
    return int(d["step"]), n, splat_arrays(splat_to_enu.transform_vertices(v, R, t, s), max_gaussians, seed)


def splat_arrays(v, max_gaussians=1_000_000, seed=0):
    """Structured splat vertices (ENU) -> viser centers, covariances, rgbs, opacities."""
    idx = np.arange(len(v))
    if len(v) > max_gaussians:
        idx = np.sort(np.random.default_rng(seed).choice(len(v), max_gaussians, replace=False))
    v = v[idx]
    centers = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    s2 = np.exp(2 * np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
    cov = np.einsum("nij,nj,nkj->nik", R, s2, R).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), 0, 1).astype(np.float32)
    opacity = (1 / (1 + np.exp(-v["opacity"].astype(np.float64))))[:, None].astype(np.float32)
    return centers, cov, rgb, opacity

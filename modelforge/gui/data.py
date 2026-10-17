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
        self.names = sorted(self.priors, key=lambda n: self.priors[n].frame)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.pts = np.array([self.priors[n].pts for n in self.names])
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
                reproj[i] = float(np.nanmean(np.linalg.norm(proj - xy, axis=1)))
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

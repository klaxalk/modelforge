"""Pose priors from ingest's poses.json, in pycolmap conventions.

World = local ENU [m]. Poses are cam_from_world (COLMAP). pycolmap Rigid3d params
are [qx, qy, qz, qw, tx, ty, tz].

Uncertainty is specified in world axes and rotated into the camera frame, where
AbsolutePosePriorCost evaluates its residual: the rotation part is
log(R_cw_param * R_cw_prior^T), i.e. -R_cw * (world-frame rotation error), and the
translation part is R_cw_param * (C_prior - C_param). Both covariances therefore
transform as R_cw * Sigma_world * R_cw^T.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pycolmap


@dataclass
class PriorSigmas:
    pos_horizontal_m: float = 1.5  # consumer GNSS, absolute
    pos_vertical_m: float = 1.0  # rel_alt (baro-fused)
    roll_pitch_deg: float = 1.0
    yaw_deg: float = 3.0  # compass

    def world_covariances(self):
        pos = np.diag([self.pos_horizontal_m ** 2] * 2 + [self.pos_vertical_m ** 2])
        # small rotations about world E, N, U; yaw is about U
        rot = np.diag(np.radians([self.roll_pitch_deg, self.roll_pitch_deg, self.yaw_deg]) ** 2)
        return rot, pos


@dataclass
class Prior:
    name: str
    frame: int  # video frame index
    pts: float
    params: np.ndarray  # cam_from_world, [qx qy qz qw tx ty tz]
    center: np.ndarray  # ENU
    R_cw: np.ndarray

    def rigid(self):
        r = pycolmap.Rigid3d()
        r.params = self.params
        return r

    def covariance(self, sigmas):
        rot_w, pos_w = sigmas.world_covariances()
        cov = np.zeros((6, 6))
        cov[:3, :3] = self.R_cw @ rot_w @ self.R_cw.T
        cov[3:, 3:] = self.R_cw @ pos_w @ self.R_cw.T
        return cov

    @property
    def optical_axis(self):
        return self.R_cw[2]  # camera z in world = third row of R_cw


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load(project):
    """Return (priors by image name, poses.json metadata)."""
    meta = json.load(open(os.path.join(project, "poses.json")))
    priors = {}
    for r in meta["images"]:
        qw, qx, qy, qz = r["colmap_qvec"]
        t = np.array(r["colmap_tvec"])
        R_cw = quat_wxyz_to_rot((qw, qx, qy, qz))
        priors[r["image"]] = Prior(
            name=r["image"], frame=r["frame"], pts=r["pts"],
            params=np.array([qx, qy, qz, qw, *t]),
            center=np.array([r["enu_x"], r["enu_y"], r["enu_z"]]),
            R_cw=R_cw,
        )
    return priors, meta

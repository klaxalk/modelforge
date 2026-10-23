"""Project data for the GUI as plain numpy: priors, reconstruction snapshots, thumbnails.

All poses are world_from_cam (ENU world, OpenCV camera), which is also viser's
frustum convention: wxyz = scalar-first quaternion of R_wc, position = center.
"""

import json
import os
import re
import struct
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


def _slerp(q0, q1, u):
    """Spherical interpolation of unit quaternions (any convention), u in [0, 1]."""
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


ROT_M_PER_DEG = 0.02  # Fly: a turn on the spot counts as this much path per degree (25 deg/s at 0.5 m/s)


class FlightPath:
    """The cameras' flight in capture order, for replaying it in the 3D view (View tab → Fly).

    The path is parametrized by distance (m; scene units for videos without metadata), so Fly moves at
    a set speed: segment length = max(distance, rot_m_per_deg × rotation in degrees), so a turn on the
    spot still takes time. Where the project timeline (Project.time) pauses longer than max_gap_s (e.g.
    between sources), the segment is shortened to at most gap_m. Positions follow a cubic Hermite curve
    through the camera centres (finite-difference tangents); orientations are slerped between
    neighbouring cameras. `t` is the distance along the path, `duration` its length."""

    def __init__(self, cams, times, max_gap_s=3.0, gap_m=2.0, rot_m_per_deg=ROT_M_PER_DEG):
        ok = cams.valid
        if ok.sum() < 2:
            raise ValueError("fewer than 2 registered cameras: nothing to fly")
        idx = np.flatnonzero(ok)
        self.index = idx  # Project.names indices of the path's cameras
        self.centers = cams.centers[idx]
        q = np.array([rot_to_wxyz(cams.R_wc[i]) for i in idx])
        for k in range(1, len(q)):  # one hemisphere, so neighbours interpolate the short way
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        self.quats = q
        dist = np.linalg.norm(np.diff(self.centers, axis=0), axis=1)
        turn = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0.0, 1.0)))
        seg = np.maximum(np.maximum(dist, rot_m_per_deg * turn), 1e-6)
        t = np.asarray(times, float)[idx]
        if np.all(np.isfinite(t)):
            seg = np.where(np.diff(t) > max_gap_s, np.minimum(seg, gap_m), seg)
        self.t = np.concatenate([[0.0], np.cumsum(seg)])
        p, tt = self.centers, self.t
        m = np.zeros_like(p)
        m[1:-1] = (p[2:] - p[:-2]) / (tt[2:] - tt[:-2])[:, None]
        m[0], m[-1] = (p[1] - p[0]) / (tt[1] - tt[0]), (p[-1] - p[-2]) / (tt[-1] - tt[-2])
        self.tangents = m

    @property
    def duration(self):
        return float(self.t[-1])

    def pose(self, t):
        """(position, R_wc, index of the camera just passed) at path time t (clamped)."""
        t = float(np.clip(t, 0.0, self.duration))
        k = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        h = self.t[k + 1] - self.t[k]
        u = (t - self.t[k]) / h
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        pos = (h00 * self.centers[k] + h10 * h * self.tangents[k] + h01 * self.centers[k + 1]
               + h11 * h * self.tangents[k + 1])
        w, x, y, z = _slerp(self.quats[k], self.quats[k + 1], u)
        R = xyzw_to_rot((x, y, z, w))
        return pos, R, int(self.index[k + 1] if u > 0.5 else self.index[k])


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
    rec: object = None  # pycolmap.Reconstruction (live refine events without a written snapshot)
    obs: object = None  # Observations (snapshots read from disk): one image's keypoints for Inspect


def _snapshot_order(name):
    if name == "priors":
        return (-1, 0)
    if name == "refined":
        return (10 ** 6, 0)
    m = re.match(r"(round|calib)_(\d+)(_tri)?$", name)  # calib_N: a local refine's calibration subset, first
    if not m:
        return (10 ** 5, 0)
    return (int(m.group(2)) - (100 if m.group(1) == "calib" else 0), 0 if m.group(3) else 1)


# ---------------------------------------------------------------- COLMAP binary model, read with numpy
# pycolmap.Reconstruction + per-keypoint Python loops took 4.5 min for 8000 images (83M keypoints,
# 8M points); reading the files as arrays takes seconds. Layouts: colmap/scene/reconstruction_io_binary.cc
POINT2D = np.dtype([("xy", "<f8", 2), ("pid", "<i8")])  # images.bin keypoint; point3D_id uint64, -1 = none
POINT3D_HEAD = 8 + 24 + 3 + 8 + 8  # id, xyz, rgb, error, track length; then (image_id, point2D_idx) uint32 pairs


def read_cameras_bin(path):
    """cameras.bin -> {camera_id: pycolmap.Camera}."""
    buf = open(path, "rb").read()
    cams, off = {}, 8
    for _ in range(struct.unpack_from("<Q", buf)[0]):
        cid, model, w, h = struct.unpack_from("<IiQQ", buf, off)
        off += 24
        cam = pycolmap.Camera.create_from_model_id(cid, pycolmap.CameraModelId(model), 1.0, w, h)
        k = len(cam.params)
        cam.params = np.frombuffer(buf, "<f8", k, off)
        off += 8 * k
        cams[cid] = cam
    return cams


def read_images_bin(path):
    """images.bin -> list of (name, camera_id, R_cw, t_cw, offset of the keypoint block, keypoint count)."""
    mm = np.memmap(path, np.uint8, mode="r")
    out, off = [], 8
    for _ in range(int(mm[:8].view("<u8")[0])):
        head = mm[off:off + 64].tobytes()
        q = struct.unpack_from("<4d", head, 4)  # w, x, y, z
        t = np.array(struct.unpack_from("<3d", head, 36))
        cam_id = struct.unpack_from("<I", head, 60)[0]
        off += 64
        end = off
        while True:  # the name is null-terminated
            z = np.flatnonzero(mm[end:end + 256] == 0)
            if len(z):
                end += int(z[0])
                break
            end += 256
        name = mm[off:end].tobytes().decode()
        off = end + 1
        n = int(mm[off:off + 8].view("<u8")[0])
        off += 8
        out.append((name, cam_id, xyzw_to_rot((q[1], q[2], q[3], q[0])), t, off, n))
        off += n * POINT2D.itemsize
    return out


def read_keypoints(path, offset, n):
    return np.fromfile(path, POINT2D, n, offset=offset)


def read_points3d_bin(path, ids, track_lengths, chunk=200_000):
    """points3D.bin -> (xyz, rgb) for `ids` (sorted, as COLMAP writes them), whose track lengths come
    from counting the keypoints in images.bin: that gives every record's offset without a Python loop
    over millions of points. Raises ValueError if the file does not match."""
    mm = np.memmap(path, np.uint8, mode="r")
    if int(mm[:8].view("<u8")[0]) != len(ids):
        raise ValueError("points3D.bin: point count differs from images.bin")
    size = POINT3D_HEAD + 8 * track_lengths.astype(np.int64)
    offs = 8 + np.concatenate([[0], np.cumsum(size[:-1])]).astype(np.int64)
    if len(ids) and offs[-1] + size[-1] != len(mm):
        raise ValueError("points3D.bin: size differs from the tracks in images.bin")
    xyz, rgb = np.empty((len(ids), 3)), np.empty((len(ids), 3), np.uint8)
    cols = np.arange(35)  # id (8), xyz (24), rgb (3)
    for k in range(0, len(ids), chunk):
        raw = mm[offs[k:k + chunk, None] + cols]
        if not np.array_equal(raw[:, :8].copy().view("<i8")[:, 0], ids[k:k + chunk]):
            raise ValueError("points3D.bin: point ids differ from images.bin")
        xyz[k:k + chunk] = raw[:, 8:32].copy().view("<f8")
        rgb[k:k + chunk] = raw[:, 32:35]
    return xyz, rgb


class Observations:
    """Where each image's keypoints lie in images.bin, so Inspect reads one image on demand."""

    def __init__(self, path, blocks, ids):
        self.path, self.blocks, self.ids = path, blocks, ids  # blocks: {name index: (offset, n)}

    def image(self, i):
        """(xy (n, 2), row in Snapshot.points or -1) of image i's keypoints, or None."""
        if i not in self.blocks:
            return None
        a = read_keypoints(self.path, *self.blocks[i])
        pid = a["pid"]
        row = np.searchsorted(self.ids, pid).clip(0, max(len(self.ids) - 1, 0))
        ok = (pid >= 0) & (len(self.ids) > 0)
        ok[ok] = self.ids[row[ok]] == pid[ok]
        return a["xy"], np.where(ok, row, -1)


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
        self.has_priors = priors_mod.has_poses(self.priors)  # False: video without metadata

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
        try:
            return self.read_snapshot(name)
        except (ValueError, OSError, struct.error):  # unexpected layout: the slow, general way
            rec = pycolmap.Reconstruction(os.path.join(self.path, "sparse", name))
            return self.snapshot_from_reconstruction(name, rec)

    def read_snapshot(self, name):
        """A snapshot straight from the binary files (no pycolmap.Reconstruction): the same numbers as
        snapshot_from_reconstruction, in seconds instead of minutes for thousands of images."""
        d = os.path.join(self.path, "sparse", name)
        cameras = read_cameras_bin(os.path.join(d, "cameras.bin"))
        images = read_images_bin(os.path.join(d, "images.bin"))
        n = len(self.names)
        cams = Cameras.empty(n)
        reproj, nobs = np.full(n, np.nan), np.zeros(n, int)
        kept, blocks = [], {}
        for img_name, cam_id, R_cw, t, off, npts in images:
            pid = read_keypoints(os.path.join(d, "images.bin"), off, npts)
            seen = pid["pid"] >= 0
            kept.append((self.index.get(img_name), cam_id, R_cw, t, pid["xy"][seen], pid["pid"][seen]))
        all_pid = np.concatenate([k[5] for k in kept]) if kept else np.zeros(0, np.int64)
        ids, lengths = np.unique(all_pid, return_counts=True)
        xyz, rgb = read_points3d_bin(os.path.join(d, "points3D.bin"), ids, lengths)
        del all_pid
        err_rows, err_vals = [], []
        for (i, cam_id, R_cw, t, xy, pid), (_, _, _, _, off, npts) in zip(kept, images):
            rows = np.searchsorted(ids, pid)
            err = np.zeros(0)
            if len(rows):
                proj = np.asarray(cameras[cam_id].img_from_cam(xyz[rows] @ R_cw.T + t))
                err = np.linalg.norm(proj - xy, axis=1)
                fin = np.isfinite(err)  # points behind the camera project to NaN before filtering
                rows, err = rows[fin], err[fin]
                err_rows.append(rows)
                err_vals.append(err)
            if i is None:
                continue
            cams.R_wc[i] = R_cw.T
            cams.centers[i] = -R_cw.T @ t
            nobs[i] = len(pid)
            if len(err):
                reproj[i] = float(err.mean())
            blocks[i] = (off, npts)
        # per point: mean reprojection error over its track, as Reconstruction.update_point_3d_errors
        rows = np.concatenate(err_rows) if err_rows else np.zeros(0, np.int64)
        vals = np.concatenate(err_vals) if err_vals else np.zeros(0)
        cnt = np.bincount(rows, minlength=len(ids))
        perr = np.bincount(rows, vals, minlength=len(ids)) / np.maximum(cnt, 1)
        cam = next(iter(cameras.values()))
        intr = {"model": cam.model.name, "width": int(cam.width), "height": int(cam.height),
                "params": [float(v) for v in cam.params]}
        return Snapshot(name, cams, xyz, rgb, perr, reproj, nobs, intr,
                        obs=Observations(os.path.join(d, "images.bin"), blocks, ids))

    def poses_from_reconstruction(self, rec):
        """Registered poses only (cheap enough to call on every BA iteration). rec: a pycolmap.Reconstruction, or
        {"poses": {image name: cam_from_world params}} from a job process (modelforge.worker)."""
        cams = Cameras.empty(len(self.names))
        if isinstance(rec, dict):
            items = rec["poses"].items()
        else:
            items = ((rec.image(i).name, rec.frame(rec.image(i).frame_id).rig_from_world.params)
                     for i in rec.reg_image_ids())
        for name, p in items:
            i = self.index.get(name)
            if i is None:
                continue
            params = np.array(p)
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

    def photo(self, i, width=640):
        """Image i at about `width` px, RGB (JPEG decoded at 1/2, 1/4 or 1/8 where that is still wide enough)."""
        path = os.path.join(self.path, "images", self.names[i])
        from PIL import Image
        with Image.open(path) as im:
            im.draft("RGB", (width, round(width * im.height / im.width)))  # DCT scaling: a 4K photo in ~20 ms
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        return img

    def inspect_image(self, snap, i, width=960, residual_gain=10.0):
        """Image with observed keypoints (green), magnified reprojection residuals (red)
        and keypoints without a 3D point (grey)."""
        img = cv2.cvtColor(cv2.imread(os.path.join(self.path, "images", self.names[i])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = width / w
        img = cv2.resize(img, (width, round(h * s)), interpolation=cv2.INTER_AREA)
        stats = {}
        rec = snap.rec
        if rec is None and snap.obs is not None:
            return self._inspect_from_files(snap, i, img, s, residual_gain)
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

    def _inspect_from_files(self, snap, i, img, s, residual_gain):
        """inspect_image for a snapshot read from disk (Snapshot.obs instead of a Reconstruction)."""
        got = snap.obs.image(i) if snap.cams.valid[i] else None
        if got is None:
            return img, {}
        xy, row = got
        for x, y in xy[row < 0] * s:
            cv2.circle(img, (int(x), int(y)), 1, (150, 150, 150), -1)
        seen = row >= 0
        if not seen.any():
            return img, {}
        intr = snap.intrinsics
        cam = pycolmap.Camera.create_from_model_name(0, intr["model"], 1.0, intr["width"], intr["height"])
        cam.params = intr["params"]
        R_cw = snap.cams.R_wc[i].T
        t = -R_cw @ snap.cams.centers[i]
        u = np.asarray(cam.img_from_cam(snap.points[row[seen]] @ R_cw.T + t))
        errs = np.linalg.norm(u - xy[seen], axis=1)
        for (x, y), (ux, uy) in zip(xy[seen] * s, u * s):
            end = (int(x + residual_gain * (ux - x)), int(y + residual_gain * (uy - y)))
            cv2.line(img, (int(x), int(y)), end, (255, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 2, (60, 220, 60), -1)
        errs = errs[np.isfinite(errs)]
        if not len(errs):
            return img, {}
        return img, {"observations": len(errs), "keypoints": len(xy),
                     "median_err_px": float(np.median(errs)), "p90_err_px": float(np.percentile(errs, 90))}


# ---------------------------------------------------------------- dense / splat
SH_C0 = 0.28209479177387814


def dense_artifacts(path):
    """Which dense / splat outputs exist in the project (paths, or None)."""
    ws = os.path.join(path, "dense")
    found = {"fused": None, "splat": None, "meshes": {}, "textured": {}}
    if os.path.exists(os.path.join(ws, "fused.ply")):
        found["fused"] = os.path.join(ws, "fused.ply")
    for method in ("delaunay", "poisson"):
        for name in (f"preview-{method}.ply", f"meshed-{method}.ply"):  # prefer the simplified copy
            p = os.path.join(ws, name)
            if os.path.exists(p):
                found["meshes"][method] = p
                break
        t = os.path.join(ws, f"textured-{method}", "mesh.ply")
        if os.path.exists(t):
            found["textured"][method] = t
    for name in ("preview.ply", "mesh.ply"):  # the mesh from the splat (vertex colours)
        p = os.path.join(path, "splatmesh", name)
        if os.path.exists(p):
            found["meshes"]["splat"] = p
            break
    # the mesh from the SfM points (Poisson): coloured from the photos if done, else the SfM points' colours
    for name in ("preview_coloured.ply", "mesh_coloured.ply", "preview.ply", "mesh.ply"):
        p = os.path.join(path, "sfmmesh", name)
        if os.path.exists(p):
            found["meshes"]["sfm"] = p
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


MAX_TEXTURED_FACES = 1_500_000  # larger textured meshes fall back to the (grey) simplified preview


def load_textured_mesh(ply_path):
    """COLMAP's textured mesh (texture stage) as a trimesh with UVs, or None if it is too big to stream.
    COLMAP's meshers write geometry only, so this is the only coloured mesh."""
    import plyfile

    from modelforge.export import products
    with open(ply_path, "rb") as f:
        n = plyfile.PlyData.read(f)["face"].count
    return products.load_textured(ply_path) if n <= MAX_TEXTURED_FACES else None


SPLAT_SHOWN = 1_000_000  # a finished splat larger than this is shown as a random subset (View: All gaussians)
LIVE_SPLAT_SHOWN = 500_000  # the same for live training snapshots


def splat_count(ply_path):
    """Number of Gaussians in a splat PLY, from its header."""
    with open(ply_path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    return 0


def load_splat(ply_path, max_gaussians=SPLAT_SHOWN, seed=0):
    """ENU splat (modelforge.export.splat_to_enu output) as viser gaussian-splat arrays (None: all)."""
    import plyfile
    return splat_arrays(plyfile.PlyData.read(ply_path)["vertex"].data, max_gaussians, seed)


def load_live_splat(npz_path, transforms_path, max_gaussians=LIVE_SPLAT_SHOWN, seed=0):
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
    if max_gaussians and n > max_gaussians:  # subsample before the transform: cheaper, same result
        v = v[np.sort(np.random.default_rng(seed).choice(n, max_gaussians, replace=False))]
    R, t, s = splat_to_enu.load_transform(transforms_path)
    return int(d["step"]), n, splat_arrays(splat_to_enu.transform_vertices(v, R, t, s), max_gaussians, seed)


def splat_subset(n, max_gaussians=SPLAT_SHOWN, seed=0):
    """Indices of the Gaussians shown (all, or a reproducible random subset of max_gaussians)."""
    if max_gaussians and n > max_gaussians:
        return np.sort(np.random.default_rng(seed).choice(n, max_gaussians, replace=False))
    return np.arange(n)


def splat_arrays(v, max_gaussians=SPLAT_SHOWN, seed=0):
    """Structured splat vertices (ENU) -> viser centers, covariances, rgbs, opacities (None: all)."""
    idx = splat_subset(len(v), max_gaussians, seed)
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

"""Frame extraction from ROS 2 bags: project kind "rosbag".

Same plan / extract interface as modelforge.dji.ingest (load_clip, make_plan, preview, run). A source
is a bag (folder with metadata.yaml, or an .mcap file) plus a time range; Settings.ros holds:
    image_topic   sensor_msgs/CompressedImage or Image
    info_topic    sensor_msgs/CameraInfo ("" = none: intrinsics are estimated in BA)
    pose_source   "tf": T_world_camera from the TF tree at each image stamp, or
                  "odometry": nav_msgs/Odometry (frame -> child) chained with TF on both ends,
                  T_world_cam = T_world_odomframe * T_odomframe_child(t) * T_child_cam
    world_frame   the frame poses are expressed in (it becomes the project's world)
    camera_frame  the camera's optical frame (x right, y down, z forward); "" = the image header frame
    odom_topic    for pose_source "odometry"
    cloud_topics  sensor_msgs/PointCloud2 topics: transformed into the world frame at their stamps,
                  merged and voxel-downsampled into PROJECT/lidar/cloud.ply
    voxel         LiDAR voxel size [m]; max_range [m] drops far returns (in the sensor frame)
Images are spaced by travelled distance + rotation credit (as for DJI) or by time; the sharpest
frame per window is kept. poses.json carries priors in the DJI format (world coordinates in the
enu_* fields), so the prior-based refine runs unchanged; camera_info becomes PROJECT/camera.json
(fixed intrinsics in BA).
"""

import csv
import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from modelforge.dji import extract as ex
from modelforge.dji import ingest as dji
from modelforge.rosbag import bag
from modelforge.rosbag.tf import inv, make, quat_to_rot, rot_to_quat, slerp

KIND = "rosbag"
SAMPLING = ("distance", "time")
DEFAULT_ROT_WEIGHT = 0.15  # m per degree of view rotation (~9 m look-at depth: ground robots, handheld rigs)


def default_settings():
    return dji.Settings(sampling="distance", num_images=200, ros=None)


def ros_defaults(path):
    """Settings.ros guessed from a bag: the first compressed (else raw) image topic, its CameraInfo,
    the TF root above the camera as world, TF poses if they reach the camera, else odometry."""
    inv_ = bag.inventory(path)
    imgs = bag.topics_of(inv_, bag.IMAGE_TYPES)
    comp = [t for t in imgs if inv_["topics"][t]["type"].endswith("CompressedImage")]
    image = (comp or imgs or [""])[0]
    infos = bag.topics_of(inv_, bag.INFO_TYPE)
    info = next((t for t in infos if os.path.dirname(t) in image), infos[0] if infos else "")
    cam = inv_["frame_of"].get(image, "")
    buf = bag.tf_buffer(path)
    children = {c for _, c in list(buf.static) + list(buf.dynamic)}
    roots = [f for f in buf.frames if f not in children and cam and buf.path(f, cam)]
    odoms = bag.topics_of(inv_, bag.ODOM_TYPE)
    world = roots[0] if roots else (inv_["frame_of"].get(odoms[0], "") if odoms else "")
    dynamic_to_cam = bool(roots) and any((a, b) not in buf.static and (b, a) not in buf.static
                                         for a, b, _ in (buf.path(world, cam) or []))
    source = "tf" if dynamic_to_cam or not odoms else "odometry"
    return {"image_topic": image, "info_topic": info, "pose_source": source, "world_frame": world,
            "camera_frame": "", "odom_topic": odoms[0] if odoms else "",
            "cloud_topics": bag.topics_of(inv_, bag.CLOUD_TYPE), "voxel": 0.05, "max_range": 60.0}


def _ros(settings, path):
    r = dict(ros_defaults(path))
    r.update({k: v for k, v in (settings.ros or {}).items() if v is not None})
    return r


class Poser:
    """T_world_<frame>(t) for any frame, through TF, or through odometry + TF."""

    def __init__(self, path, ros):
        self.buf, self.ros = bag.tf_buffer(path), ros
        self.world = ros["world_frame"]
        self.odom = None
        if ros["pose_source"] == "odometry":
            if not ros.get("odom_topic"):
                raise ValueError("pose source odometry needs an odometry topic")
            self.odom = bag.odometry(path, ros["odom_topic"])

    def _odom_at(self, t):
        ts, Ts, _, _ = self.odom
        if len(ts) == 0 or t < ts[0] - 0.1 or t > ts[-1] + 0.1:
            return None
        i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
        a = float(np.clip((t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9), 0, 1))
        q = slerp(rot_to_quat(Ts[i - 1][:3, :3]), rot_to_quat(Ts[i][:3, :3]), a)
        return make(quat_to_rot(q), Ts[i - 1][:3, 3] + a * (Ts[i][:3, 3] - Ts[i - 1][:3, 3]))

    def world_from(self, frame, t):
        if self.odom is None:
            return self.buf.lookup(self.world, frame, t)
        _, _, oframe, child = self.odom
        T_wo = self.buf.lookup(self.world, oframe, t) if self.world != oframe else np.eye(4)
        T_oc = self._odom_at(t)
        T_cf = self.buf.lookup(child, frame, t) if frame != child else np.eye(4)
        if T_wo is None or T_oc is None or T_cf is None:
            return None
        return T_wo @ T_oc @ T_cf


class Clip:
    """One bag's image topic with the camera pose at every image (NaN where it cannot be looked up)."""

    def __init__(self, path, ros):
        self.video, self.ros = path, ros
        self.topic = ros["image_topic"]
        if not self.topic:
            raise ValueError("the bag has no image topic")
        self.stamps = bag.image_stamps(path, self.topic)
        self.n = len(self.stamps)
        self.pts = self.stamps - self.stamps[0] if self.n else np.zeros(0)
        self.duration = float(self.pts[-1]) if self.n else 0.0
        self.fps = (self.n - 1) / self.duration if self.duration > 0 else 1.0
        first = next(bag.iter_images(path, self.topic, [0]))[2] if self.n else np.zeros((1, 1, 3), np.uint8)
        self.height, self.width = first.shape[:2]
        self.wall_start = float(self.stamps[0]) if self.n else None
        self.camera_frame = ros.get("camera_frame") or bag.inventory(path)["frame_of"].get(self.topic, "")
        self.lead, self.srt_path, self.takeoff_abs_alt = 0.0, None, None
        self.align = {"method": ros["pose_source"], "world_frame": ros["world_frame"],
                      "camera_frame": self.camera_frame}
        poser = Poser(path, ros)
        self.T = np.full((self.n, 4, 4), np.nan)
        for k, t in enumerate(self.stamps):
            T = poser.world_from(self.camera_frame, t)
            if T is not None:
                self.T[k] = T
        self.valid_pose = np.isfinite(self.T[:, 0, 0])
        self.centers = self.T[:, :3, 3]
        self.axes = self.T[:, :3, 2]  # optical axis (z) in the world

    def frame_image(self, t, width=640):
        k = int(np.clip(np.searchsorted(self.pts, t), 0, max(self.n - 1, 0)))
        rgb = next(bag.iter_images(self.video, self.topic, [k]))[2]
        h = int(round(rgb.shape[0] * width / rgb.shape[1] / 2)) * 2
        return cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA)


_CLIPS = {}


def load_clip(video, lead=None, srt=None, ros=None):
    ros = ros or ros_defaults(video)
    key = (os.path.abspath(video), os.path.getmtime(video), json.dumps(ros, sort_keys=True))
    if key not in _CLIPS:
        _CLIPS[key] = Clip(video, ros)
    return _CLIPS[key]


@dataclass
class SourcePlan:
    index: int
    source: dji.Source
    clip: Clip
    prefix: str
    valid: np.ndarray
    s: np.ndarray
    windows: list
    report: dict
    alt_offset: float = 0.0
    time_base: float = 0.0

    def describe(self):
        return {"video": self.source.video, "prefix": self.prefix, "start": self.source.start, "end": self.source.end,
                "duration_s": round(self.clip.duration, 3), "fps": round(self.clip.fps, 4),
                "size": [self.clip.width, self.clip.height], "alignment": self.clip.align,
                "time_base_s": round(self.time_base, 3), "sampling": self.report}


@dataclass
class Plan:
    sources: list
    report: dict
    warnings: list = field(default_factory=list)

    @property
    def num_images(self):
        return sum(len(sp.windows) for sp in self.sources)


def _curve(clip, mode, w):
    if mode == "time":
        return clip.pts.copy()
    c = clip.centers.copy()
    ok = clip.valid_pose
    if not ok.any():
        return np.zeros(clip.n)
    idx = np.flatnonzero(ok)  # hold the last valid pose across gaps
    fill = np.searchsorted(idx, np.arange(clip.n), side="right") - 1
    c = c[idx[np.clip(fill, 0, len(idx) - 1)]]
    ax = clip.axes[idx[np.clip(fill, 0, len(idx) - 1)]]
    dp = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)]
    da = np.r_[0.0, np.degrees(np.arccos(np.clip(np.sum(ax[1:] * ax[:-1], 1), -1, 1)))]
    return np.cumsum(dp + w * da)


def make_plan(sources, settings):
    enabled = [(i, s) for i, s in enumerate(sources) if s.enabled]
    if not enabled:
        return Plan([], {}, ["no enabled sources"])
    mode = settings.sampling if settings.sampling in SAMPLING else "distance"
    w = settings.rot_weight if settings.rot_weight is not None else DEFAULT_ROT_WEIGHT
    clips = {i: load_clip(s.video, ros=_ros(settings, s.video)) for i, s in enabled}
    enabled.sort(key=lambda x: (clips[x[0]].wall_start or 0.0) + x[1].start)
    warnings, curves, valids = [], {}, {}
    for i, s in enabled:
        c = clips[i]
        end = c.duration if s.end is None else s.end
        valids[i] = (c.pts >= s.start) & (c.pts <= end) & c.valid_pose
        curves[i] = _curve(c, mode, w)
        if not c.valid_pose.any():
            warnings.append(f"{os.path.basename(s.video)}: no camera pose at any image (world frame "
                            f"'{c.ros['world_frame']}' not connected to '{c.camera_frame}' at the image stamps)")
        elif not c.valid_pose.all():
            warnings.append(f"{os.path.basename(s.video)}: {int((~c.valid_pose).sum())} of {c.n} images have no pose "
                            "(outside the TF / odometry time range) and are skipped")
    total = sum(float(curves[i][valids[i]][-1] - curves[i][valids[i]][0]) for i, _ in enabled if valids[i].any())
    spacing = settings.spacing or (total / settings.num_images if total > 0 else 1.0)
    prefixes = dji._prefixes([s.video for _, s in enabled])
    t_first = min(clips[i].wall_start or 0.0 for i, _ in enabled)
    plans = []
    for k, (i, s) in enumerate(enabled):
        c = clips[i]
        windows, rep = ex.plan_windows(curves[i], valids[i], spacing=spacing, window=settings.window) \
            if valids[i].any() else ([], {})
        if mode == "distance" and valids[i].any():
            v = valids[i]
            rep["path_m"] = float(np.nansum(np.linalg.norm(np.diff(c.centers[v], axis=0), axis=1)))
        plans.append(SourcePlan(k, s, c, prefixes[s.video], valids[i], curves[i], windows,
                                {**rep, "num_images": len(windows)}, 0.0, (c.wall_start or 0.0) - t_first))
    sizes = {(c.width, c.height) for c in clips.values()}
    if len(sizes) > 1:
        warnings.append(f"sources differ in resolution {sorted(sizes)}: one project = one camera")
    report = {"mode": mode, "unit": "m" if mode == "distance" else "s", "rot_weight_m_per_deg": w,
              "total": round(total, 3), "spacing": round(spacing, 4), "num_images": sum(len(p.windows) for p in plans),
              "window": settings.window, "sources": len(plans)}
    return Plan(plans, report, warnings)


def preview(plan, *args, **kwargs):
    """Camera tracks (bright inside the range) and planned positions for the GUI's 3D view."""
    out = []
    for sp in plan.sources:
        c = sp.clip
        ok = np.flatnonzero(c.valid_pose)
        track = c.centers.copy()
        if len(ok):
            fill = np.clip(np.searchsorted(ok, np.arange(c.n), side="right") - 1, 0, len(ok) - 1)
            track = c.centers[ok[fill]]
        else:
            track = np.zeros((max(c.n, 1), 3))
        k = [(a + b) // 2 for a, b in sp.windows]
        out.append({"track": track, "track_valid": sp.valid, "samples": track[k] if k else np.zeros((0, 3))})
    return {"origin": None, "sources": out}


def _sharpest(clip, windows):
    """(frame, rgb, sharpness) per window: every image inside a window is decoded once."""
    inside = sorted({k for a, b in windows for k in range(a, b + 1)})
    best = {}
    for k, _, rgb in bag.iter_images(clip.video, clip.topic, inside):
        wi = next(j for j, (a, b) in enumerate(windows) if a <= k <= b)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(g, (max(1, g.shape[1] // 4), max(1, g.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
        if wi not in best or sharp > best[wi][2]:
            best[wi] = (k, rgb, sharp)
    for wi in sorted(best):
        yield best[wi]


def accumulate_clouds(path, ros, t0, t1, log=print, max_scans=400):
    """LiDAR returns of every cloud topic in [t0, t1] (bag time), in the world frame, voxel-downsampled.
    Returns (xyz (N, 3), intensity (N,) or None, per-topic stats)."""
    poser = Poser(path, ros)
    inv_ = bag.inventory(path)
    pts, its, stats = [], [], {}
    for topic in ros.get("cloud_topics") or []:
        n_msgs = inv_["topics"].get(topic, {}).get("count", 0)
        every = max(1, int(np.ceil(n_msgs / max_scans)))
        used = skipped = 0
        for t, frame, xyz, inten in bag.iter_clouds(path, topic, t0, t1, every):
            T = poser.world_from(frame, t)
            if T is None:
                skipped += 1
                continue
            keep = np.linalg.norm(xyz, axis=1) <= ros.get("max_range", 60.0)
            pts.append(xyz[keep] @ T[:3, :3].T + T[:3, 3])
            its.append(inten[keep] if inten is not None else np.zeros(int(keep.sum()), np.float32))
            used += 1
        stats[topic] = {"scans": used, "skipped_no_pose": skipped, "every": every}
        log(f"lidar {topic}: {used} scans" + (f", {skipped} without a pose" if skipped else ""))
    if not pts:
        return np.zeros((0, 3)), None, stats
    xyz, inten = np.concatenate(pts), np.concatenate(its)
    v = float(ros.get("voxel", 0.05))
    if v > 0:  # one point per voxel (the first)
        _, first = np.unique(np.floor(xyz / v).astype(np.int64), axis=0, return_index=True)
        xyz, inten = xyz[first], inten[first]
    return xyz, inten, stats


def write_cloud(path, xyz, inten):
    from plyfile import PlyData, PlyElement
    v = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    v["intensity"] = inten if inten is not None else 0.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(path)


def run(project, sources, settings, progress=None, overwrite=False, log=print):
    """Extract images + pose priors (+ camera.json, + lidar/cloud.ply) for the plan."""
    plan = make_plan(sources, settings)
    sizes = {(sp.clip.width, sp.clip.height) for sp in plan.sources}
    if len(sizes) > 1:
        raise ValueError(f"sources differ in resolution {sorted(sizes)}; one project = one camera")
    if not plan.num_images:
        raise ValueError("the plan selects no images: " + "; ".join(plan.warnings))
    if dji.has_outputs(project):
        if not overwrite:
            raise FileExistsError(f"{project} already has images; re-extract with overwrite "
                                  "(old outputs are moved to previous/)")
        log(f"previous outputs moved to {dji.archive_outputs(project)}")
    for name in ("camera.json", "lidar"):  # derived from the bag: replaced on every extraction
        p = os.path.join(project, name)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    dji.save_plan(project, sources, settings, kind=KIND)
    os.makedirs(os.path.join(project, "images"), exist_ok=True)
    for w in plan.warnings:
        log(f"warning: {w}")
    log(f"sampling: {plan.report}")
    ros0 = plan.sources[0].clip.ros
    camera = bag.camera_info(plan.sources[0].source.video, ros0["info_topic"]) if ros0.get("info_topic") else None
    if camera:
        w, h = plan.sources[0].clip.width, plan.sources[0].clip.height
        if (camera["width"], camera["height"]) != (w, h):  # e.g. a resized image topic: scale the intrinsics
            sx, sy = w / camera["width"], h / camera["height"]
            p = camera["params"]
            camera["params"] = [p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy] + p[4:]
            camera["width"], camera["height"] = w, h
            log(f"camera_info is {camera['width']}x{camera['height']} for {w}x{h} images: intrinsics scaled")
        json.dump(camera, open(os.path.join(project, "camera.json"), "w"), indent=1)
    total, records, seen = plan.num_images, [], set()
    progress = progress or (lambda done, total, msg: None)
    for sp in plan.sources:
        c = sp.clip
        log(f"{sp.prefix}: {len(sp.windows)} images, poses from {c.ros['pose_source']} in '{c.ros['world_frame']}'")
        for k, rgb, sharp in _sharpest(c, sp.windows):
            name = f"{sp.prefix}_{k:05d}.jpg"
            if name in seen:
                continue
            seen.add(name)
            Image.fromarray(rgb).save(os.path.join(project, "images", name), quality=settings.quality)
            R, p = c.T[k][:3, :3], c.T[k][:3, 3]
            ok = np.flatnonzero(c.valid_pose)
            a, b = ok[max(0, np.searchsorted(ok, k) - 1)], ok[min(len(ok) - 1, np.searchsorted(ok, k) + 1)]
            vel = (c.centers[b] - c.centers[a]) / max(c.stamps[b] - c.stamps[a], 1e-6) if b != a else np.zeros(3)
            Rcw = R.T
            records.append({
                "image": name, "source": sp.index, "video": sp.prefix, "frame": k, "pts": float(c.pts[k]),
                "time_s": float(sp.time_base + c.pts[k]), "stamp": float(c.stamps[k]), "s": float(sp.s[k]),
                "sharpness": round(sharp, 2),
                "enu_x": float(p[0]), "enu_y": float(p[1]), "enu_z": float(p[2]),
                "vel_e": float(vel[0]), "vel_n": float(vel[1]), "vel_d": float(-vel[2]),
                "q_world_from_cam": [float(x) for x in np.roll(rot_to_quat(R), 1)],
                "colmap_qvec": [float(x) for x in np.roll(rot_to_quat(Rcw), 1)],
                "colmap_tvec": [float(x) for x in -Rcw @ p],
            })
            progress(len(records), total, f"{sp.prefix} t = {c.pts[k]:.1f} s")
    lidar = {}
    xyz = np.zeros((0, 3))
    for sp in plan.sources:  # LiDAR over each source's range (bag time)
        c, s = sp.clip, sp.source
        if not (c.ros.get("cloud_topics")):
            continue
        t0 = c.stamps[0] + s.start
        t1 = c.stamps[0] + (s.end if s.end is not None else c.duration)
        pts, inten, st = accumulate_clouds(s.video, c.ros, t0, t1, log)
        lidar[sp.prefix] = st
        xyz = np.concatenate([xyz, pts])
        if len(pts):
            prev = None
            path = os.path.join(project, "lidar", "cloud.ply")
            if os.path.exists(path):
                from plyfile import PlyData
                prev = PlyData.read(path)["vertex"]
            if prev is not None:
                pts = np.concatenate([np.c_[prev["x"], prev["y"], prev["z"]], pts])
                inten = np.concatenate([np.asarray(prev["intensity"]), inten])
            write_cloud(path, pts, inten)
    cams = np.array([[r["enu_x"], r["enu_y"], r["enu_z"]] for r in records])
    ground_z = float(np.percentile(xyz[:, 2], 2)) if len(xyz) else float(cams[:, 2].min() - 1.5)
    out = {
        "kind": KIND,
        "sources": [sp.describe() for sp in plan.sources],
        "source": os.path.abspath(plan.sources[0].source.video),
        "sampling": plan.report,
        "warnings": plan.warnings,
        "plan": {"sources": [asdict(x) for x in sources], "settings": asdict(settings)},  # as in ingest.json
        "ros": ros0,
        "camera": camera,
        "lidar": {"points": int(len(xyz)), "topics": lidar, "path": "lidar/cloud.ply" if len(xyz) else None},
        "ground_z": ground_z,
        "enu_origin_lla": None,
        "conventions": {
            "world": f"the bag's '{ros0['world_frame']}' frame [m] (the enu_* fields hold world x, y, z)",
            "camera": "OpenCV optical (x right, y down, z forward), as the ROS *_optical_frame convention",
            "q_world_from_cam": "(w,x,y,z), rotates camera-frame vectors into the world frame",
            "colmap_qvec_tvec": "cam_from_world, as in COLMAP images.txt",
            "velocity": "finite differences of the camera track; vel_d = -z velocity",
            "time_s": "project timeline: seconds since the first source's first image",
        },
        "images": records,
    }
    json.dump(out, open(os.path.join(project, "poses.json"), "w"), indent=1)
    with open(os.path.join(project, "poses.csv"), "w", newline="") as f:
        if records:
            flat = [{k: (" ".join(f"{x:.9g}" for x in v) if isinstance(v, list) else v) for k, v in r.items()}
                    for r in records]
            wr = csv.DictWriter(f, fieldnames=list(flat[0]))
            wr.writeheader()
            wr.writerows(flat)
    log(f"wrote {len(records)} images" + (f", {len(xyz):,} LiDAR points" if len(xyz) else "") + f" to {project}")
    return out

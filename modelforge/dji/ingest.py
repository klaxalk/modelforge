"""Multi-video ingest: sources (a DJI video + a time range) -> one project folder.

A project's ingest plan (PROJECT/ingest.json) lists sources plus one set of sampling
settings. A source is a video and a time range [start, end]; the same video may appear
several times (several pieces of one clip). `run` writes the selected frames of all
sources into PROJECT/images and one poses.json, in one ENU frame:

- sources are ordered chronologically (UTC start: MP4 creation_time, refined to ms by the SRT
  when there is one; else list order). The first one
  is the reference: ENU origin = its first image, z from its rel_alt (above takeoff);
- other flights are lifted by the difference of their takeoff altitudes,
  median(abs_alt - rel_alt), which is constant within a flight (-31.011 m on the sample
  clip, spread 0.00001 m). Pieces of one flight therefore get 0;
- the sampling step is shared: S_total over all sources / N, so images spread in
  proportion to each piece's travelled distance + view rotation, and the rotation
  credit w is one value for the whole project;
- images are named <video stem>_<frame>.jpg, and each record carries `source` (index
  into poses.json["sources"]). Relative priors and time-neighbour pairs never cross
  sources; refinement estimates a per-source position offset (GNSS bias per flight).

All sources must share one resolution, because they share one COLMAP camera.

Usage: python -m modelforge.dji.extract A.MP4 [B.MP4 ...] -o PROJECT, or the GUI Ingest tab.
"""

import csv
import datetime
import glob
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from modelforge.dji import extract as ex
from modelforge.dji.djmd import decode
from modelforge.dji.srt import parse_srt

PLAN_FILE = "ingest.json"
# Everything derived from the images; moved to PROJECT/previous/<time>/ on re-extraction.
OUTPUTS = ("images", "poses.json", "poses.csv", "features.db", "database.db", "database.db-wal", "database.db-shm",
           "pairs.txt", "sparse", "dense", "transforms.json", "sparse_pc.ply", "report.json", "splat", "nerfstudio",
           "export", "images_*", "splatmesh",  # images_N: downscaled frames for splat training
           "extracted",  # all extracted frames while a subset is in use (projects.apply_subsample)
           "analysis")  # their overlap analysis (sfm/overlap.py)
VIDEO_EXTS = (".mp4", ".mov")


@dataclass
class Source:
    video: str
    start: float = 0.0
    end: float = None  # None = end of video
    enabled: bool = True
    lead: float = None  # telemetry lead [frames]; None = estimate
    srt: str = None  # None = next to the video


@dataclass
class Settings:
    sampling: str = "distance"  # or "time"
    num_images: int = 300
    spacing: float = None  # m (distance) or s (time); overrides num_images
    rot_weight: float = None  # m/deg; None = median look-at depth * pi/180
    window: float = 0.5
    yaw_source: str = "auto"  # auto: per clip, the gimbal when its yaw is a true heading (Clip.yaw_source)
    quality: int = 95
    ros: dict = None  # kind "rosbag": topics, frames, pose source (modelforge.rosbag.ingest.ros_defaults)


def load_plan(project):
    path = os.path.join(project, PLAN_FILE)
    if not os.path.exists(path):
        return [], Settings()
    d = json.load(open(path))
    return [Source(**s) for s in d.get("sources", [])], Settings(**d.get("settings", {}))


KINDS = {"dji": "DJI drone video (embedded position + attitude)",
         "video": "video without metadata (phone, action camera, ...)",
         "rosbag": "ROS 2 bag (images + TF / odometry + optional LiDAR)"}


def load_kind(project):
    """The project kind from ingest.json ("dji" when missing: projects made before kinds existed)."""
    path = os.path.join(project, PLAN_FILE)
    try:
        return json.load(open(path)).get("kind", "dji")
    except (OSError, ValueError):
        return "dji"


def save_plan(project, sources, settings, kind=None):
    """Write ingest.json; kind None keeps the project's current kind."""
    os.makedirs(project, exist_ok=True)
    kind = kind or load_kind(project)
    json.dump({"kind": kind, "sources": [asdict(s) for s in sources], "settings": asdict(settings)},
              open(os.path.join(project, PLAN_FILE), "w"), indent=1)


def find_srt(video):
    base = os.path.splitext(video)[0]
    return next((base + e for e in (".SRT", ".srt") if os.path.exists(base + e)), None)


def find_videos(root):
    """Videos under root, skipping project folders (anything with a poses.json or ingest.json)."""
    out = []
    for d, dirs, files in os.walk(root):
        if any(f in files for f in ("poses.json", PLAN_FILE)):
            dirs[:] = []
            continue
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
    return out


# --- one analysed video ---------------------------------------------------------

NO_YAW_AXIS = ("neo",)  # model names (lower case) whose gimbal cannot yaw
YAWING_GIMBAL = ("mavic4", "mavic 4")  # model names whose gimbal yaw is the camera's true heading


class Clip:
    """Decoded djmd telemetry of one video, its lead and per-frame aligned telemetry."""

    def __init__(self, video, srt_path=None, lead=None):
        self.video = video
        meta = decode(video)
        from modelforge.dji.djmd import model_name
        self.model = model_name(video)
        self.n = len(meta)
        self.pts = np.array([m.pts for m in meta])
        self.fps = (self.n - 1) / (self.pts[-1] - self.pts[0]) if self.n > 1 else 29.97
        self.duration = float(self.pts[-1]) if self.n else 0.0
        self.width, self.height = ex.video_size(video)
        self.aircraft, self.gimbal = ex.build_streams(meta)
        self.srt_path = srt_path or find_srt(video)
        srt = parse_srt(self.srt_path) if self.srt_path else None
        self.wall_start = _wall_start(srt, video)
        if lead is not None:
            self.align = {"method": "manual", "lead_frames": lead}
        else:
            self.align = (ex.estimate_lead_srt(srt, self.gimbal, self.n) if srt else None) \
                or _cached_flow_lead(video, self.gimbal, self.n, self.pts) \
                or {"method": "default", "lead_frames": ex.DEFAULT_LEAD_FRAMES}
        self.lead = self.align["lead_frames"]
        self.align["lead_s"] = round(self.lead / self.fps, 3)
        d = np.array([m.abs_alt_m - m.rel_alt_m for m in meta])
        # constant within a flight when abs_alt is present; otherwise unusable
        self.takeoff_abs_alt = float(np.median(d)) if len(d) and np.ptp(np.percentile(d, [5, 95])) < 1.0 else None
        self._tel = {}

    def yaw_source(self, setting="auto"):
        """The camera heading source for this clip, "aircraft" or "gimbal". "auto" decides by the model
        whether the gimbal can yaw (its yaw is then the camera's own heading): the NEO / NEO2 gimbal has
        no yaw axis and reports yaw in an offset reference (aircraft yaw - 36.5 deg), the Mavic 4's gimbal
        turns 360 deg. Unknown models: the gimbal unless its yaw sits over 20 deg from the aircraft yaw
        for the whole clip (an offset reference). The data alone cannot tell: on the Mavic 4 clips the
        difference was 1.4-5.1 deg with the same ~1 deg spread as the NEO2's offset."""
        if setting != "auto":
            return setting
        if getattr(self, "_auto_yaw", None) is None:
            model = (getattr(self, "model", "") or "").lower()
            if any(k in model for k in NO_YAW_AXIS):
                self._auto_yaw = "aircraft"
            elif any(k in model for k in YAWING_GIMBAL):
                self._auto_yaw = "gimbal"
            else:
                x = np.arange(self.n)
                d = (self.gimbal(x)[:, 2] - self.aircraft(x)[:, 9] + 180.0) % 360.0 - 180.0
                d = d[np.isfinite(d)]
                self._auto_yaw = "gimbal" if len(d) and abs(float(np.median(d))) < 20.0 else "aircraft"
        return self._auto_yaw

    def telemetry(self, yaw_source="auto"):
        """Aligned per-frame arrays: aircraft (n, 10), gimbal (n, 3), camera yaw/pitch, lead-valid mask."""
        yaw_source = self.yaw_source(yaw_source)
        if yaw_source not in self._tel:
            x = np.clip(np.arange(self.n) - self.lead, 0, self.n - 1)  # a negative lead reads ahead
            a, g = self.aircraft(x), self.gimbal(x)
            yaw = g[:, 2] if yaw_source == "gimbal" else a[:, 9]
            self._tel[yaw_source] = {"a": a, "g": g, "yaw": yaw, "pitch": g[:, 0],
                                     "valid": (np.arange(self.n) - self.lead >= 0)
                                     & (np.arange(self.n) - self.lead <= self.n - 1)}
        return self._tel[yaw_source]

    def frame_image(self, t, width=640):
        """RGB frame at video time t (accurate seek), scaled to `width`."""
        h = int(round(self.height * width / self.width / 2)) * 2
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", self.video, "-map", "0:v:0",
                              "-frames:v", "1", "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        if len(out) < width * h * 3:
            return np.zeros((h, width, 3), np.uint8)
        return np.frombuffer(out[:width * h * 3], np.uint8).reshape(h, width, 3)


def _mp4_creation_time(video):
    """The MP4 creation_time tag (UTC, whole seconds) as POSIX seconds, or None."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of",
                              "default=nw=1:nk=1", video], capture_output=True, text=True, check=True).stdout.strip()
        return datetime.datetime.fromisoformat(out.replace("Z", "+00:00")).timestamp() if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _wall_start(srt, video):
    """UTC POSIX seconds of the first frame, so clips with and without an SRT share one clock.
    The SRT timestamp is local time (ms resolution), the MP4 creation_time is UTC (1 s resolution):
    the SRT's time-zone offset is taken from their difference (rounded to 15 min) and removed."""
    s = next((e["datetime"] for e in (srt or [])[:5] if "datetime" in e), None)
    local = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp() if s else None
    utc = _mp4_creation_time(video)
    if local is not None and utc is not None:
        return local - round((local - utc) / 900.0) * 900.0
    return utc if utc is not None else local


LEAD_CACHE = ".modelforge_leads.json"  # per video folder: optical-flow leads, keyed by file size + mtime


def _cached_flow_lead(video, gimbal, n, pts=None):
    """ex.estimate_lead_flow (about a minute per clip) remembered on disk, so a clip without an SRT
    pays it once. The key (size, mtime) survives renames; a failed estimate is remembered too."""
    st = os.stat(video)
    key = f"{st.st_size}:{st.st_mtime_ns}"
    path = os.path.join(os.path.dirname(os.path.abspath(video)), LEAD_CACHE)
    try:
        cache = json.load(open(path))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    align = ex.estimate_lead_flow(video, gimbal, n, pts=pts)
    if align is not None:
        align = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in align.items()}
    try:
        cache[key] = align
        with open(path, "w") as f:
            json.dump(cache, f, indent=1)
    except OSError:  # read-only folder: just not cached
        pass
    return align


_CLIPS = {}


def load_clip(video, lead=None, srt=None):
    """Cached Clip (decoding takes a few seconds per video)."""
    st = os.stat(video)
    key = (os.path.abspath(video), st.st_size, st.st_mtime, lead, srt)
    if key not in _CLIPS:
        _CLIPS[key] = Clip(video, srt, lead)
    return _CLIPS[key]


# --- planning --------------------------------------------------------------------

@dataclass
class SourcePlan:
    index: int  # position in chronological order = poses.json source index
    source: Source
    clip: Clip
    prefix: str
    valid: np.ndarray
    s: np.ndarray
    windows: list
    report: dict
    alt_offset: float = 0.0
    time_base: float = 0.0  # timeline offset [s] of this video's t = 0

    def describe(self):
        return {"video": self.source.video, "prefix": self.prefix, "srt": self.clip.srt_path,
                "start": self.source.start, "end": self.source.end, "duration_s": round(self.clip.duration, 3),
                "fps": round(self.clip.fps, 4), "size": [self.clip.width, self.clip.height],
                "alignment": self.clip.align, "takeoff_abs_alt_m": self.clip.takeoff_abs_alt,
                "alt_offset_m": round(self.alt_offset, 3), "time_base_s": round(self.time_base, 3),
                "sampling": self.report}


@dataclass
class Plan:
    sources: list  # SourcePlan, chronological
    report: dict
    warnings: list = field(default_factory=list)

    @property
    def num_images(self):
        return sum(len(sp.windows) for sp in self.sources)


def _prefixes(videos):
    out, used = {}, set()
    for v in videos:
        if v in out:
            continue
        p = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(v))[0]) or "video"
        q, k = p, 2
        while q in used:
            q, k = f"{p}_{k}", k + 1
        out[v] = q
        used.add(q)
    return out


def make_plan(sources, settings):
    enabled = [(i, s) for i, s in enumerate(sources) if s.enabled]
    clips = {i: load_clip(s.video, s.lead, s.srt) for i, s in enabled}
    enabled.sort(key=lambda x: (clips[x[0]].wall_start + x[1].start if clips[x[0]].wall_start is not None
                                else math.inf, x[0]))
    warnings = []
    if not enabled:
        return Plan([], {}, ["no enabled sources"])
    sizes = {(clips[i].width, clips[i].height) for i, _ in enabled}
    if len(sizes) > 1:
        warnings.append(f"sources differ in resolution {sorted(sizes)}: they must share one camera")
    prefixes = _prefixes([os.path.abspath(s.video) for _, s in enabled])

    valids = {}
    for i, s in enabled:
        c = clips[i]
        end = s.end if s.end is not None else math.inf
        valids[i] = c.telemetry(settings.yaw_source)["valid"] & (c.pts >= s.start) & (c.pts <= end)
        if not valids[i].any():
            warnings.append(f"{s.video} [{s.start}, {s.end}]: no frames in range")
    w = settings.rot_weight
    if settings.sampling == "distance" and w is None:
        depth = [look_at_depth(clips[i].telemetry(settings.yaw_source))[valids[i]] for i, _ in enabled]
        depth = np.concatenate(depth)
        w = float(np.median(depth)) * math.pi / 180 if len(depth) else 0.0
    curves = {}
    for i, s in enabled:
        c = clips[i]
        curves[i] = ex.sampling_curve(settings.sampling, c.pts, c.aircraft, c.gimbal, c.lead,
                                      c.yaw_source(settings.yaw_source), valids[i], w)
    totals = {i: float(curves[i][0][valids[i]][-1] - curves[i][0][valids[i]][0]) if valids[i].any() else 0.0
              for i, _ in enabled}
    total = sum(totals.values())
    spacing = settings.spacing or (total / settings.num_images if total > 0 else 1.0)

    ref = clips[enabled[0][0]]
    walls = [clips[i].wall_start for i, _ in enabled]
    plans, cum = [], 0.0
    bases = {}
    for k, (i, s) in enumerate(enabled):
        c = clips[i]
        s_curve, rep = curves[i]
        windows, wrep = ex.plan_windows(s_curve, valids[i], spacing=spacing, window=settings.window) \
            if valids[i].any() else ([], {})
        rep = {**rep, **wrep, "num_images": len(windows)}
        alt = 0.0
        if c is not ref:
            if c.takeoff_abs_alt is not None and ref.takeoff_abs_alt is not None:
                alt = c.takeoff_abs_alt - ref.takeoff_abs_alt
            else:
                warnings.append(f"{s.video}: no usable abs_alt, assuming the same takeoff height as the reference")
        key = os.path.abspath(s.video)
        if key not in bases:  # one timeline offset per video
            if all(x is not None for x in walls):
                bases[key] = c.wall_start - walls[0]
            else:
                bases[key] = cum
                cum += c.duration
        plans.append(SourcePlan(k, s, c, prefixes[key], valids[i], s_curve, windows, rep, alt, bases[key]))
    report = {"mode": settings.sampling, "unit": "m" if settings.sampling == "distance" else "s",
              "rot_weight_m_per_deg": round(w, 4) if w is not None else None, "total": round(total, 2),
              "spacing": round(spacing, 4), "num_images": sum(len(p.windows) for p in plans),
              "window": settings.window, "sources": len(plans)}
    if settings.sampling == "distance":
        report["path_m"] = round(sum(p.report.get("path_m", 0.0) for p in plans), 1)
        report["rotation_deg"] = round(sum(p.report.get("rotation_deg", 0.0) for p in plans), 1)
    return Plan(plans, report, warnings)


def look_at_depth(tel):
    """Distance to flat ground at takeoff height along the optical axis, clipped to [5, 200] m."""
    return np.clip(tel["a"][:, 2] / np.maximum(np.sin(np.radians(-tel["pitch"])), 1e-3), 5.0, 200.0)


def enu(lat, lon, alt, origin):
    """Vectorized WGS84 -> local ENU (rows = points)."""
    p = ex.lla_to_ecef(np.asarray(lat), np.asarray(lon), np.asarray(alt))
    lat0, lon0 = np.radians(origin[0]), np.radians(origin[1])
    r = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                  [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                  [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
    return (r @ (p.reshape(3, -1) - ex.lla_to_ecef(*origin)[:, None])).T


def preview(plan, yaw_source="auto", step=3, origin=None):
    """ENU flight tracks and planned image positions for display. origin (lat, lon, alt):
    pass the project's enu_origin_lla to overlay extracted cameras; default = first planned image."""
    if not plan.sources or not plan.num_images:
        return None
    if origin is None:
        first = next(sp for sp in plan.sources if sp.windows)
        a0 = first.clip.telemetry(yaw_source)["a"][(first.windows[0][0] + first.windows[0][1]) // 2]
        origin = (a0[0], a0[1], a0[2] + first.alt_offset)
    out = []
    for sp in plan.sources:
        tel = sp.clip.telemetry(yaw_source)
        a = tel["a"]
        idx = np.arange(0, sp.clip.n, step)
        track = enu(a[idx, 0], a[idx, 1], a[idx, 2] + sp.alt_offset, origin)
        centers = np.array([(w0 + w1) // 2 for w0, w1 in sp.windows], int)
        samples = enu(a[centers, 0], a[centers, 1], a[centers, 2] + sp.alt_offset, origin) if len(centers) \
            else np.zeros((0, 3))
        out.append({"track": track, "track_valid": sp.valid[idx], "samples": samples,
                    "sample_frames": centers, "track_frames": idx})
    return {"origin": origin, "sources": out}


# --- extraction --------------------------------------------------------------------

def has_outputs(project):
    img = os.path.join(project, "images")
    return os.path.exists(os.path.join(project, "poses.json")) or (os.path.isdir(img) and bool(os.listdir(img)))


def archive_outputs(project, names=OUTPUTS):
    """Move image-derived outputs (names: OUTPUTS or a part of it) to PROJECT/previous/<time>/ (nothing
    is deleted)."""
    dst = os.path.join(project, "previous", time.strftime("%Y%m%d-%H%M%S"))
    moved = False
    for p in sorted(q for name in names for q in glob.glob(os.path.join(project, name))):
        os.makedirs(dst, exist_ok=True)
        shutil.move(p, os.path.join(dst, os.path.basename(p)))
        moved = True
    return dst if moved else None


def run(project, sources, settings, progress=None, overwrite=False, log=print):
    """Extract every source of the plan into PROJECT. progress(done, total, message)."""
    plan = make_plan(sources, settings)
    sizes = {(sp.clip.width, sp.clip.height) for sp in plan.sources}
    if len(sizes) > 1:
        raise ValueError(f"sources differ in resolution {sorted(sizes)}; one project = one camera")
    if not plan.num_images:
        raise ValueError("the plan selects no images: " + "; ".join(plan.warnings))
    if has_outputs(project):
        if not overwrite:
            raise FileExistsError(f"{project} already has images; re-extract with overwrite "
                                  "(old outputs are moved to previous/)")
        moved = archive_outputs(project)
        log(f"previous outputs moved to {moved}")
    save_plan(project, sources, settings)
    os.makedirs(os.path.join(project, "images"), exist_ok=True)
    for w in plan.warnings:
        log(f"warning: {w}")
    log(f"sampling: {plan.report}")

    total = plan.num_images
    progress = progress or (lambda done, total, msg: None)
    origin, records, seen = None, [], set()
    for sp in plan.sources:
        c = sp.clip
        yaw_src = c.yaw_source(settings.yaw_source)
        log(f"{sp.prefix}: lead {c.lead} frames via {c.align['method']}, {len(sp.windows)} images, "
            f"alt offset {sp.alt_offset:+.2f} m, camera heading from the {yaw_src} yaw")
        for k, rgb, sharp in ex.select_sharpest(c.video, sp.windows, c.n):
            name = f"{sp.prefix}_{k:05d}.jpg"
            if name in seen:  # overlapping pieces of one video
                continue
            seen.add(name)
            a = dict(zip(ex.AIRCRAFT_COLS, c.aircraft(k - c.lead)[0]))
            g = dict(zip(ex.GIMBAL_COLS, c.gimbal(k - c.lead)[0]))
            alt = a["rel_alt_m"] + sp.alt_offset
            if origin is None:
                origin = (a["lat"], a["lon"], alt)
            p = ex.ecef_to_enu(ex.lla_to_ecef(a["lat"], a["lon"], alt), origin)
            yaw = g["gimbal_yaw_deg"] if yaw_src == "gimbal" else a["ac_yaw_deg"]
            pitch, roll = g["gimbal_pitch_deg"], g["gimbal_roll_deg"]
            r_world_cam = ex.R_ENU_NED @ ex.r_ned_body(yaw, pitch, roll) @ ex.R_BODY_CAM
            r_cam_world = r_world_cam.T
            Image.fromarray(rgb).save(os.path.join(project, "images", name), quality=settings.quality,
                                      exif=ex.make_exif(a["lat"], a["lon"], alt))
            records.append({
                "image": name, "source": sp.index, "video": sp.prefix, "frame": k, "pts": float(c.pts[k]),
                "time_s": float(sp.time_base + c.pts[k]), "s": float(sp.s[k]), "sharpness": round(sharp, 2),
                **{col: float(a[col]) for col in ex.AIRCRAFT_COLS}, **{col: float(g[col]) for col in ex.GIMBAL_COLS},
                "alt_m": float(alt), "cam_yaw_deg": float(yaw), "cam_pitch_deg": float(pitch),
                "cam_roll_deg": float(roll),
                "enu_x": float(p[0]), "enu_y": float(p[1]), "enu_z": float(p[2]),
                "q_world_from_cam": [float(x) for x in ex.rot_to_quat(r_world_cam)],
                "colmap_qvec": [float(x) for x in ex.rot_to_quat(r_cam_world)],
                "colmap_tvec": [float(x) for x in -r_cam_world @ p],
            })
            progress(len(records), total, f"{sp.prefix} t = {c.pts[k]:.1f} s")
            if len(records) % 50 == 0:
                log(f"{len(records)}/{total} frames written ({sp.prefix} t = {c.pts[k]:.1f} s)")

    out = {
        "sources": [sp.describe() for sp in plan.sources],
        "source": os.path.abspath(plan.sources[0].source.video),  # reference video (single-video readers)
        "srt": plan.sources[0].clip.srt_path,
        "alignment": plan.sources[0].clip.align,
        "yaw_source": settings.yaw_source,
        "yaw_source_used": [sp.clip.yaw_source(settings.yaw_source) for sp in plan.sources],  # per source
        "sampling": plan.report,
        "warnings": plan.warnings,
        "plan": {"sources": [asdict(x) for x in sources], "settings": asdict(settings)},  # as in ingest.json
        "enu_origin_lla": origin,
        "ground_z": -origin[2],
        "conventions": {
            "world": "local ENU (x east, y north, z up) [m], origin = first image of the first source; z from "
                     "rel_alt (above the reference takeoff); other flights lifted by alt_offset_m",
            "camera": "OpenCV (x right, y down, z forward)",
            "attitude": "NED yaw-pitch-roll (ZYX) [deg]; cam_yaw from yaw_source, cam_pitch/roll from the "
                        "earth-stabilized gimbal; cam_pitch < 0 looks down",
            "q_world_from_cam": "(w,x,y,z), rotates camera-frame vectors into ENU",
            "colmap_qvec_tvec": "cam_from_world, as in COLMAP images.txt",
            "exif_gps_altitude": "alt_m = rel_alt + alt_offset [m above the reference takeoff]; local by design",
            "telemetry": "every value is resampled at (frame - lead) to match the image content",
            "time_s": "project timeline: wall-clock seconds since the first source started (pts if unknown)",
            "source": "index into sources[]; frame / pts are per video",
        },
        "images": records,
    }
    json.dump(out, open(os.path.join(project, "poses.json"), "w"), indent=1)
    with open(os.path.join(project, "poses.csv"), "w", newline="") as f:
        flat = [{k: (" ".join(f"{x:.9g}" for x in v) if isinstance(v, list) else v) for k, v in r.items()}
                for r in records]
        if flat:
            wr = csv.DictWriter(f, fieldnames=list(flat[0]))
            wr.writeheader()
            wr.writerows(flat)
    log(f"wrote {len(records)} images + poses to {project}")
    return out

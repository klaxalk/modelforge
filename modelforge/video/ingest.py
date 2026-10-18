"""Frame extraction for videos without position metadata (phone, action camera, ...): project kind "video".

Same plan / extract interface as modelforge.dji.ingest (load_clip, make_plan, preview, run), so the
GUI Ingest tab drives either; sources (video + time range), Settings and ingest.json are shared.
There is no telemetry, so images are spaced by
  motion  (default) equal steps of cumulative image motion: mean dense optical flow between frames
          on a 160 px wide copy, in fractions of the image width. Standing still gives no images,
          fast pans give more; forward motion counts too (its flow is radial).
  time    equal steps of video time.
The sharpest frame in each window is kept (as for DJI). poses.json lists the images without poses;
the Sparse stage then runs incremental SfM (modelforge.sfm.reconstruct) instead of the prior refine.

Usage:
    python -m modelforge.video.ingest phone.mp4 [more.mp4 ...] -o PROJECT [--num-images N] [--sampling time]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from modelforge.dji import extract as ex
from modelforge.dji import ingest as dji

KIND = "video"
SAMPLING = ("motion", "time")
MOTION_CACHE = ".modelforge_motion.json"  # per video folder, keyed by file size + mtime
FLOW_WIDTH = 160


def default_settings():
    return dji.Settings(sampling="motion", num_images=200)


def probe(video):
    """{n, fps, duration} from the container (frame count by counting packets: exact, no decoding)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets,avg_frame_rate,r_frame_rate:format=duration", "-of", "json", video],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    num, _, den = (st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    n = int(st.get("nb_read_packets") or 0)
    duration = float(d.get("format", {}).get("duration") or (n / fps if fps else 0.0))
    if not n:
        n = int(round(duration * fps))
    return {"n": n, "fps": fps or 30.0, "duration": duration}


def motion(video, n=None):
    """Per-frame image motion (mean optical-flow magnitude / image width; frame 0 = 0), cached on disk."""
    st = os.stat(video)
    key = f"{st.st_size}:{st.st_mtime_ns}"
    path = os.path.join(os.path.dirname(os.path.abspath(video)), MOTION_CACHE)
    try:
        cache = json.load(open(path))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return np.array(cache[key], float)
    w, h = ex.video_size(video)
    fw, fh = FLOW_WIDTH, max(2, int(round(FLOW_WIDTH * h / w / 2)) * 2)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-map", "0:v:0", "-fps_mode", "passthrough",
                          "-vf", f"scale={fw}:{fh},format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    prev, out = None, [0.0]
    while True:
        b = p.stdout.read(fw * fh)
        if len(b) < fw * fh:
            break
        f = np.frombuffer(b, np.uint8).reshape(fh, fw)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, f, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            out.append(float(np.linalg.norm(flow, axis=2).mean()) / fw)
        prev = f
    p.wait()
    m = np.array(out)
    if n is not None and len(m) != n:  # container count vs decoded frames: pad / trim to the clip
        m = np.pad(m, (0, max(0, n - len(m))))[:n]
    try:
        cache[key] = [round(x, 7) for x in m.tolist()]
        with open(path, "w") as fh_:
            json.dump(cache, fh_)
    except OSError:
        pass
    return m


class Clip:
    """One video without telemetry: frame count, timing, size; image motion on demand."""

    def __init__(self, video):
        self.video = video
        info = probe(video)
        self.n, self.fps, self.duration = info["n"], info["fps"], info["duration"]
        self.pts = np.arange(self.n) / self.fps  # constant frame rate assumed (spacing only)
        self.width, self.height = ex.video_size(video)
        self.wall_start = dji._wall_start(None, video)
        self.srt_path, self.lead, self.align, self.takeoff_abs_alt = None, 0.0, {"method": "none"}, None
        self._motion = None

    @property
    def motion(self):
        if self._motion is None:
            self._motion = motion(self.video, self.n)
        return self._motion

    frame_image = dji.Clip.frame_image


_CLIPS = {}


def load_clip(video, lead=None, srt=None):
    """Cached Clip (lead / srt accepted for interface compatibility with the DJI ingest; unused)."""
    st = os.stat(video)
    key = (os.path.abspath(video), st.st_size, st.st_mtime)
    if key not in _CLIPS:
        _CLIPS[key] = Clip(video)
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
        return {"video": self.source.video, "prefix": self.prefix, "start": self.source.start,
                "end": self.source.end, "duration_s": round(self.clip.duration, 3), "fps": round(self.clip.fps, 4),
                "size": [self.clip.width, self.clip.height], "time_base_s": round(self.time_base, 3),
                "sampling": self.report}


@dataclass
class Plan:
    sources: list
    report: dict
    warnings: list = field(default_factory=list)

    @property
    def num_images(self):
        return sum(len(sp.windows) for sp in self.sources)


def make_plan(sources, settings):
    if settings.sampling not in SAMPLING:
        settings = dji.Settings(**{**asdict(settings), "sampling": "motion"})
    enabled = [(i, s) for i, s in enumerate(sources) if s.enabled]
    if not enabled:
        return Plan([], {}, ["no enabled sources"])
    clips = {i: load_clip(s.video) for i, s in enabled}
    enabled.sort(key=lambda x: (clips[x[0]].wall_start + x[1].start if clips[x[0]].wall_start is not None
                                else float("inf"), x[0]))
    curves, valids, warnings = {}, {}, []
    for i, s in enabled:
        c = clips[i]
        end = c.duration if s.end is None else s.end
        valids[i] = (c.pts >= s.start) & (c.pts <= end)
        curves[i] = np.cumsum(c.motion) if settings.sampling == "motion" else c.pts.copy()
    total = sum(float(curves[i][valids[i]][-1] - curves[i][valids[i]][0]) for i, _ in enabled if valids[i].any())
    spacing = settings.spacing or (total / settings.num_images if total > 0 else 1.0)
    prefixes = dji._prefixes([s.video for _, s in enabled])
    walls = [clips[i].wall_start for i, _ in enabled]
    plans, bases, cum = [], {}, 0.0
    for k, (i, s) in enumerate(enabled):
        c = clips[i]
        windows, rep = ex.plan_windows(curves[i], valids[i], spacing=spacing, window=settings.window) \
            if valids[i].any() else ([], {})
        rep = {**rep, "num_images": len(windows)}
        key = os.path.abspath(s.video)
        if key not in bases:  # one timeline offset per video
            if all(w is not None for w in walls):
                bases[key] = c.wall_start - walls[0]
            else:
                bases[key], cum = cum, cum + c.duration
        plans.append(SourcePlan(k, s, c, prefixes[s.video], valids[i], curves[i], windows, rep, 0.0, bases[key]))
    unit = "image widths of motion" if settings.sampling == "motion" else "s"
    report = {"mode": settings.sampling, "unit": unit, "total": round(total, 3), "spacing": round(spacing, 5),
              "num_images": sum(len(p.windows) for p in plans), "window": settings.window, "sources": len(plans)}
    sizes = {(c.width, c.height) for c in clips.values()}
    if len(sizes) > 1:
        warnings.append(f"sources differ in resolution {sorted(sizes)}: one project = one camera")
    return Plan(plans, report, warnings)


def preview(plan, *args, **kwargs):
    """No telemetry, so nothing to show in 3D before reconstruction."""
    return None


def run(project, sources, settings, progress=None, overwrite=False, log=print):
    """Extract every source of the plan into PROJECT (images + poses.json without poses)."""
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
    dji.save_plan(project, sources, settings, kind=KIND)
    os.makedirs(os.path.join(project, "images"), exist_ok=True)
    for w in plan.warnings:
        log(f"warning: {w}")
    log(f"sampling: {plan.report}")
    total, records, seen = plan.num_images, [], set()
    progress = progress or (lambda done, total, msg: None)
    for sp in plan.sources:
        c = sp.clip
        log(f"{sp.prefix}: {len(sp.windows)} images")
        for k, rgb, sharp in ex.select_sharpest(c.video, sp.windows, c.n):
            name = f"{sp.prefix}_{k:05d}.jpg"
            if name in seen:  # overlapping pieces of one video
                continue
            seen.add(name)
            Image.fromarray(rgb).save(os.path.join(project, "images", name), quality=settings.quality)
            records.append({"image": name, "source": sp.index, "video": sp.prefix, "frame": k,
                            "pts": float(c.pts[k]), "time_s": float(sp.time_base + c.pts[k]),
                            "s": float(sp.s[k]), "sharpness": round(sharp, 2)})
            progress(len(records), total, f"{sp.prefix} t = {c.pts[k]:.1f} s")
    out = {
        "kind": KIND,
        "sources": [sp.describe() for sp in plan.sources],
        "source": os.path.abspath(plan.sources[0].source.video),
        "sampling": plan.report,
        "warnings": plan.warnings,
        "plan": {"sources": [asdict(x) for x in sources], "settings": asdict(settings)},  # as in ingest.json
        "conventions": {
            "poses": "none: the video has no position metadata; the Sparse stage reconstructs them (incremental "
                     "SfM), in an arbitrary frame (oriented up, scale normalized; see report.json)",
            "time_s": "project timeline: wall-clock seconds since the first source started (pts if unknown)",
            "source": "index into sources[]; frame / pts are per video",
        },
        "images": records,
    }
    json.dump(out, open(os.path.join(project, "poses.json"), "w"), indent=1)
    with open(os.path.join(project, "poses.csv"), "w", newline="") as f:
        if records:
            wr = csv.DictWriter(f, fieldnames=list(records[0]))
            wr.writeheader()
            wr.writerows(records)
    log(f"wrote {len(records)} images to {project}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--spacing", type=float)
    ap.add_argument("--sampling", choices=SAMPLING, default="motion")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    settings = dji.Settings(sampling=args.sampling, num_images=args.num_images, spacing=args.spacing)
    try:
        run(args.output, [dji.Source(v, args.start, args.end) for v in args.videos], settings,
            overwrite=args.overwrite)
    except (ValueError, FileExistsError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

"""Extract frames from a DJI video together with time-aligned 6-DoF pose priors.

The djmd aircraft/gimbal records lead the images by ~0.7 s (docs/djmd_format.md),
so the lead is estimated per clip and every telemetry stream is resampled at
`frame - lead` before being attached to a frame.

Lead estimation, in order of preference:
  1. SRT `pp_current` (camera attitude, in sync with video) vs. djmd gimbal yaw;
  2. global optical flow vs. djmd gimbal yaw rate (no SRT);
  3. --lead / DEFAULT_LEAD_FRAMES.

Sampling: frames are placed at equal steps of a cumulative curve s(frame), and the
sharpest frame in a window around each target is kept.
  distance (default): ds = |v| dt + w * d(optical axis angle)
      v is the aligned NED velocity (exactly 0 when hovering, so no GPS-jitter creep);
      w [m/deg] credits pure rotation; default = median look-at depth * pi/180, i.e. one
      degree counts like the translation that shifts the view equally. --rot-weight 0
      gives pure travelled distance.
  time: s = pts.
The step is --spacing X (m or s), or S_total / --num-images N.

This module holds the per-video machinery; modelforge.dji.ingest combines one or more
videos (each with a time range) into one project.

Outputs (in OUT/):
  images/<video>_NNNNN.jpg  selected frames (NNNNN = video frame), EXIF GPS = aligned prior
  poses.json                per-image priors + sources + conventions + alignment + sampling report
  poses.csv                 same, flat
  ingest.json               the sources + settings used (editable in the GUI Ingest tab)

Usage:
    python -m modelforge.dji.extract data/clip.MP4 -o data/clip_frames --num-images 300
    python -m modelforge.dji.extract data/a.MP4 data/b.MP4 -o data/site --num-images 500
"""

import argparse
import json
import math
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image


DEFAULT_LEAD_FRAMES = 20.7  # measured on DJI NEO2, docs/djmd_format.md
SAMPLE_PERIOD_FRAMES = 3  # aircraft and gimbal records update at 10 Hz on 29.97 fps video

# Body (x fwd, y right, z down) -> OpenCV camera (x right, y down, z fwd).
R_BODY_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
R_ENU_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# --- telemetry resampling ----------------------------------------------------

class Stream:
    """A 10 Hz sample-and-hold telemetry stream, linearly interpolable at fractional frames.

    Samples are reconstructed from runs of identical values: a run of length L
    holds ~L/3 samples, so knots are placed every SAMPLE_PERIOD_FRAMES inside it.
    That keeps hovering (constant values) from turning into a slow ramp.
    """

    def __init__(self, values, angle_cols=()):
        values = np.asarray(values, float)
        change = np.r_[True, np.any(values[1:] != values[:-1], axis=1)]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(values)]
        knots = []
        for s, e in zip(starts, ends):
            n = max(1, round((e - s) / SAMPLE_PERIOD_FRAMES))
            knots.extend(s + SAMPLE_PERIOD_FRAMES * j for j in range(n) if s + SAMPLE_PERIOD_FRAMES * j < e)
        self.t = np.asarray(knots, float)
        self.v = values[knots].copy()
        self.angle_cols = tuple(angle_cols)
        for c in self.angle_cols:
            self.v[:, c] = np.degrees(np.unwrap(np.radians(self.v[:, c])))

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.stack([np.interp(x, self.t, self.v[:, c]) for c in range(self.v.shape[1])], axis=1)
        for c in self.angle_cols:
            out[:, c] = wrap180(out[:, c])
        return out


def build_streams(meta):
    aircraft = Stream([(m.lat, m.lon, m.rel_alt_m, m.abs_alt_m, m.vel_n, m.vel_e, m.vel_d,
                        m.ac_pitch_deg, m.ac_roll_deg, m.ac_yaw_deg) for m in meta], angle_cols=(7, 8, 9))
    gimbal = Stream([(m.gimbal_pitch_deg, m.gimbal_roll_deg, m.gimbal_yaw_deg) for m in meta],
                    angle_cols=(0, 1, 2))
    return aircraft, gimbal


AIRCRAFT_COLS = ("lat", "lon", "rel_alt_m", "abs_alt_m", "vel_n", "vel_e", "vel_d",
                 "ac_pitch_deg", "ac_roll_deg", "ac_yaw_deg")
GIMBAL_COLS = ("gimbal_pitch_deg", "gimbal_roll_deg", "gimbal_yaw_deg")


# --- lead estimation ----------------------------------------------------------

def quat_yaw_pitch(q):
    w, x, y, z = q
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return yaw, pitch


def estimate_lead_srt(srt, gimbal, n_frames):
    """Fractional lead minimizing median |pp_current yaw - gimbal yaw(k - lead)|."""
    k = np.array([i for i in range(min(len(srt), n_frames)) if "pp_current" in srt[i]])
    pp_yaw = np.array([quat_yaw_pitch(srt[i]["pp_current"])[0] for i in k])
    best = None
    for lead in np.arange(0.0, 45.0, 0.1):
        sel = k - lead >= 0
        err = np.median(np.abs(wrap180(pp_yaw[sel] - gimbal(k[sel] - lead)[:, 2])))
        if best is None or err < best[1]:
            best = (float(lead), float(err))
    return {"method": "srt_pp_current", "lead_frames": round(best[0], 2),
            "median_yaw_err_deg": round(best[1], 3)}


def global_flow(mp4, w=320, h=180, start=None, duration=None):
    """Horizontal global image shift between consecutive frames (phase correlation), optionally of the
    piece [start, start + duration) s only (accurate input seek, so the first value is that frame's)."""
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.4f}"] if start else []) + ["-i", mp4] + \
        (["-t", f"{duration:.4f}"] if duration else []) + \
        ["-map", "0:v:0", "-fps_mode", "passthrough", "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    prev, dx = None, [0.0]
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            break
        f = np.frombuffer(b, np.uint8).reshape(h, w).astype(np.float32)
        if prev is not None:
            dx.append(cv2.phaseCorrelate(prev, f, win)[0][0])
        prev = f
    p.wait()
    return np.array(dx)


def flow_windows(n, fps, window_s=45.0, count=4):
    """Frame ranges [(k0, k1)] to estimate the lead from: the whole clip when it is short, else `count`
    windows of window_s spread evenly over it (decoding a long 4K clip in full takes over 10 minutes)."""
    m = int(round(window_s * fps))
    if n <= 1.5 * count * m:
        return [(0, n)]
    starts = np.linspace(0, n - m, count).astype(int)
    return [(int(k0), int(k0) + m) for k0 in starts]


LEAD_RANGE = (-15.0, 45.0)  # searched lead [frames]; negative = telemetry trails the images


def _best_lead(k, img, gimbal, sm, n_frames):
    """(lead, |corr|) maximizing the correlation of image shift and gimbal yaw rate over frames k."""
    best = (0.0, 0.0)
    for lead in np.arange(LEAD_RANGE[0], LEAD_RANGE[1] + 0.25, 0.5):
        yaw = np.degrees(np.unwrap(np.radians(gimbal(k - lead)[:, 2])))
        rate = sm(np.r_[0.0, np.diff(yaw)])
        sel = (k - lead >= 0) & (k - lead <= n_frames - 1)
        if sel.sum() < 30 or np.std(rate[sel]) == 0 or np.std(img[sel]) == 0:
            continue
        c = abs(np.corrcoef(img[sel], rate[sel])[0, 1])
        if c > best[1]:
            best = (float(lead), float(c))
    return best


def estimate_lead_flow(mp4, gimbal, n_frames, min_corr=0.5, pts=None):
    """Lead [frames] maximizing |corr| of the global image shift with the gimbal yaw rate.

    With pts, a long clip is sampled in flow_windows, each estimated on its own, and the lead is the
    median over the windows with |corr| >= min_corr: on the NEO2 sample the correlation is -0.9..-0.99
    in some minutes and breaks down in others (fast turns, where phase correlation fails), so one
    estimate over everything is weak (0.55 over the whole clip)."""
    sm = lambda x: np.convolve(x, np.ones(15) / 15, mode="same")  # noqa: E731
    fps = (len(pts) - 1) / (pts[-1] - pts[0]) if pts is not None and len(pts) > 1 else 29.97
    windows = flow_windows(n_frames, fps) if pts is not None else [(0, n_frames)]
    found = []
    for k0, k1 in windows:
        if (k0, k1) == (0, n_frames):
            dx = global_flow(mp4)
        else:  # seek a quarter frame early so that frame k0 comes first
            dx = global_flow(mp4, start=max(0.0, pts[k0] - 0.25 / fps), duration=(k1 - k0) / fps)
        n = min(len(dx), n_frames - k0, k1 - k0)
        lead, c = _best_lead(np.arange(k0, k0 + n, dtype=float), sm(dx[:n]), gimbal, sm, n_frames)
        found.append({"start_s": round(float(k0 / fps), 1), "lead_frames": lead, "corr": round(c, 3)})
    # a best lead at the edge of the search range is a failed fit, not an estimate
    good = [w for w in found if w["corr"] >= min_corr and LEAD_RANGE[0] < w["lead_frames"] < LEAD_RANGE[1]]
    if not good:
        return None
    return {"method": "optical_flow", "lead_frames": float(np.median([w["lead_frames"] for w in good])),
            "corr": float(np.median([w["corr"] for w in good])), "windows": found}


# --- geometry -----------------------------------------------------------------

_A, _E2 = 6378137.0, 6.69437999014e-3


def lla_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.array([(n + h) * np.cos(la) * np.cos(lo),
                     (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - _E2) + h) * np.sin(la)])


def ecef_to_enu(p, origin_lla):
    lat, lon = np.radians(origin_lla[0]), np.radians(origin_lla[1])
    r = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
    return r @ (p - lla_to_ecef(*origin_lla))


def r_ned_body(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return rz @ ry @ rx


def rot_to_quat(r):
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


# --- EXIF ---------------------------------------------------------------------

def _dms(deg):
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return (float(d), float(m), round(s, 6))


def make_exif(lat, lon, alt):
    ex = Image.Exif()
    ex[0x010F] = "DJI"  # Make
    # GPS IFD must be assigned as a dict; mutating get_ifd() of a fresh Exif is not serialized.
    ex[0x8825] = {
        1: "N" if lat >= 0 else "S", 2: _dms(lat),
        3: "E" if lon >= 0 else "W", 4: _dms(lon),
        5: b"\x00" if alt >= 0 else b"\x01", 6: round(abs(alt), 3),
    }
    return ex.tobytes()


# --- frame selection + writing -----------------------------------------------

def video_size(mp4):
    """Frame size as ffmpeg decodes it: phones store portrait video as landscape plus a rotation
    (display matrix / rotate tag), which ffmpeg applies, so +-90 deg swaps width and height."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_tags=rotate:stream_side_data=rotation", "-of", "json", mp4],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = st.get("tags", {}).get("rotate")
    for sd in st.get("side_data_list", []):
        rot = sd.get("rotation", rot)
    if rot is not None and int(round(float(rot))) % 180 != 0:
        w, h = h, w
    return w, h


def optical_axis_enu(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.cos(p) * np.sin(y), np.cos(p) * np.cos(y), np.sin(p)], axis=-1)


def sampling_curve(mode, pts, aircraft, gimbal, lead, yaw_source, valid, rot_weight=None):
    """Cumulative sampling coordinate s(frame) and a report dict (statistics over `valid` frames).

    Telemetry is read at frame - lead (clamped), so s follows what the images show.
    """
    n = len(pts)
    if mode == "time":
        return np.asarray(pts, float), {"mode": "time", "unit": "s"}
    x = np.clip(np.arange(n) - lead, 0, n - 1)  # both ends: a negative lead reads ahead
    a, g = aircraft(x), gimbal(x)
    dt = np.r_[0.0, np.diff(pts)]
    ds_trans = np.linalg.norm(a[:, 4:7], axis=1) * dt
    yaw = g[:, 2] if yaw_source == "gimbal" else a[:, 9]
    axis = optical_axis_enu(yaw, g[:, 0])
    dtheta = np.r_[0.0, np.degrees(np.arccos(np.clip(np.sum(axis[1:] * axis[:-1], axis=1), -1, 1)))]
    if rot_weight is None:
        # look-at depth assuming flat ground at takeoff height
        depth = np.clip(a[:, 2] / np.maximum(np.sin(np.radians(-g[:, 0])), 1e-3), 5.0, 200.0)
        rot_weight = float(np.median(depth[valid])) * math.pi / 180
    s = np.cumsum(ds_trans + rot_weight * dtheta)
    return s, {"mode": "distance", "unit": "m", "rot_weight_m_per_deg": round(rot_weight, 4),
               "path_m": round(float(ds_trans[valid].sum()), 1),
               "rotation_deg": round(float(dtheta[valid].sum()), 1)}


def plan_windows(s, valid, num_images=None, spacing=None, window=0.5):
    """Frame ranges [a, b] around equally spaced targets on s, one per output image.

    window is the fraction of the spacing searched for the sharpest frame; < 1 keeps
    ranges disjoint.
    """
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return [], {}
    sv = s[idx]
    total = float(sv[-1] - sv[0])
    if spacing is None:
        spacing = total / num_images if total > 0 else 1.0
    n = max(1, int(round(total / spacing))) if total > 0 else 1
    targets = sv[0] + (np.arange(n) + 0.5) * total / n if total > 0 else np.array([sv[0]])
    half = window * spacing / 2
    windows = []
    for t in targets:
        a = int(np.searchsorted(sv, t - half, side="left"))
        b = int(np.searchsorted(sv, t + half, side="right")) - 1
        if a > b:  # s jumps over the window; take the nearest frame
            a = b = int(np.clip(np.searchsorted(sv, t), 0, len(sv) - 1))
        windows.append((int(idx[a]), int(idx[b])))
    return windows, {"total": round(total, 2), "spacing": round(total / n if total > 0 else 0.0, 4),
                     "num_images": n, "window": window}


def select_sharpest(mp4, windows, n_frames):
    """Single decode pass; keep the sharpest frame in each [a, b] window. Yields (frame, rgb, sharpness)."""
    if not windows:
        return
    w, h = video_size(mp4)
    # trim by frame index (not -ss) so frame numbering stays exact; frames before k0 are decoded but not piped
    k0, k_last = windows[0][0], windows[-1][1]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", mp4, "-map", "0:v:0", "-fps_mode", "passthrough",
                          "-vf", f"trim=start_frame={k0}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    size = w * h * 3
    wi, best, k = 0, None, k0
    while wi < len(windows):
        b = p.stdout.read(size)
        if len(b) < size:
            break
        a_, b_ = windows[wi]
        if k >= a_:
            img = np.frombuffer(b, np.uint8).reshape(h, w, 3)
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
            if best is None or sharp > best[2]:
                best = (k, img.copy(), sharp)
        if k >= b_:
            if best:
                yield best
            best, wi = None, wi + 1
        k += 1
    p.kill()
    p.wait()
    if wi < len(windows):
        print(f"warning: video ended at frame {k} before window {wi} (djmd has {n_frames} packets)", file=sys.stderr)


def main():
    from modelforge.dji import ingest
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="*", help="one or more DJI videos (or --plan)")
    ap.add_argument("-o", "--output", required=True, help="project directory")
    ap.add_argument("--plan", help="ingest.json with sources + settings (as written by the GUI); "
                                   "replaces the videos and sampling options")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract into a project that has images (old outputs move to previous/)")
    ap.add_argument("--sampling", choices=("distance", "time"), default="distance",
                    help="sampling curve: travelled distance + rotation credit (default), or video time")
    amount = ap.add_mutually_exclusive_group()
    amount.add_argument("--num-images", type=int, default=300,
                        help="number of images over all videos (default 300)")
    amount.add_argument("--spacing", type=float, help="step along the curve instead: meters (distance) or seconds (time)")
    ap.add_argument("--rot-weight", type=float,
                    help="distance mode: meters credited per degree of view rotation "
                         "(default: median look-at depth * pi/180; 0 = pure distance)")
    ap.add_argument("--window", type=float, default=0.5,
                    help="fraction of the spacing searched for the sharpest frame (default 0.5)")
    ap.add_argument("--start", type=float, default=0.0, help="start time [s], applied to every video")
    ap.add_argument("--end", type=float, help="end time [s], applied to every video")
    ap.add_argument("--srt", help="SRT sidecar (single video only; default: next to the video)")
    ap.add_argument("--lead", type=float, help="telemetry lead [frames]; skips estimation")
    ap.add_argument("--yaw-source", choices=("auto", "aircraft", "gimbal"), default="auto",
                    help="camera heading source (default auto: per clip, the gimbal yaw when it is a true "
                         "heading, as on the Mavic 4 whose gimbal can yaw; else the aircraft yaw, as on the "
                         "NEO2, whose gimbal has no yaw axis and reports yaw in an offset (~-36.5 deg) frame)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args()

    if args.plan:
        d = json.load(open(args.plan))
        sources = [ingest.Source(**x) for x in d["sources"]]
        settings = ingest.Settings(**d.get("settings", {}))
    else:
        if not args.videos:
            ap.error("give at least one video, or --plan")
        if args.srt and len(args.videos) > 1:
            ap.error("--srt needs a single video")
        sources = [ingest.Source(v, args.start, args.end, lead=args.lead, srt=args.srt) for v in args.videos]
        settings = ingest.Settings(args.sampling, args.num_images, args.spacing, args.rot_weight, args.window,
                                   args.yaw_source, args.quality)
    try:
        ingest.run(args.output, sources, settings, overwrite=args.overwrite,
                   log=lambda m: print(m, file=sys.stderr, flush=True))
    except (FileExistsError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

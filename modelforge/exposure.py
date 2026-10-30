"""Per-photo exposure and white balance from the DJI metadata (djmd camera record at the photo's frame).

DJI flights run auto exposure and auto white balance (2026-10-23, djmd camera records): dira (kladno) ISO 100-320 at
1/50 - 1/2585 s, 4.0 EV over the flight (p5-p95 2.2 EV), colour temperature 3647-6454 K; berlin_3_2 5.8 EV; karlak_1
2.6 EV. The same surface is then darker or brighter from photo to photo, which a splat can only explain with
view-dependent colour or fog. `per_image` gives each photo's relative exposure

    gain = (shutter x ISO / f-number^2) / median over the project's photos

so a render at the reference exposure (the median) times `gain` (in linear light) matches the photo; the export
writes it into transforms.json ("mf_exposure" per frame) for the splat plugin.

**The median is per camera** (poses.json images' "camera"; none = 0): the metadata compares exposures within one camera
only. berlin_all (2026-10-29): the Mavic 3 at a fixed ISO 800, 1/1000 s, f/7.1 (likely an ND filter, which the metadata
does not know), the NEO2 at ISO 400, 1/200 s, f/2.2: 4.7 stops apart on paper, while their photos are equally bright
(mean 0.45 vs 0.38). One median over both gave the NEO2's photos gains of x14-60, and tiles they dominate trained
~30x too dark. Cached in PROJECT/exposure.json
(keyed by poses.json's size + mtime). Projects without djmd (videos, ROS bags) get none.
"""

import json
import os

import numpy as np

CACHE = "exposure.json"
VERSION = 2  # 2: the median per camera


def _video(project, path):
    for p in (path, os.path.join(os.path.dirname(os.path.abspath(project)), "..", path)):
        if p and os.path.exists(p):
            return p
    return None


def per_image(project):
    """{image name: {"iso", "shutter_s", "fnum", "color_temp", "gain"}} for the photos with djmd metadata ({} if none)."""
    poses = os.path.join(project, "poses.json")
    if not os.path.exists(poses):
        return {}
    st = os.stat(poses)
    key = [st.st_size, st.st_mtime]
    cache = os.path.join(project, CACHE)
    if os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if c.get("key") == key and c.get("version") == VERSION:
                return c["images"]
        except (OSError, ValueError, KeyError):
            pass
    p = json.load(open(poses))
    from modelforge.dji import djmd
    meta = {}
    for k, s in enumerate(p.get("sources") or []):
        v = _video(project, s.get("video"))
        if v is None or not v.lower().endswith((".mp4", ".mov")):
            continue
        try:
            meta[k] = djmd.decode(v)
        except Exception:  # noqa: BLE001 - not a DJI clip (no djmd track): no exposure
            continue
    out, cam = {}, {}
    for r in p.get("images", []):
        m = meta.get(r.get("source", 0))
        f = r.get("frame")
        if m is None or f is None or not 0 <= f < len(m):
            continue
        c = m[f]
        if not (c.iso > 0 and c.shutter_s > 0 and c.fnum > 0):
            continue
        out[r["image"]] = {"iso": float(c.iso), "shutter_s": float(c.shutter_s), "fnum": float(c.fnum),
                           "color_temp": int(c.color_temp), "e": float(c.shutter_s * c.iso / c.fnum ** 2)}
        cam[r["image"]] = r.get("camera") or 0
    for k in set(cam.values()):
        ref = float(np.median([x["e"] for n, x in out.items() if cam[n] == k]))
        for n, x in out.items():
            if cam[n] == k:
                x["gain"] = round(x.pop("e") / ref, 5)
    json.dump({"key": key, "version": VERSION, "images": out}, open(cache, "w"))
    return out


def gains_for(project, records):
    """{name: gain} for photos outside the set in use (records like poses.json images: "image", "source",
    "frame", "camera"; e.g. extracted frames registered into the model, sfm/register.py), on the same reference (the
    median of the photos in use of the same camera)."""
    ref_imgs = per_image(project)
    if not ref_imgs:
        return {}
    p = json.load(open(os.path.join(project, "poses.json")))
    used_cam = {r["image"]: r.get("camera") or 0 for r in p.get("images", [])}
    refs = {}
    for n, x in ref_imgs.items():
        refs.setdefault(used_cam.get(n, 0), []).append(x["shutter_s"] * x["iso"] / x["fnum"] ** 2)
    refs = {k: float(np.median(v)) for k, v in refs.items()}
    src_cam = {k: s.get("camera") or 0 for k, s in enumerate(p.get("sources") or [])}
    from modelforge.dji import djmd
    meta, out = {}, {}
    for r in records:
        k = r.get("source", 0)
        if k not in meta:
            s = (p.get("sources") or [{}])[k] if k < len(p.get("sources") or []) else {}
            v = _video(project, s.get("video"))
            try:
                meta[k] = djmd.decode(v) if v and v.lower().endswith((".mp4", ".mov")) else None
            except Exception:  # noqa: BLE001
                meta[k] = None
        m, f = meta[k], r.get("frame")
        if m is None or f is None or not 0 <= f < len(m):
            continue
        c = m[f]
        ref = refs.get(r.get("camera", src_cam.get(k, 0)) or 0)
        if ref and c.iso > 0 and c.shutter_s > 0 and c.fnum > 0:
            out[r["image"]] = round(float(c.shutter_s * c.iso / c.fnum ** 2) / ref, 5)
    return out


# ------------------------------------------------------------------------------------------------- per-camera stops
STOPS = "exposure_stops.json"  # PROJECT/: {camera key (poses.json "cameras"): stops to brighten that camera's photos}


def cameras(project):
    """The project's camera keys in poses.json order (images' "camera" indexes them); [] without poses. Cached by
    poses.json's size + mtime (the GUI asks on every refresh; berlin_all's is tens of MB)."""
    path = os.path.join(project, "poses.json")
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime)
        if key not in _CAMERAS:
            cams = json.load(open(path)).get("cameras") or []
            _CAMERAS[key] = [c.get("key") or f"camera {k}" for k, c in enumerate(cams)] or ["camera 0"]
        return list(_CAMERAS[key])
    except (OSError, ValueError):
        return []


_CAMERAS = {}


def stops(project):
    """{camera key: stops} set for the project (Splat tab → Camera exposure); missing = 0."""
    try:
        return {k: float(v) for k, v in json.load(open(os.path.join(project, STOPS))).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def set_stops(project, camera, value):
    """Store one camera's correction (0 removes it)."""
    s = stops(project)
    if float(value):
        s[camera] = float(value)
    else:
        s.pop(camera, None)
    tmp = os.path.join(project, STOPS + ".part")
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(project, STOPS))


def factors(project, by_camera):
    """{image name: 2^-stops} for every photo of the cameras with a correction (poses.json, and the extracted pool's
    for photos registered later): the training render is multiplied by it next to the metadata gain, so a camera
    whose photos came out `stops` too dark is trained to the scene's brighter colours (the user, 2026-10-29: the
    berlin_all NEO2's photos ~0.75 stop under the Mavic 3's)."""
    keys = cameras(project)
    f = {k: 2.0 ** -float(by_camera.get(key, 0.0)) for k, key in enumerate(keys) if float(by_camera.get(key, 0.0))}
    out = {}
    if not f:
        return out
    for path in (os.path.join(project, "extracted", "poses.json"), os.path.join(project, "poses.json")):
        try:
            imgs = json.load(open(path)).get("images", [])
        except (OSError, ValueError):
            continue
        for r in imgs:
            c = r.get("camera") or 0
            if c in f:
                out[r["image"]] = f[c]
    return out

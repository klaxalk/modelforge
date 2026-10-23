"""Per-photo exposure and white balance from the DJI metadata (djmd camera record at the photo's frame).

DJI flights run auto exposure and auto white balance (2026-10-23, djmd camera records): dira (kladno) ISO 100-320 at
1/50 - 1/2585 s, 4.0 EV over the flight (p5-p95 2.2 EV), colour temperature 3647-6454 K; berlin_3_2 5.8 EV; karlak_1
2.6 EV. The same surface is then darker or brighter from photo to photo, which a splat can only explain with
view-dependent colour or fog. `per_image` gives each photo's relative exposure

    gain = (shutter x ISO / f-number^2) / median over the project's photos

so a render at the reference exposure (the median) times `gain` (in linear light) matches the photo; the export
writes it into transforms.json ("mf_exposure" per frame) for the splat plugin. Cached in PROJECT/exposure.json
(keyed by poses.json's size + mtime). Projects without djmd (videos, ROS bags) get none.
"""

import json
import os

import numpy as np

CACHE = "exposure.json"


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
            if c.get("key") == key:
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
    out = {}
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
    if out:
        ref = float(np.median([x["e"] for x in out.values()]))
        for x in out.values():
            x["gain"] = round(x.pop("e") / ref, 5)
    json.dump({"key": key, "images": out}, open(cache, "w"))
    return out

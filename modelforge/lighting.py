"""Lighting of the photos: the sun per photo and how much hard shadow each photo shows (docs/todo_lighting.md, step 2).

Sun: each photo's GPS position (poses.json lat / lon) and UTC time (its video's MP4 creation time + the frame's
time in the video, `pts`) through the solar ephemeris (modelforge/sun.py): azimuth (° from north, clockwise) and
elevation (°).

Shadow score: a photo at 512 px, its luminance in log units smoothed over ~1 % of the width (texture away, shadows
and surfaces stay); `contrast` = the spread of that (2nd to 98th percentile, in stops), `dark` = the share of pixels
more than 1.5 stops below the photo's median (shadow area), the sky left out (above the horizon: the photo's pixels
brighter than its 99th percentile are not counted). Direct sun gives hard, dark shadows: more contrast, more dark
area than overcast light over the same ground.

python -m modelforge.lighting PROJECT  ->  PROJECT/lighting/survey.json (+ a per-source summary on stdout)
"""
import argparse
import json
import os
import time

import numpy as np

from modelforge import sun

DIR = "lighting"


def _video_start(video):
    from modelforge.dji import ingest
    return ingest._mp4_creation_time(video)


def sun_per_photo(project):
    """{image: {"utc", "azimuth", "elevation", "source"}} for the photos of poses.json (those whose video's start
    time is known)."""
    p = json.load(open(os.path.join(project, "poses.json")))
    starts = {}
    out = {}
    for im in p["images"]:
        src = int(im.get("source", 0))
        if src not in starts:
            video = p["sources"][src]["video"]
            if not os.path.isabs(video) and not os.path.exists(video):
                video = os.path.join(project, video)
            starts[src] = _video_start(video) if os.path.exists(video) else None
        if starts[src] is None or im.get("lat") is None:
            continue
        utc = starts[src] + float(im.get("pts", 0.0))
        az, el = sun.position(float(im["lat"]), float(im["lon"]), utc)
        out[im["image"]] = {"utc": round(utc, 1), "azimuth": round(az, 2), "elevation": round(el, 2), "source": src}
    return out


def shadow_score(path, width=512):
    """{"contrast": stops, "dark": share} of a photo (see the top)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        raise OSError(f"cannot read {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    lum = img.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32) / 255.0  # BGR -> luma
    lg = np.log2(np.clip(lum, 1 / 255, 1.0))
    lg = cv2.GaussianBlur(lg, (0, 0), max(1.0, 0.01 * width))
    keep = lum < np.percentile(lum, 99)  # the sky and specular glints out
    v = lg[keep]
    med = np.median(v)
    return {"contrast": round(float(np.percentile(v, 98) - np.percentile(v, 2)), 3),
            "dark": round(float(np.mean(v < med - 1.5)), 4)}


def survey(project, every=1, on_line=print):
    """Sun + shadow score of every `every`-th photo -> PROJECT/lighting/survey.json; returns it."""
    t0 = time.time()
    suns = sun_per_photo(project)
    p = json.load(open(os.path.join(project, "poses.json")))
    photos = {}
    for k, im in enumerate(p["images"]):
        if k % every:
            continue
        rec = dict(suns.get(im["image"], {"source": int(im.get("source", 0))}))
        try:
            rec.update(shadow_score(os.path.join(project, "images", im["image"])))
        except OSError:
            continue
        photos[im["image"]] = rec
    sources = []
    for s, src in enumerate(p["sources"]):
        recs = [r for r in photos.values() if r["source"] == s]
        if not recs:
            continue
        med = lambda k: float(np.median([r[k] for r in recs if k in r])) if any(k in r for r in recs) else None  # noqa: E731
        sources.append({"source": s, "video": os.path.basename(src["video"]), "photos": len(recs),
                        "elevation": med("elevation"), "azimuth": med("azimuth"), "contrast": med("contrast"),
                        "dark": med("dark"),
                        "utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(min(r["utc"] for r in recs)))
                        if all("utc" in r for r in recs) else None})
    out = {"photos": photos, "sources": sources, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(project, DIR), exist_ok=True)
    json.dump(out, open(os.path.join(project, DIR, "survey.json"), "w"), indent=1)
    for s in sources:
        on_line(f"lighting: {s['video']}: {s['photos']} photos, {s['utc']} UTC, sun {s['elevation']:.1f}° up at "
                f"{s['azimuth']:.0f}°, contrast {s['contrast']:.2f} stops, dark share {100 * s['dark']:.1f} %"
                if s["elevation"] is not None else f"lighting: {s['video']}: contrast {s['contrast']:.2f} stops")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    survey(a.project, a.every)


if __name__ == "__main__":
    main()

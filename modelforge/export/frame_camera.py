"""Intrinsics of a transforms.json frame: a frame's own keys, else the file's top-level ones. (nerfstudio takes a
top-level key over the frames', so the export writes each key in one place only: several cameras (multi-camera DJI
projects) per frame, with only camera_model at the top; one camera at the top.)

No imports at all: the scripts run in nerfstudio's venv load it by its file path (importlib), since putting
this folder on sys.path would shadow nerfstudio with export/nerfstudio.py.
"""

KEYS = ("camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h", "k1", "k2", "k3", "k4", "p1", "p2")


def camera(t, frame):
    """{key: value} of the frame's camera (KEYS present in the frame or at the top)."""
    return {k: frame[k] if k in frame else t[k] for k in KEYS if k in frame or k in t}


def cameras(t):
    """(list of distinct cameras, camera index per frame)."""
    out, index, seen = [], [], {}
    for f in t["frames"]:
        c = camera(t, f)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen[key] = len(out)
            out.append(c)
        index.append(seen[key])
    return out, index


def largest(t):
    """The camera with the widest image (the training plan's resolution rules use it)."""
    cams, _ = cameras(t)
    return max(cams, key=lambda c: c.get("w", 0)) if cams else {k: t[k] for k in KEYS if k in t}


def long_side(t):
    """The longest image side of the dataset's cameras: the export's (and a tile's) "mf_long_side" = the project's
    largest camera, else the largest here."""
    if t.get("mf_long_side"):
        return int(t["mf_long_side"])
    cams, _ = cameras(t)
    return max((max(c["w"], c["h"]) for c in cams), default=max(t.get("w", 0), t.get("h", 0)))


def train_scale(t, cam, downscale):
    """One common training resolution (multi-camera projects): the largest camera trains at 1 / downscale, every
    other camera at the largest size within that long side, never upscaled (a 1080p camera beside 4K ones at x2
    trains at 1920, not 960). With one camera: 1 / downscale, as nerfstudio's --downscale-factor."""
    top, lc = long_side(t), max(cam["w"], cam["h"])
    return 1.0 / downscale if lc >= top else min(1.0, top / (downscale * lc))


def train_camera(t, cam, downscale):
    """The camera as it trains: intrinsics x train_scale, size int(w x scale) (nerfstudio's rescale_output_resolution
    rounding; the same as floor(w / N) for the factors used)."""
    s = train_scale(t, cam, downscale)
    out = dict(cam)
    for k in ("fl_x", "fl_y", "cx", "cy"):
        if k in out:
            out[k] = cam[k] * s
    out["w"], out["h"] = int(cam["w"] * s), int(cam["h"] * s)
    return out

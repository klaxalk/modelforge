"""Intrinsics of a transforms.json frame: its own keys over the file's top-level ones (nerfstudio's rule). Several
cameras (multi-camera DJI projects) carry them per frame; one camera only at the top.

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

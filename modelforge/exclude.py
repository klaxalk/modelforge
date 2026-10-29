"""Photos the user leaves out of the splat and everything after it (the user asked, 2026-10-28: some photos should not
go into the splat, without extracting or running SfM again).

PROJECT/excluded.json holds the choice in the order it was made: {"entries": [{"name": image} or {"t0", "t1",
"source"}, ...]}, a range covering the photos of that source (None: of every source) with a project time
(poses.json time_s) inside [t0, t1]. The SfM model keeps them; the export leaves them out like the isolated photos
(pipeline.stage_export): the points are triangulated again from the other photos (sparse/export, which the SfM mesh
uses), transforms.json lists the others, and photos registered later (sfm/register.py) are left out by the same
rule. So every stage reading transforms.json or sparse/export (splat, tiles, Difix, semantic features, the SfM mesh
and its colours, the held-out evaluation) leaves them out; COLMAP's dense (sparse/refined) does not.
A change applies at the next export.
"""
import json
import os

FILE = "excluded.json"


def load(project):
    """The entries, oldest first ([] without a file)."""
    try:
        return list(json.load(open(os.path.join(project, FILE))).get("entries", []))
    except (OSError, ValueError, AttributeError):
        return []


def save(project, entries):
    path = os.path.join(project, FILE)
    if not entries:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = path + ".part"
    json.dump({"entries": entries}, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def records(project):
    """poses.json's image records (image, source, time_s), [] without one."""
    try:
        return json.load(open(os.path.join(project, "poses.json")))["images"]
    except (OSError, ValueError, KeyError):
        return []


def names(project, recs=None, entries=None):
    """The image names excluded: those picked by name, and those of recs (default: poses.json's; registered.json's
    records work too: they carry source and time_s) inside a range."""
    entries = load(project) if entries is None else entries
    out = {e["name"] for e in entries if "name" in e}
    ranges = [e for e in entries if "t0" in e]
    if ranges:
        for r in records(project) if recs is None else recs:
            t = r.get("time_s")
            if t is None:
                continue
            for e in ranges:
                if e["t0"] <= t <= e["t1"] and e.get("source") in (None, r.get("source", 0)):
                    out.add(r.get("image") or r.get("name"))
                    break
    out.discard(None)
    return out


def toggle(project, name):
    """Exclude the photo, or include it again when it was picked by name. Returns True when it is now excluded."""
    entries = load(project)
    kept = [e for e in entries if e.get("name") != name]
    if len(kept) < len(entries):
        save(project, kept)
        return False
    save(project, entries + [{"name": name}])
    return True


def add_range(project, t0, t1, source=None):
    save(project, load(project) + [{"t0": float(min(t0, t1)), "t1": float(max(t0, t1)), "source": source}])


def undo(project):
    """Drop the last entry; returns it (None when there was none)."""
    entries = load(project)
    if not entries:
        return None
    save(project, entries[:-1])
    return entries[-1]


def clear(project):
    save(project, [])

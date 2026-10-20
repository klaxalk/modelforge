"""Projects: one folder each under a root (default data/), holding its sources, intermediate
results and final results.

  ingest.json                         sources (video + time range) and sampling settings
  final:        poses.json, transforms.json, sparse_pc.ply, sparse/refined/,
                dense/fused.ply, dense/meshed-*.ply, dense/textured-*/, splat/splat_enu.ply
  intermediate: images/ (the extracted frames, needed by every later stage), features.db,
                database.db, pairs.txt, sparse/round_*, dense/images, dense/sparse, dense/stereo
                (PatchMatch maps, ~10 GB), dense/preview-*.ply, nerfstudio/ (training runs), images_N/,
                splat/splat.ply, previous/ (archived earlier extractions)

`clean` deletes the intermediates that are only needed to re-run stages, never the final
results, the sources or the images.
"""

import glob
import json
import os
import re
import shutil

from modelforge.dji import ingest

MARKERS = (ingest.PLAN_FILE, "poses.json")
FINAL = (
    ("poses.json", "priors (telemetry poses)"),
    ("transforms.json", "nerfstudio dataset"),
    ("sparse_pc.ply", "sparse points (nerfstudio init)"),
    ("sparse/refined", "refined COLMAP model"),
    ("dense/fused.ply", "dense point cloud"),
    ("dense/meshed-delaunay.ply", "mesh (Delaunay)"),
    ("dense/meshed-poisson.ply", "mesh (Poisson)"),
    ("dense/textured-delaunay", "textured mesh (Delaunay)"),
    ("dense/textured-poisson", "textured mesh (Poisson)"),
    ("splat/splat_enu.ply", "Gaussian splat (ENU)"),
    ("export", "exported mesh / splat"),
)
INTERMEDIATE = (
    ("images", "extracted frames"),
    ("features.db", "SIFT features"),
    ("database.db", "matches"),
    ("pairs.txt", "pair list"),
    ("sparse/round_*", "refinement rounds"),
    ("sparse/calib_*", "refinement calibration rounds (local BA)"),
    ("dense/images", "undistorted images"),
    ("dense/sparse", "undistorted model"),
    ("dense/stereo", "PatchMatch depth/normal maps"),
    ("dense/preview-*.ply", "GUI mesh previews"),
    ("nerfstudio", "splatfacto training runs"),
    ("tiles", "splat tiles (per-tile datasets and training runs; the merged splat is splat/splat_enu.ply)"),
    ("images_*", "downscaled frames for splat training"),
    ("splat/splat.ply", "splat (nerfstudio frame)"),
    ("previous", "archived earlier extractions"),
    ("extracted", "all extracted frames (a subset of them is in images/)"),
    ("analysis", "overlap analysis of the extracted frames (features + matches)"),
)
# Never deleted by clean: the extracted frames are an input of every stage.
KEEP_ON_CLEAN = ("images", "extracted")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
LIBRARY_DIR = "videos"  # ROOT/videos holds the video library (modelforge.videos), so it is no project name


def _check_name(name):
    if not NAME_RE.match(name):
        raise ValueError("project names use letters, digits, '_', '-', '.' (max 64), starting with a letter or digit")
    if name.lower() == LIBRARY_DIR:
        raise ValueError(f"'{LIBRARY_DIR}' is the video library folder; choose another project name")


def is_project(path):
    return os.path.isdir(path) and any(os.path.exists(os.path.join(path, m)) for m in MARKERS)


def list_projects(root):
    """Project folders directly under root, most recently modified first."""
    if not os.path.isdir(root):
        return []
    out = [os.path.join(root, d) for d in os.listdir(root) if not d.startswith(".")]
    out = [p for p in out if is_project(p)]
    return sorted(out, key=_mtime, reverse=True)


def _mtime(path):
    times = [os.path.getmtime(os.path.join(path, f)) for f in ("report.json", "poses.json", ingest.PLAN_FILE)
             if os.path.exists(os.path.join(path, f))]
    return max(times or [os.path.getmtime(path)])


def create(root, name, copy_plan_from=None, kind="dji"):
    """New project folder root/name with an ingest.json of the given kind (ingest.KINDS): empty, or
    the sources and settings of another project (whose kind it then takes)."""
    _check_name(name)
    if kind not in ingest.KINDS:
        raise ValueError(f"project kind must be one of {tuple(ingest.KINDS)}")
    path = os.path.join(root, name)
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists")
    if copy_plan_from:
        sources, settings = ingest.load_plan(copy_plan_from)
        kind = ingest.load_kind(copy_plan_from)
    else:
        from modelforge.video import ingest as video_ingest
        sources, settings = [], (video_ingest.default_settings() if kind == "video" else ingest.Settings())
    ingest.save_plan(path, sources, settings, kind=kind)
    return path


def subsample_selection(images, every=1, offset=0, start=None, end=None):
    """Indices (into poses.json "images") kept by `subsample`: inside [start, end] on the project timeline
    (time_s), then every `every`-th image of each source from its `offset`-th, in capture order."""
    every, offset = max(1, int(every)), max(0, int(offset))
    by_source = {}
    for k, im in enumerate(images):
        t = im.get("time_s")
        if t is not None and ((start is not None and t < start) or (end is not None and t > end)):
            continue
        by_source.setdefault(im.get("source", 0), []).append(k)
    keep = []
    for ks in by_source.values():
        ks.sort(key=lambda k: (images[k].get("time_s", 0.0), k))
        keep += ks[offset::every]
    return sorted(keep)


POOL = "extracted"  # all extracted images + their poses, once a subset is in use (images/ = hard links into it)


def subsample_info(path):
    """{"extracted": images extracted (pool), "used": images in images/ now, "setting": the subsample
    applied (poses.json "subsample") or None}."""
    def count(p):
        try:
            return len(json.load(open(p))["images"])
        except (OSError, ValueError, KeyError):
            return 0
    used = count(os.path.join(path, "poses.json"))
    pool = os.path.join(path, POOL, "poses.json")
    setting = None
    if os.path.exists(os.path.join(path, "poses.json")):
        setting = json.load(open(os.path.join(path, "poses.json"))).get("subsample")
    return {"extracted": count(pool) if os.path.exists(pool) else used, "used": used, "setting": setting}


def extracted_images(path):
    """poses.json "images" of everything extracted (the pool if a subset is in use)."""
    pool = os.path.join(path, POOL, "poses.json")
    src = pool if os.path.exists(pool) else os.path.join(path, "poses.json")
    return json.load(open(src))["images"]


def apply_subsample(path, every=1, offset=0, start=None, end=None, names=None, setting=None):
    """Use a subset of the project's extracted images (subsample_selection) from now on: images/,
    poses.json and poses.csv become that subset, so every stage runs on it unchanged. The first time,
    the full extraction moves to PROJECT/extracted/ (the pool); images/ then holds hard links into it.
    Results computed from the previous image set (features, matches, sparse, dense, splat, ...) are
    moved to previous/<time>/, as on re-extraction. every=1 over the whole timeline uses all images
    again. names: an explicit list of extracted images instead (keyframes, sfm/overlap.py), recorded
    with `setting`. Returns (images used, images extracted, archive folder or None)."""
    import csv
    path = os.path.abspath(path)
    if not os.path.exists(os.path.join(path, "poses.json")):
        raise ValueError("no extracted images (poses.json) yet: extract them in the Ingest tab first")
    pool = os.path.join(path, POOL)
    if not os.path.exists(os.path.join(pool, "poses.json")):  # first subset: the extraction becomes the pool
        os.makedirs(pool)
        os.rename(os.path.join(path, "images"), os.path.join(pool, "images"))
        for f in ("poses.json", "poses.csv"):
            if os.path.exists(os.path.join(path, f)):
                shutil.copy2(os.path.join(path, f), os.path.join(pool, f))
    poses = json.load(open(os.path.join(pool, "poses.json")))
    if names is not None:
        wanted = set(names)
        keep = [k for k, im in enumerate(poses["images"]) if im["image"] in wanted]
    else:
        keep = subsample_selection(poses["images"], every, offset, start, end)
    if not keep:
        raise ValueError("no images left with these settings")
    stale = [n for n in ingest.OUTPUTS if n not in ("images", "poses.json", "poses.csv", POOL, "analysis")]
    archived = ingest.archive_outputs(path, stale)
    images = os.path.join(path, "images")
    shutil.rmtree(images, ignore_errors=True)  # hard links only: the pool keeps the files
    os.makedirs(images)
    kept = [poses["images"][k] for k in keep]
    for im in kept:
        a, b = os.path.join(pool, "images", im["image"]), os.path.join(images, im["image"])
        try:
            os.link(a, b)
        except OSError:  # a file system without hard links
            shutil.copy2(a, b)
    full = len(keep) == len(poses["images"])
    out = dict(poses, images=kept)
    if not full:
        out["subsample"] = (dict(setting or {"method": "list"}, images_extracted=len(poses["images"]))
                            if names is not None else
                            {"every": max(1, int(every)), "offset": max(0, int(offset)), "start": start, "end": end,
                             "images_extracted": len(poses["images"])})
    json.dump(out, open(os.path.join(path, "poses.json"), "w"), indent=1)
    names = {im["image"] for im in kept}
    if os.path.exists(os.path.join(pool, "poses.csv")):
        with open(os.path.join(pool, "poses.csv"), newline="") as f, \
                open(os.path.join(path, "poses.csv"), "w", newline="") as g:
            rows, w = csv.reader(f), csv.writer(g)
            header = next(rows)
            w.writerow(header)
            col = header.index("image") if "image" in header else 0
            w.writerows(r for r in rows if r and r[col] in names)
    return len(keep), len(poses["images"]), archived


def _check_project(root, path):
    """Absolute path of a project folder directly under root; ValueError otherwise."""
    p = os.path.abspath(path)
    if os.path.dirname(p) != os.path.abspath(root) or not is_project(p):
        raise ValueError(f"not a project folder in {root}: {path}")
    return p


def rename(root, path, new_name):
    """Rename a project folder (same root). Paths inside it are relative, so nothing else changes;
    only old nerfstudio run configs keep the old absolute path (they are not reused)."""
    p = _check_project(root, path)
    _check_name(new_name)
    dst = os.path.join(os.path.abspath(root), new_name)
    if os.path.exists(dst):
        raise FileExistsError(f"{new_name} already exists")
    os.rename(p, dst)
    return dst


def delete(root, path):
    """Delete a project folder with everything in it (videos live in the shared library, not here).
    Returns the bytes freed."""
    p = _check_project(root, path)
    freed = size(p)
    shutil.rmtree(p)
    return freed


def size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _entries(path, table):
    out = []
    for pattern, desc in table:
        for p in sorted(glob.glob(os.path.join(path, pattern))):
            out.append({"path": os.path.relpath(p, path), "desc": desc, "bytes": size(p)})
    return out


def status(path):
    """Sources, stage progress and the final / intermediate files with their sizes."""
    sources, settings = ingest.load_plan(path)
    report = {}
    if os.path.exists(os.path.join(path, "report.json")):
        report = json.load(open(os.path.join(path, "report.json")))
    n_images = 0
    if os.path.exists(os.path.join(path, "poses.json")):
        n_images = len(json.load(open(os.path.join(path, "poses.json")))["images"])
    stages = [s for s in ("features", "pairs", "match", "refine", "export") if s in report]
    stages += [f"dense:{s}" for s in report.get("dense", {})]
    if os.path.exists(os.path.join(path, "splat", "splat_enu.ply")):
        stages.append("splat")
    return {"name": os.path.basename(os.path.normpath(path)), "path": path, "kind": ingest.load_kind(path),
            "sources": sources, "settings": settings,
            "images": n_images, "stages": stages, "final": _entries(path, FINAL),
            "intermediate": _entries(path, INTERMEDIATE)}


def clean(path, keep=KEEP_ON_CLEAN):
    """Delete intermediate results (not the images, sources or final results). Returns bytes freed."""
    freed = 0
    for e in _entries(path, INTERMEDIATE):
        if e["path"] in keep:
            continue
        p = os.path.join(path, e["path"])
        freed += e["bytes"]
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)
    return freed


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024

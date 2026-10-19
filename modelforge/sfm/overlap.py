"""Image overlap of a project's extracted images, and keyframe selection from it (Sparse tab →
Images used → Analyze / keyframes).

analyze (optional, minutes): SIFT features of every extracted image and verified matches between each
image and the next `window` of the same source, in PROJECT/analysis/ (a workspace with its own
features.db / database.db; images/ links to the extracted frames). The result, analysis/overlap.json,
holds every pair's inlier count and every image's keypoint count. A pair's overlap is
inliers / min(keypoints of the two): the share of features that see the same scene, measured on the
images themselves, so it needs no scene geometry (no ground plane) and works for any view direction.

select_keyframes walks each source in capture order: from the current keyframe it jumps to the farthest
of the next `window` images that overlaps it by at least min_overlap and also overlaps the keyframe
before by at least triple_overlap (so every point is seen by about 3 consecutive keyframes); when none
passes (blur, a fast turn) it takes the next image. Straight cruising gets few images, turns and close
passes many. The levels come from the calibration sweep (modelforge/subset_sweep.py), see LEVELS.
"""

import json
import os
import sqlite3
import time

from modelforge import jobs

DIR = "analysis"
WINDOW = 16  # pairs per image: the next WINDOW images of the same source
LEVELS = {"dense": 0.20, "normal": 0.12, "sparse": 0.06}  # min_overlap per level
# Calibrated on kladno_dopravnik (72 images, 6000 steps, 8 held-out photos): dense kept 57 images at the
# quality of all 72 (25.50 vs 25.45 dB, LPIPS 0.252 vs 0.250); normal 43 images -1.0 dB; sparse 32 images
# -2.6 dB (about every 2nd image's quality, which the user rated "OK"); every 4th image (18) -7.7 dB fails.
QUALITY = {"dense": "same as all images", "normal": "about 1 dB below all images",
           "sparse": "about 2.5 dB below: usable, softer"}
DEFAULT_LEVEL = "dense"
TRIPLE = 0.5  # triple_overlap = TRIPLE x min_overlap
PAIR_ID = 2147483647  # COLMAP: pair_id = id1 * PAIR_ID + id2 (id1 < id2)


def path(project):
    return os.path.join(project, DIR, "overlap.json")


def pool(project):
    """(images folder, poses.json path) of everything extracted: the pool if a subset is in use."""
    if os.path.exists(os.path.join(project, "extracted", "poses.json")):
        return os.path.join(project, "extracted", "images"), os.path.join(project, "extracted", "poses.json")
    return os.path.join(project, "images"), os.path.join(project, "poses.json")


def ordered(images):
    """Names per source in capture order: {source: [name, ...]}."""
    out = {}
    for k, im in sorted(enumerate(images), key=lambda e: (e[1].get("source", 0), e[1].get("time_s", 0.0), e[0])):
        out.setdefault(im.get("source", 0), []).append(im["image"])
    return out


def sequential_pairs(images, window=WINDOW):
    pairs = []
    for names in ordered(images).values():
        for i, a in enumerate(names):
            pairs += [(a, b) for b in names[i + 1:i + 1 + window]]
    return pairs


def analyze(project, kind="dji", window=WINDOW, token=None, log=print):
    """Features + sequential matches of all extracted images → analysis/overlap.json. Returns its dict."""
    from modelforge import pipeline
    img_dir, poses_path = pool(project)
    poses = json.load(open(poses_path))
    ws = os.path.join(project, DIR)
    os.makedirs(ws, exist_ok=True)
    link = os.path.join(ws, "images")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.relpath(img_dir, ws), link)
    json.dump(poses, open(os.path.join(ws, "poses.json"), "w"))
    for f in ("camera.json",):  # known intrinsics (ROS camera_info)
        if os.path.exists(os.path.join(project, f)):
            json.dump(json.load(open(os.path.join(project, f))), open(os.path.join(ws, f), "w"))
    t0 = time.time()
    report = pipeline.Report(ws)
    log(f"analysis: features of {len(poses['images'])} images")
    step = [0]

    def progress(k, n, name):  # a log line every ~10 %
        if k >= step[0] or k == n:
            log(f"analysis: features {k} / {n}")
            step[0] = k + max(1, n // 10)
    pipeline.stage_features(ws, report, kind=kind, progress=progress)
    jobs.check(token)
    pairs = sequential_pairs(poses["images"], window)
    with open(os.path.join(ws, "pairs.txt"), "w") as f:
        f.writelines(f"{a} {b}\n" for a, b in pairs)
    log(f"analysis: matching {len(pairs)} pairs (each image with the next {window})")
    pipeline.stage_match(ws, report, token=token)
    out = read_overlap(os.path.join(ws, "database.db"))
    out.update(window=window, images_total=len(poses["images"]), seconds=round(time.time() - t0, 1))
    json.dump(out, open(path(project), "w"))
    log(f"analysis: done in {out['seconds']:.0f} s")
    return out


def read_overlap(db_path):
    """{"keypoints": {name: n}, "pairs": [[a, b, inliers], ...]} from a COLMAP database."""
    con = sqlite3.connect(db_path)
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        kps = {names[i]: n for i, n in con.execute("SELECT image_id, rows FROM keypoints")}
        pairs = []
        for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
            i1, i2 = divmod(pid, PAIR_ID)
            if i1 in names and i2 in names:
                pairs.append([names[i1], names[i2], int(rows)])
    finally:
        con.close()
    return {"keypoints": kps, "pairs": pairs}


def load(project):
    try:
        return json.load(open(path(project)))
    except (OSError, ValueError):
        return None


class Overlap:
    """Pair overlap lookups from an overlap.json dict: inliers / min(keypoints)."""

    def __init__(self, data):
        self.kp = data["keypoints"]
        self.inl = {}
        for a, b, n in data["pairs"]:
            self.inl[(a, b)] = self.inl[(b, a)] = n
        self.window = data.get("window", WINDOW)

    def __call__(self, a, b):
        n = self.inl.get((a, b), 0)
        return n / max(1, min(self.kp.get(a, 0), self.kp.get(b, 0))) if n else 0.0


def select_keyframes(images, data, min_overlap, triple_overlap=None, window=None):
    """Names of the keyframes (see the module doc), in poses.json order."""
    ov = Overlap(data)
    window = window or ov.window
    triple = TRIPLE * min_overlap if triple_overlap is None else triple_overlap
    keep = set()
    for names in ordered(images).values():
        kf = [0]
        while kf[-1] < len(names) - 1:
            c = kf[-1]
            nxt = c + 1
            for j in range(c + 1, min(len(names), c + 1 + window)):
                if ov(names[c], names[j]) >= min_overlap and (
                        len(kf) < 2 or ov(names[kf[-2]], names[j]) >= triple):
                    nxt = j
            kf.append(nxt)
        keep.update(names[k] for k in kf)
    return [im["image"] for im in images if im["image"] in keep]


def overlap_stats(images, data, names):
    """Overlap between consecutive kept images (and every other one) per source: min / p5 / median."""
    import numpy as np
    ov, kept = Overlap(data), set(names)
    one, two = [], []
    for seq in ordered(images).values():
        s = [n for n in seq if n in kept]
        one += [ov(a, b) for a, b in zip(s, s[1:])]
        two += [ov(a, b) for a, b in zip(s, s[2:])]

    def st(v):
        return {"min": round(float(np.min(v)), 3), "p5": round(float(np.percentile(v, 5)), 3),
                "median": round(float(np.median(v)), 3)} if v else None
    return {"consecutive": st(one), "skip_one": st(two)}

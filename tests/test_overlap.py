"""Overlap analysis and keyframe selection (modelforge/sfm/overlap.py) on synthetic overlap tables."""

import json
import os
import sqlite3

from modelforge import projects
from modelforge.sfm import overlap


def images(n, sources=1):
    return [{"image": f"s{s}_{i:03d}.jpg", "source": s, "time_s": float(i)} for s in range(sources) for i in range(n)]


def table(ims, share):
    """overlap.json-like dict: share(i, j) = overlap fraction between images i < j of a source."""
    kp = {im["image"]: 1000 for im in ims}
    pairs = []
    for seq in overlap.ordered(ims).values():
        for i, a in enumerate(seq):
            for j in range(i + 1, min(len(seq), i + 1 + overlap.WINDOW)):
                v = share(i, j)
                if v > 0:
                    pairs.append([a, seq[j], int(1000 * v)])
    return {"keypoints": kp, "pairs": pairs, "window": overlap.WINDOW}


def test_keyframes_follow_overlap():
    ims = images(40)
    # overlap decays with the gap; slow part (images 0-19) decays 3x slower than the fast part
    def share(i, j):
        gap = j - i
        return max(0.0, 0.4 - (0.04 if i < 20 else 0.12) * (gap - 1))
    data = table(ims, share)
    kf = overlap.select_keyframes(ims, data, min_overlap=0.2)
    idx = [int(n[3:6]) for n in kf]
    assert idx[0] == 0 and idx[-1] == 39  # both ends kept
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    slow = [g for a, g in zip(idx, gaps) if a + g < 20]
    fast = [g for a, g in zip(idx, gaps) if a >= 20]
    assert sum(slow) / len(slow) > sum(fast) / len(fast)  # fewer images where the view changes slowly
    st = overlap.overlap_stats(ims, data, kf)["consecutive"]
    assert st["min"] >= 0.2 - 1e-9  # every neighbour pair keeps the asked overlap
    assert len(overlap.select_keyframes(ims, data, 0.3)) > len(kf) > len(overlap.select_keyframes(ims, data, 0.1))


def test_keyframes_take_the_next_image_when_nothing_overlaps():
    ims = images(10)
    data = table(ims, lambda i, j: 0.0)
    assert overlap.select_keyframes(ims, data, 0.1) == [im["image"] for im in ims]


def test_triple_overlap_and_sources():
    ims = images(20, sources=2)
    data = table(ims, lambda i, j: max(0.0, 0.6 - 0.1 * (j - i - 1)))  # 0.3 up to 4 apart, 0.15 up to 5
    kf = overlap.select_keyframes(ims, data, 0.3, triple_overlap=0.15)
    for s in (0, 1):
        idx = [int(n[3:6]) for n in kf if n.startswith(f"s{s}")]
        assert idx[0] == 0 and idx[-1] == 19
        assert all(b - a <= 4 for a, b in zip(idx, idx[1:]))
        assert all(c - a <= 6 for a, c in zip(idx, idx[2:]))  # every keyframe overlaps the one two back


def test_read_overlap_and_apply_list(tmp_path):
    db = str(tmp_path / "database.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    con.executemany("INSERT INTO images VALUES (?, ?)", [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")])
    con.executemany("INSERT INTO keypoints VALUES (?, ?)", [(1, 100), (2, 200), (3, 50)])
    con.executemany("INSERT INTO two_view_geometries VALUES (?, ?)",
                    [(1 * overlap.PAIR_ID + 2, 40), (2 * overlap.PAIR_ID + 3, 10)])
    con.commit()
    con.close()
    d = overlap.read_overlap(db)
    assert d["keypoints"] == {"a.jpg": 100, "b.jpg": 200, "c.jpg": 50}
    ov = overlap.Overlap(d)
    assert ov("a.jpg", "b.jpg") == ov("b.jpg", "a.jpg") == 0.4 and ov("b.jpg", "c.jpg") == 0.2
    assert ov("a.jpg", "c.jpg") == 0.0

    p = projects.create(str(tmp_path), "proj")
    ims = images(6)
    os.makedirs(os.path.join(p, "images"))
    for im in ims:
        open(os.path.join(p, "images", im["image"]), "w").write("x")
    json.dump({"images": ims}, open(os.path.join(p, "poses.json"), "w"))
    names = ["s0_000.jpg", "s0_003.jpg", "s0_005.jpg"]
    n, total, _ = projects.apply_subsample(p, names=names, setting={"method": "keyframes", "level": "normal"})
    assert (n, total) == (3, 6) and sorted(os.listdir(os.path.join(p, "images"))) == names
    info = projects.subsample_info(p)
    assert info["setting"]["method"] == "keyframes" and info["setting"]["images_extracted"] == 6


def test_feature_progress(tmp_path):
    """features.extract reports images as COLMAP finishes them (it has no callback: the database is polled)."""
    import numpy as np
    from PIL import Image
    from modelforge.sfm import features
    os.makedirs(tmp_path / "images")
    rng = np.random.default_rng(0)
    ims = []
    for i in range(12):
        name = f"im_{i:02d}.jpg"
        Image.fromarray(rng.integers(0, 255, (240, 320, 3), np.uint8)).save(tmp_path / "images" / name)
        ims.append({"image": name})
    json.dump({"images": ims}, open(tmp_path / "poses.json", "w"))
    seen = []
    stats = features.extract(str(tmp_path), progress=lambda k, n, name: seen.append((k, n, name)))
    assert stats["images"] == 12 and seen[-1] == (12, 12, "im_11.jpg")
    assert all(n == 12 for _, n, _ in seen) and [k for k, _, _ in seen] == sorted(k for k, _, _ in seen)

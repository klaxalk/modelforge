"""The learned sun visibility's segments (splat.sun_segments) and its training flags (docs/todo_lighting.md)."""
import json
import os

from modelforge import lighting, splat


def _project(tmp_path, monkeypatch):
    p = tmp_path / "proj"
    (p / "images").mkdir(parents=True)
    images = []
    suns = {}
    # source 0: 25 min (3 segments of <= 10 min), source 1: 5 min (1 segment), one photo without a time
    for src, (t0, n, dt) in enumerate([(1000.0, 26, 60.0), (90000.0, 6, 60.0)]):
        for i in range(n):
            name = f"s{src}_{i:03d}.jpg"
            images.append({"image": name, "source": src})
            suns[name] = {"utc": t0 + i * dt, "azimuth": 200.0, "elevation": 30.0, "source": src}
    images.append({"image": "untimed.jpg", "source": 1})
    json.dump({"images": images, "sources": [{"video": "a.MP4"}, {"video": "b.MP4"}]}, open(p / "poses.json", "w"))
    monkeypatch.setattr(lighting, "sun_per_photo", lambda root: suns)
    # the look: source 1 has the least hard shadow
    os.makedirs(p / lighting.DIR)
    json.dump({"sources": [{"source": 0, "dark": 0.03}, {"source": 1, "dark": 0.01}]},
              open(p / lighting.DIR / "survey.json", "w"))
    return str(p)


def test_sun_segments(tmp_path, monkeypatch):
    p = _project(tmp_path, monkeypatch)
    out = splat.sun_segments(p, on_line=lambda m: None)
    segs = json.load(open(os.path.join(out, "segments.json")))
    assert "untimed.jpg" not in segs
    s0 = [segs[f"s0_{i:03d}.jpg"] for i in range(26)]
    assert sorted(set(s0)) == [0, 1, 2] and s0 == sorted(s0)  # capture order, 3 pieces of <= 10 min
    assert {segs[f"s1_{i:03d}.jpg"] for i in range(6)} == {3}
    ref = json.load(open(os.path.join(out, "ref.json")))
    assert set(ref) == {f"s1_{i:03d}.jpg" for i in range(6)} | {"untimed.jpg"}


def test_sun_learned_command():
    cmd = splat.train_command("/p", lidar_depth=0.0, regularize={"sun": "learned"}, sun_maps="/p/lighting/learned")
    assert "--pipeline.model.sun-learned" in cmd and "--sun-maps" in cmd
    cmd = splat.train_command("/p", lidar_depth=0.0, regularize={"sun": True}, sun_maps="/p/sfmmesh/sun_x2")
    assert "--pipeline.model.sun-light" in cmd and "--pipeline.model.sun-learned" not in cmd

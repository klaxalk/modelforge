"""sfm.isolated: photos no other pass ties in are left out of the export only when their ground copy is off."""
import json
import os
import shutil
from types import SimpleNamespace as NS

import numpy as np
import pytest

from modelforge.sfm import isolated


class FakeRec:
    """The parts of pycolmap.Reconstruction that isolated.find / remove use."""

    def __init__(self):
        self.images, self.points, self.obs = {}, {}, {}

    def add_image(self, i, name):
        self.images[i] = NS(image_id=i, name=name, get_observation_points2D=lambda i=i: [
            NS(point3D_id=p) for p in self.obs.get(i, [])])

    def add_point(self, pid, xyz, images):
        self.points[pid] = NS(xyz=np.asarray(xyz, float), track=NS(elements=[NS(image_id=i) for i in images]))
        for i in images:
            self.obs.setdefault(i, []).append(pid)

    def reg_image_ids(self):
        return list(self.images)

    def image(self, i):
        return self.images[i]

    def num_reg_images(self):
        return len(self.images)

    def num_points3D(self):
        return len(self.points)

    @property
    def points3D(self):
        return self.points

    def find_image_with_name(self, name):
        return next(im for im in self.images.values() if im.name == name)

    def delete_point3D(self, pid):
        del self.points[pid]


def scene():
    """Ground 20 x 20 m seen by pass A (t 0-9) and pass B (t 100-109): every ground point links them. Pass C (t 200)
    triangulated its own copy 1.5 m above it, pass D (t 300) its own copy on it (5 cm), pass E (t 400) ground nobody
    else saw."""
    rng = np.random.default_rng(0)
    rec, times, pid = FakeRec(), {}, 0
    for k, (label, t0) in enumerate((("a", 0), ("b", 100), ("c", 200), ("d", 300), ("e", 400))):
        for j in range(10):
            rec.add_image(10 * k + j, f"{label}{j}")
            times[f"{label}{j}"] = t0 + j
    gx, gy = np.meshgrid(np.arange(0, 20, 0.25), np.arange(0, 20, 0.25))
    ground = np.c_[gx.ravel(), gy.ravel(), rng.normal(0, 0.05, gx.size)]
    strip = (ground[:, 0] // 2).astype(int)  # image j of a pass sees the strip 2j..2j+2 m
    for x, s in zip(ground, strip):
        rec.add_point(pid, x, [s, 10 + s])
        pid += 1
    for k, dz, dx in ((2, 1.5, 0.0), (3, 0.05, 0.0), (4, 0.0, 50.0)):
        for x, s in zip(ground[::2], strip[::2]):
            rec.add_point(pid, x + [dx, 0, dz], [10 * k + s, 10 * k + (s + 1) % 10])
            pid += 1
    return rec, times


def test_only_offset_copies_are_dropped():
    rec, times = scene()
    r = isolated.find(rec, times)
    assert r["isolated"] == 30  # passes C, D and E link to no other pass
    assert sorted(r["drop"]) == [f"c{j}" for j in range(10)]  # only C's ground is off (1.5 m); E's is its own
    assert len(r["stretches"]) == 1 and r["stretches"][0][2] == 10 and r["stretches"][0][3] > 0.9
    n_before = len(rec.points)
    removed = isolated.remove(rec, r["drop"])
    assert removed == n_before - len(rec.points) > 0
    assert all(p.xyz[2] < 0.5 for p in rec.points.values())  # C's copy is gone, everything else stays


def test_a_tilted_copy_is_dropped():
    """A copy whose middle lies on the ground but whose ends are 2 m off (median offset ~0) goes too."""
    rec, times = scene()
    for p in rec.points.values():
        if any(e.image_id // 10 == 3 for e in p.track.elements):  # pass D: tilt it about its middle
            p.xyz = p.xyz + [0, 0, (p.xyz[0] - 10) * 0.25]
    assert "d0" in isolated.find(rec, times)["drop"]


def test_different_surfaces_are_not_a_copy():
    """Pass C sees the ground under a canopy (or a hall's floor under its roof) that A and B saw from above, plus
    open ground beside it: its cells are off, but on the open ground it agrees, so it stays. With its open ground
    1.5 m off as well it goes; with only cells 10 m off (no open ground) it stays: floor under roof."""
    def build(open_dz, roof_z=8.0, with_open=True):
        rec, times = scene()
        times = {n: t for n, t in times.items() if n[0] in "abc"}  # D and E out: A and B are the reference
        for p in rec.points.values():
            el = {e.image_id // 10 for e in p.track.elements}
            if el <= {0, 1} and p.xyz[1] < 12:  # A and B: a canopy / roof over y < 12
                p.xyz = p.xyz + [0, 0, roof_z]
            if el == {2}:  # C: its copy at the ground (C's own +1.5 m of the scene removed)
                p.xyz = p.xyz - [0, 0, 1.5] + [0, 0, open_dz if p.xyz[1] >= 12 else 0.0]
                if not with_open and p.xyz[1] >= 12:
                    p.xyz = p.xyz + [0, 0, roof_z]  # all of C under the roof: nothing open to compare
        return rec, times
    rec, times = build(open_dz=0.03)
    r = isolated.find(rec, times)
    assert r["candidates"] >= 10 and not any(n.startswith("c") for n in r["drop"])
    rec, times = build(open_dz=1.5)
    assert sorted(n for n in isolated.find(rec, times)["drop"] if n.startswith("c")) == [f"c{j}" for j in range(10)]
    rec, times = build(open_dz=0.0, roof_z=10.0, with_open=False)
    assert not any(n.startswith("c") for n in isolated.find(rec, times)["drop"])


def test_flight_without_revisits_drops_nothing():
    rec, times = scene()
    one_pass = {n: t for n, t in times.items() if n[0] in "a"}  # images without a time are ignored
    r = isolated.find(rec, one_pass)
    assert r["drop"] == [] and r["linking_points"] == 0 and r["isolated"] == 10  # weakly tied, but no reference


REAL = os.environ.get("MODELFORGE_TEST_PROJECT", os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")), reason="needs a matched, refined project")
def test_export_triangulates_without_the_dropped_photos(tmp_path, monkeypatch):
    """The export leaves the photos out of transforms.json and triangulates the points again from the rest."""
    import pycolmap
    from modelforge import pipeline
    p = tmp_path / "p"
    (p / "sparse").mkdir(parents=True)
    shutil.copytree(os.path.join(REAL, "sparse", "refined"), p / "sparse" / "refined")
    for f in ("database.db", "poses.json"):
        shutil.copy(os.path.join(REAL, f), p / f)
    os.symlink(os.path.abspath(os.path.join(REAL, "images")), p / "images")
    names = sorted(r["image"] for r in json.load(open(p / "poses.json"))["images"])
    drop = names[10:16]
    monkeypatch.setattr(isolated, "find", lambda rec, times, **kw: {
        "drop": drop, "isolated": len(drop), "photos": rec.num_reg_images(), "linking_points": 0,
        "stretches": [(0.0, 1.0, len(drop), 0.5)]})
    seen = []
    pipeline.stage_export(str(p), pipeline.Report(str(p)), progress=lambda text, k, n: seen.append((text, k, n)))
    tri = [(k, n) for text, k, n in seen if text.startswith("triangulating")]
    assert tri and tri[-1] == (len(names) - len(drop), len(names) - len(drop))  # read from COLMAP's log
    assert any(text.startswith("refining points") for text, _, _ in seen) and seen[-1][0].startswith("writing")
    t = json.load(open(p / "transforms.json"))
    assert len(t["frames"]) == len(names) - len(drop)
    assert not {f["file_path"].split("/")[-1] for f in t["frames"]} & set(drop)
    rec = pycolmap.Reconstruction(str(p / "sparse" / "export"))
    seen = {rec.image(e.image_id).name for pt in rec.points3D.values() for e in pt.track.elements}
    assert rec.num_points3D() > 1000 and not seen & set(drop)  # no point keeps an observation of them
    iso = json.load(open(p / "report.json"))["export"]["isolated"]
    assert iso["triangulated"] and iso["dropped"] == len(drop) and iso["points"] == rec.num_points3D()
    pipeline.stage_export(str(p), pipeline.Report(str(p)), drop_isolated=False)  # nothing left out this time
    assert not (p / "sparse" / "export").exists()  # no stale model for the SfM mesh to pick up

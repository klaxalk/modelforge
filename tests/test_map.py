"""The 2D map (modelforge/map2d.py, gui/map.py), its geodesy (modelforge/geo.py), the orthophoto's GeoTIFF
(modelforge/ortho.py) and the Ingest rectangle (dji/ingest.sources_in_area)."""

import json
import os
import shutil
import time

import numpy as np
import pytest

from modelforge import geo, map2d, ortho
from modelforge.dji import ingest

ORIGIN = (50.0755, 14.4378, 300.0)


def test_enu_round_trip_and_utm():
    xyz = np.array([[0.0, 0.0, 0.0], [350.0, -120.0, 40.0], [-2000.0, 1500.0, -5.0]])
    lat, lon, alt = geo.enu_to_lla(xyz, ORIGIN)
    assert np.allclose(geo.lla_to_enu(lat, lon, alt, ORIGIN), xyz, atol=1e-3)
    assert abs(lat[0] - ORIGIN[0]) < 1e-9 and abs(alt[0] - ORIGIN[2]) < 1e-3
    e, n = geo.lla_to_utm(50.0, 15.0, 33)  # on zone 33's central meridian
    assert abs(e - 500000.0) < 1e-3 and abs(n - 5538630.70) < 0.05
    assert geo.utm_zone(50.0755, 14.4378) == (33, True, 32633)
    x, y = geo.mercator_px(0.0, 0.0, 1)
    assert np.allclose([x, y], [256, 256])


def test_view_runs_polygon():
    v = map2d.View(100.0, 50.0, 0.5, 640, 640)
    assert np.allclose(v.to_px([[100.0, 50.0]]), [[320, 320]])
    assert np.allclose(v.to_world(*v.to_px([[110.0, 40.0]])[0]), [110.0, 40.0])  # north up: y falls downwards
    assert v.to_px([[100.0, 60.0]])[0, 1] < 320
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert list(map2d.inside_polygon([[5, 5], [11, 5], [-1, 3]], sq)) == [True, False, False]
    t = np.arange(0, 60, 0.5)
    m = ((t >= 5) & (t < 20)) | ((t >= 21.5) & (t < 30)) | ((t >= 40) & (t < 42))  # a 1.5 s gap, a 2 s stretch
    assert map2d.runs(m, t, min_s=5, gap_s=3) == [(5.0, 29.5)]


def test_basemap_uses_cached_tiles(tmp_path):
    cache = map2d.TileCache(str(tmp_path))
    view = map2d.View(0.0, 0.0, 0.3, 64, 64)
    z = int(np.ceil(np.log2(geo.mercator_mpp(ORIGIN[0], 0) / 0.3)))
    lat, lon, _ = geo.enu_to_lla(np.array([[0.0, 0.0]]), ORIGIN)
    x, y = geo.mercator_px(lat, lon, z)
    import cv2
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            p = cache.path("aerial", z, int(x[0] // 256) + dx, int(y[0] // 256) + dy)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, np.full((256, 256, 3), (30, 60, 200), np.uint8))  # BGR on disk
    img, complete = map2d.basemap(view, ORIGIN, "aerial", cache)
    assert complete and np.allclose(img[32, 32], (200, 60, 30), atol=2)


def test_coverage_counts_photos_looking_down():
    C = np.array([[0.0, 0.0, 50.0], [20.0, 0.0, 50.0]])
    R = np.repeat(np.diag([1.0, -1.0, -1.0])[None], 2, 0)  # OpenCV camera looking straight down
    grid, (x0, y0, cell) = map2d.coverage(C, R, (500.0, 500.0, 320.0, 240.0), (640, 480),
                                          lambda q: np.zeros(len(q)), (-60, -60, 80, 60), 10.0)
    at = lambda x, y: grid[int((y - y0) // cell), int((x - x0) // cell)]  # noqa: E731
    assert at(10, 0) == 2 and at(-25, 0) == 1 and at(-55, 0) == 0


def test_geotiff(tmp_path):
    from PIL import Image
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "splat"))
    open(ortho.splat_path(p), "wb").write(b"x")
    img = np.zeros((40, 60, 4), np.uint8)
    img[..., 1] = 200
    img[..., 3] = 255
    import cv2
    cv2.imwrite(os.path.join(p, ortho.PNG), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    json.dump({"bounds": [-30.0, -20.0, 30.0, 20.0], "mpp": 1.0, "splat": ortho._key(ortho.splat_path(p))},
              open(os.path.join(p, ortho.META), "w"))
    out, epsg = ortho.geotiff(p, os.path.join(p, "export", "ortho.tif"), origin=ORIGIN)
    assert epsg == 32633
    with Image.open(out) as im:
        m = im.tag_v2[34264]
        assert im.size == (60, 40) and im.tag_v2[34735][-1] == 32633
    e0, n0 = geo.lla_to_utm(*geo.enu_to_lla(np.array([[-30.0, 20.0]]), ORIGIN)[:2], 33)
    assert abs(m[3] - e0[0]) < 0.01 and abs(m[7] - n0[0]) < 0.01  # pixel (0, 0) = the north-west corner
    assert abs(np.hypot(m[0], m[4]) - 1.0) < 1e-3  # 1 m per pixel, rotated a little (meridian convergence)
    open(ortho.splat_path(p), "wb").write(b"xy")  # a new splat: stale
    assert ortho.current(p) is None


class FakeClip:
    """A flight east along y = 0 at 30 m, camera looking north 45° down, 1 frame per second."""

    def __init__(self):
        n = 100
        self.pts = np.arange(n, dtype=float)
        e = np.arange(n) * 5.0 - 250.0
        lat, lon, _ = geo.enu_to_lla(np.c_[e, np.zeros(n)], ORIGIN)
        self.tel = {"a": np.c_[lat, lon, np.full(n, 30.0)], "yaw": np.zeros(n), "pitch": np.full(n, -45.0),
                    "valid": np.ones(n, bool)}

    def telemetry(self, yaw_source="auto"):
        return self.tel


def test_sources_from_a_rectangle(monkeypatch):
    clip = FakeClip()
    monkeypatch.setattr(ingest, "load_clip", lambda *a, **k: clip)
    src = [ingest.Source("a.mp4", 0, 99, lead=3.0, alt_offset=1.5), ingest.Source("a.mp4", 10, 20)]
    rect = [(-50, 20), (50, 20), (50, 40), (-50, 40)]  # north of the path: what the camera looks at (30 m ahead)
    out, found = ingest.sources_in_area(src, ingest.Settings(), rect, ORIGIN, step=1)
    assert len(out) == 1 and out[0].lead == 3.0 and out[0].alt_offset == 1.5  # one piece, the video's settings kept
    assert abs(out[0].start - 40) <= 1 and abs(out[0].end - 60) <= 1  # e = -50 .. 50
    out, _ = ingest.sources_in_area(src, ingest.Settings(), rect, ORIGIN, by="position", step=1)
    assert out == []  # the drone itself never enters it


REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project (MODELFORGE_TEST_PROJECT)")
def test_map_window(tmp_path):
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path / "proj")
    os.makedirs(p)
    for name in ("poses.json", "images", "sparse", "transforms.json", "sparse_pc.ply"):
        if os.path.exists(os.path.join(REAL, name)):
            os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))
    a = App(p, port=free_port())
    try:
        m = a.map
        m.g_bg.value = "none"  # no network in tests
        m.g_heat.value = "photos per area"

        def drawn(cond, what):
            for _ in range(200):
                m.redraw()
                time.sleep(0.05)
                if cond():
                    return
            raise AssertionError(what)
        drawn(lambda: m.g_img.content.startswith("<img") and m._base_key is not None and ("cover", id(a.snap))
              in m._cache, "map with the coverage heat map")
        assert "photos per area" in m.g_md.content and "m across" in m.g_md.content
        v0 = m.view
        m.g_mode.value = "Centre the map here"
        m._clicked("0.75,0.5,1", None)
        assert m.view.cx > v0.cx and abs(m.view.cy - v0.cy) < 1e-6
        # an area: three corners around the cameras' middle, then Apply: Images used and Register get it
        m.g_mode.value = "Draw an area (photos)"
        for f in ("0.2,0.2,1", "0.8,0.2,2", "0.8,0.8,3", "0.2,0.8,4"):
            m._clicked(f, None)
        assert len(m.draft) == 4
        m._draw_action("Apply")
        drawn(lambda: a.subset.sel.area is not None, "the area applied")
        names = a.subset.sel.area["names"]
        assert names and a.register.sel.area["names"] == names
        a.subset.sel.g_mode.value = "every N-th image"  # keyframes would need the overlap analysis
        args = a.subset.sel.args(a.subset._images, a.subset._overlap)
        assert set(args["names"]) <= names and args["setting"]["area"]
        assert "in the map's area" in a.subset.sel.describe()
        m._draw_action("Clear")
        assert a.subset.sel.area is None and not a.subset.sel.g_area.visible
        # the quick bar opens and closes it
        a._quick_click("Map", None)
        assert a.map_open
        a._quick_click("Map", None)
        assert not a.map_open
    finally:
        a.server.stop()

"""GUI dense/splat layers and dense stage runs, on a temp project that links the real
priors/images/sparse model and carries synthetic dense outputs and a synthetic splat."""

import os
import shutil
import socket
import threading
import time

import numpy as np
import plyfile
import pytest

from modelforge import dense
from modelforge.gui import app as gapp
from tests.test_dense_chain import make_workspace
from tests.test_splat_to_enu import make_splat

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",  # any refined project
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                                reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("gui_dense"))
    for name in ("poses.json", "images", "sparse", "transforms.json", "sparse_pc.ply"):
        if os.path.exists(os.path.join(REAL, name)):
            os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))  # stages write to it
    make_workspace(p)
    dense.fusion(p)
    dense.mesh(p, "delaunay")
    dense.preview(p, "delaunay", max_faces=2000)
    dense.texture(p, "delaunay")
    v, _ = make_splat(np.random.default_rng(0), n=500)
    os.makedirs(os.path.join(p, "splat"))
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(os.path.join(p, "splat", "splat_enu.ply"))
    return p


@pytest.fixture(scope="module")
def app(project):
    from modelforge.gui.app import App
    a = App(project, port=free_port())
    yield a
    a.server.stop()


def test_layers_become_available(app):
    assert not app.g_dense_pts.disabled and not app.g_splat.disabled and not app.g_mesh.disabled
    assert "delaunay" in app.g_mesh.options
    assert "dense points" in app.g_dense_md.content and "splat" in app.g_dense_md.content


def test_toggle_layers(app):
    app.g_dense_pts.value = True
    app._draw_dense_points()
    assert "dense points:" in app.g_dense_md.content
    app.g_mesh.value = "delaunay"
    app._draw_mesh()
    assert "faces, textured" in app.g_dense_md.content  # the coloured mesh from the texture stage
    app.g_mesh_tex.value = False
    app._draw_mesh()
    assert "untextured preview" in app.g_dense_md.content
    app.g_mesh_tex.value = True
    app.g_splat.value = True
    app._draw_splat()
    assert "500 gaussians" in app.g_dense_md.content
    for h in (app.g_dense_pts, app.g_splat):
        h.value = False
    app.g_mesh.value = "off"
    app._refresh_dense()  # hiding paths must not fail either


def test_run_mesh_stage_from_gui(app, project):
    os.remove(os.path.join(project, "dense", "meshed-delaunay.ply"))
    app.g_mesh_method.value = "delaunay"
    app.run_stage("mesh")
    t0 = time.time()
    while app.running and time.time() - t0 < 120:
        time.sleep(0.2)
    assert "finished" in app.g_state.content, app.g_state.content
    assert os.path.exists(os.path.join(project, "dense", "meshed-delaunay.ply"))
    assert os.path.exists(os.path.join(project, "dense", "preview-delaunay.ply"))


def test_run_a_stage_list(app, project):
    """A tuple of stages (main's --run) runs as one job: here the SfM mesh, then its holes patched."""
    app.run_stage(("sfmmesh", "sfmmesh-fill"))
    t0 = time.time()
    while app.running and time.time() - t0 < 300:
        time.sleep(0.2)
    assert "sfmmesh + sfmmesh-fill finished" in app.g_state.content, app.g_state.content
    assert os.path.exists(os.path.join(project, "sfmmesh", "mesh.ply"))
    assert os.path.exists(os.path.join(project, "sfmmesh", "patched.npy"))


def test_a_double_click_starts_one_job(app, monkeypatch):
    """Two clicks at once (a double click, two browsers) start one job: the busy check and the claim are atomic, also
    while the memory estimate takes its time (berlin_all: tile 11 trained twice at once, one OOM-killed)."""
    from modelforge import memory
    started = []

    def slow_estimate(*a, **k):
        time.sleep(0.3)
        return 0.0, "test"
    monkeypatch.setattr(memory, "estimate", slow_estimate)
    monkeypatch.setattr(app, "_run", lambda stage, run_id: (started.append(stage), time.sleep(0.5),
                                                            setattr(app, "running", False)))
    clicks = [threading.Thread(target=app.run_stage, args=("export",)) for _ in range(2)]
    for c in clicks:
        c.start()
    for c in clicks:
        c.join()
    t0 = time.time()
    while app.running and time.time() - t0 < 5:
        time.sleep(0.05)
    assert started == ["export"] and "busy" in app.g_state.content
    for b in app.g_buttons.values():
        b.disabled = False


def test_camera_exposure_in_the_splat_tab(app, project):
    """Splat tab → Camera exposure: the project's cameras, a correction stored per project and passed to the splat's
    regularize; 0 removes it."""
    from modelforge import exposure
    cams = exposure.cameras(project)
    app._refresh_camera_exposure()
    assert list(app.g_cam_exp_camera.options) == cams and app.g_cam_exp_stops.value == 0
    app.g_cam_exp_camera.value = cams[-1]
    app.g_cam_exp_stops.value = 0.75
    app._set_camera_exposure()  # what the number's on_update does
    assert exposure.stops(project) == {cams[-1]: 0.75} and "+0.75" in app.g_cam_exp_note.content
    assert app._regularize()["exposure_stops"] == {cams[-1]: 0.75}
    app.g_cam_exp_stops.value = 0.0
    app._set_camera_exposure()
    assert exposure.stops(project) == {} and app._regularize().get("exposure_stops", {}) == {}


def test_features_progress_in_the_state_line(app):
    """Feature extraction shows "image k / n · name" while it runs (COLMAP's database is polled)."""
    details = []
    real = app._on_features_progress
    app._on_features_progress = lambda k, n, name: (real(k, n, name), details.append(app._job["detail"]))
    try:
        app.run_stage("features")
        t0 = time.time()
        while app.running and time.time() - t0 < 300:
            time.sleep(0.1)
    finally:
        app._on_features_progress = real
    n = len(app.project.names)
    assert "finished" in app.g_state.content, app.g_state.content
    assert details and details[-1].startswith(f"image {n:,} / {n:,}") and "`" in details[-1]


def test_refine_summary_tells_wrong_priors_from_little_overlap(app):
    """0 points with plenty of inlier matches = priors that don't fit the images (franta: a camera mount off
    by yaw 180); with few matches it is too little overlap."""
    final = {"reg_images": 200, "points3D": 0}
    wrong = app._stage_summary("refine", {"refine": {"final": final}, "match": {"inlier_matches": 4_554_039}})
    sparse = app._stage_summary("refine", {"refine": {"final": final}, "match": {"inlier_matches": 1074}})
    assert "do not fit the images" in wrong and "too little overlap" in sparse


def test_full_pipeline_includes_semantic(app):
    """Run full pipeline ends with the semantic features when the Search tab says so (and RADIO is there);
    the History line summarizes them."""
    from modelforge import semantic as sem
    assert app.semantic.g_in_all.value  # on by default
    line = app._stage_summary("semantic", {"semantic": {"seen": 900283, "gaussians": 1018377, "photos": 56,
                                                         "version": "c-radio_v3-l", "width": 2048, "seconds": 37.7}})
    assert "900,283 of 1,018,377" in line and "56 photos" in line
    assert sem.available() is None or "RADIO" in sem.available() or "nerfstudio" in sem.available()


def test_tab_defaults(app):
    assert app.g_upright.value and not app.g_mapper.visible  # DJI project: upright SIFT, refine (no mapper)
    assert gapp.DENSE_QUALITY[app.g_dense_quality.value] == "superfast"  # the default (320 px)
    assert not app.g_dense_in_all.value  # the full pipeline's dense result is the splat mesh
    assert app.g_splat_cache.value == "auto" and app.g_splat_down.value == "auto"
    assert app.g_splat_strategy.value == "auto"
    assert app.g_splat_sunlight.value == "off" and not (app._regularize() or {}).get("sun")
    app.g_splat_sunlight.value = "learned per Gaussian"  # docs/todo_lighting.md: the sun visibility learned
    assert app._regularize()["sun"] == "learned"
    app.g_splat_sunlight.value = "off"
    app._refresh_splat_plan()  # auto values from the cameras, shown in the Splat tab
    t0 = time.time()
    while "Plan" not in app.g_splat_plan.content and time.time() - t0 < 30:
        time.sleep(0.1)
    assert "steps" in app.g_splat_plan.content and "GSD" in app.g_splat_plan.content, app.g_splat_plan.content


def test_export_tab(app, project):
    app._refresh_dense()
    assert not app.g_export_buttons["mesh"].disabled and not app.g_export_buttons["splat"].disabled
    assert app.g_export_mesh.options == ("delaunay",)

    class Client:
        sent = []

        def send_file_download(self, name, data):
            self.sent.append((name, len(data)))

    def wait():
        t0 = time.time()
        while app._exporting and time.time() - t0 < 60:
            time.sleep(0.1)

    app.g_export_mesh_fmt.value = "glb"
    app._export("mesh", Client(), download=True)
    wait()
    assert "✅" in app.g_export_md.content, app.g_export_md.content
    assert Client.sent[-1][0].endswith("_mesh-delaunay.glb")
    app.g_export_mesh_fmt.value = "obj"
    app._export("mesh", Client(), download=True)
    wait()
    assert Client.sent[-1][0].endswith("_mesh-delaunay.zip")  # textured: obj + mtl + png in one download
    app.g_export_splat_fmt.value = "splat"
    app._export("splat", None, download=False)
    wait()
    assert os.path.getsize(os.path.join(project, "export", os.path.basename(project) + "_splat.splat")) == 500 * 32
    assert "export/" in app.g_export_md.content


def test_download_payload_zips_several_files(tmp_path):
    import io
    import zipfile
    paths = []
    for name in ("m.obj", "m.mtl", "m.png"):
        paths.append(str(tmp_path / name))
        open(paths[-1], "w").write(name)
    assert gapp._download_payload(paths[:1]) == ("m.obj", b"m.obj")
    name, data = gapp._download_payload(paths)
    assert name == "m.zip" and sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == ["m.mtl", "m.obj", "m.png"]


def test_refresh_does_not_resend_unchanged_layers(app, project, monkeypatch):
    """Refreshes after each dense stage redraw only what changed (the splat can be tens of MB)."""
    calls = []
    real = __import__("modelforge.gui.data", fromlist=["load_splat"]).load_splat
    monkeypatch.setattr("modelforge.gui.data.load_splat", lambda *a, **k: calls.append(1) or real(*a, **k))
    app.g_splat.value = True  # may draw at once through its update callback
    app._drawn = {}
    calls.clear()
    app._refresh_dense()
    app._refresh_dense()
    assert app.g_splat.value and len(calls) == 1  # ticked by default; drawn once
    app.g_splat.value = False
    app._draw_splat()
    app.g_splat.value = True
    app._draw_splat()
    assert len(calls) == 2  # hidden then shown again: redrawn
    p = os.path.join(project, "splat", "splat_enu.ply")
    os.utime(p, (time.time() + 5, time.time() + 5))  # a new splat file
    app._refresh_dense()
    assert len(calls) == 3
    app.g_splat_all.value = False  # a subset (All gaussians is on by default): redrawn with the limit
    assert len(calls) == 4 and "splat:" in app.g_dense_md.content
    from modelforge.gui import data as gdata
    assert gdata.splat_count(p) == len(plyfile_vertices(p))
    app.g_splat_all.value = True


def plyfile_vertices(path):
    import plyfile
    return plyfile.PlyData.read(path)["vertex"].data


def test_quick_bar(app):
    import contextlib

    class Camera:
        position, look_at = (10.0, 0.0, 5.0), (0.0, 0.0, 0.0)

    class Client:
        client_id, camera = 11, Camera()

        def atomic(self):
            return contextlib.nullcontext()

    app._refresh_dense()
    state = lambda: {lbl.split(" ", 1)[1]: lbl[0] for lbl in app._quick_labels}  # noqa: E731
    app.g_splat.value, app.g_mesh.value, app.g_dense_pts.value = True, "off", False
    app._sync_quickbar()
    assert state() == {"Splat": "●", "Mesh": "○", "Dense": "○", "Points": "●", "Cams": "○", "Thumbs": "·",
                       "Resid": "○", "Traj": "●", "Plan": state()["Plan"], "Tiles": "○", "Orbit": "○",
                       "Center": "⌖", "Fly": "○", "Map": "○"}  # the default view: points and trajectories; cameras, thumbs off
    app._quick_click("Splat", None)
    assert not app.g_splat.value and state()["Splat"] == "○"
    app._quick_click("Tiles", None)  # View: Tile boundaries
    assert app.g_tiles_view.value and state()["Tiles"] == "●"
    app._quick_click("Tiles", None)
    assert not app.g_tiles_view.value and state()["Tiles"] == "○"
    app._quick_click("Mesh", None)
    assert app.g_mesh.value == "delaunay" and state()["Mesh"] == "●"
    app._quick_click("Mesh", None)
    assert app.g_mesh.value == "off"
    app._quick_click("Mesh", None)
    assert app.g_mesh.value == "delaunay"  # back to the last mesh
    before = app.g_points.value
    app._quick_click("Points", None)
    assert app.g_points.value is (not before)
    app._quick_click("Thumbs", None)  # nothing while the cameras are off
    assert not app.g_thumbs.value
    app._quick_click("Cameras", None)
    assert app.g_cams.value and state()["Cams"] == "●" and state()["Thumbs"] == "○"
    app._quick_click("Thumbs", None)
    assert app.g_thumbs.value and state()["Thumbs"] == "●"
    before = app.g_corr.value
    app._quick_click("Residuals", None)
    assert app.g_corr.value is (not before) and state()["Resid"] == ("●" if not before else "○")
    before = app.g_traj.value
    app._quick_click("Trajectory", None)
    assert app.g_traj.value is (not before) and state()["Traj"] == ("●" if not before else "○")
    app._quick_click("Dense", None)
    assert app.g_dense_pts.value and state()["Dense"] == "●"
    c = Client()
    app._quick_click("Orbit", c)
    assert state()["Orbit"] == "●"
    app._quick_click("Orbit", c)
    assert state()["Orbit"] == "○"
    c.camera.position, c.camera.look_at = (1e4, 1e4, 1e4), (9e3, 9e3, 9e3)  # wandered off
    app._quick_click("Center", c)
    mid, extent = app._view_frame()
    assert np.allclose(c.camera.look_at, mid) and np.isclose(np.linalg.norm(np.subtract(c.camera.position, mid)),
                                                             np.linalg.norm([0, -0.6, 0.5]) * extent)
    cams = app.snap.cams.centers[app.snap.cams.valid]
    assert np.all(np.abs(np.median(cams, 0) - mid) < extent)  # the cameras are in view
    import time
    t0 = time.time()
    while not app.ingest.prev and time.time() - t0 < 60:  # Plan is clickable once the planner thread has a plan
        time.sleep(0.2)
    app._set_plan_visible(True)  # the Ingest flight plan: quick bar, View and Ingest checkboxes in step
    app._quick_click("Plan", None)
    assert not app.ingest.g_show.value and not app.g_plan.value
    app.g_plan.value = True
    assert app.ingest.g_show.value


def test_fly_the_camera_path(app):
    """Fly: the browsers fly the path themselves (docker/patch_viser_camera.py), from the path and the clock in
    hidden elements; the server keeps the slider and the photo window on that clock, ends the flight at the path's
    end with Loop off, and gives back each browser's field of view and up; the slider moves the view."""
    import contextlib
    import html
    import json
    import re
    import time

    class Camera:
        position, look_at, up_direction, fov = (50.0, 0.0, 30.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0

    class Client:
        client_id, camera = 21, Camera()

        def atomic(self):
            return contextlib.nullcontext()

    def attr(content, name):
        return json.loads(html.unescape(re.search(f'{name}="([^"]*)"', content).group(1)) or "null")

    c = Client()
    path = app._flight_path()
    app.g_fly_speed.value, app.g_fly_loop.value, app.g_fly_fov.value = 20.0, False, True
    app.g_fly_pos.value = 0.0
    app.toggle_fly(c)
    assert app.g_fly.label == "Stop flying" and "● Fly" in app._quick_labels
    assert np.isclose(c.camera.fov, app.fov)  # the real camera's field of view while flying
    sent = attr(app.g_fly_path_html.content, "data-path")  # the path, as FlightPath has it
    assert np.allclose(np.reshape(sent["p"], (-1, 3)), path.centers, atol=1e-4)
    assert np.allclose(sent["t"], path.t, atol=1e-4)
    assert np.allclose(np.reshape(sent["q"], (-1, 4)), path.quats, atol=1e-6)
    clock = attr(app.g_speed_html.content, "data-fly")
    assert clock["path"] == sent["id"] and clock["t0"] == 0.0 and clock["speed"] == 20.0 and not clock["loop"]
    app.g_fly_speed.value = 25.0  # a new speed restarts the clock where the flight is
    clock2 = attr(app.g_speed_html.content, "data-fly")
    assert clock2["e"] > clock["e"] and clock2["speed"] == 25.0 and clock2["t0"] >= 0.0
    t0 = time.time()
    while app._flight is not None and time.time() - t0 < path.duration / 25 + 10:
        time.sleep(0.1)
    assert app._flight is None and app.g_fly.label == "Fly the camera path" and "finished" in app.g_fly_md.content
    assert attr(app.g_speed_html.content, "data-fly") is None  # the browsers stop flying
    assert app.g_fly_pos.value == 100.0 and app._photo_shown == int(path.index[-1])
    assert c.camera.fov == 1.0 and np.allclose(c.camera.up_direction, (0, 0, 1))  # restored
    app.g_fly_pos.value = 0.0
    app._fly_scrub(c)  # the slider, while not flying: the view jumps there
    assert np.allclose(c.camera.position, path.centers[0])
    i = int(path.index[0])  # the Photo window shows the camera the flight is at
    assert app._photo_shown == i and app.project.names[i] in app.g_photo_md.content
    app.toggle_fly(c)  # the slider while flying restarts the clock there
    app.g_fly_pos.value = 50.0
    app._fly_scrub(c)
    assert abs(attr(app.g_speed_html.content, "data-fly")["t0"] - path.duration / 2) < 1e-6
    # a browser's user takes the view over: the flight pauses where that browser stopped, for everyone, until the
    # last browser holding it lets go; then it continues from there
    app._fly_report(21, "pause:1:12.5")
    clock = attr(app.g_speed_html.content, "data-fly")
    assert clock["paused"] and clock["t0"] == 12.5 and app._fly_now() == 12.5
    app._fly_report(22, "pause:1:13.0")  # a second browser: still paused at the first one's point
    app._fly_report(21, "resume:2:12.5")
    time.sleep(0.3)
    assert attr(app.g_speed_html.content, "data-fly")["paused"] and app._fly_now() == 12.5
    assert abs(app.g_fly_pos.value - round(100 * 12.5 / path.duration, 1)) < 0.11  # the slider waits too
    app._fly_report(22, "resume:2:13.0")
    clock = attr(app.g_speed_html.content, "data-fly")
    assert not clock["paused"] and clock["t0"] == 12.5 and app._fly_now() >= 12.5
    app._fly_report(21, "garbage")  # ignored
    app.toggle_fly(c)  # and Stop works mid-flight
    assert app._flight is None and app.g_fly.label == "Fly the camera path"

    # the last browser gone ends the flight (here no browser is connected at all)
    app.g_fly_loop.value = True
    app.toggle_fly(c)
    app._on_disconnect(c)
    assert app._flight is None and app.g_fly.label == "Fly the camera path"
    app._set_orbit(False)
    app.g_fly_loop.value = False


def test_tile_boundaries(app, monkeypatch):
    """View → Tile boundaries: planned cells (no tiled run) drawn as boxes with labels; off removes them."""
    import time as _time
    added = []
    real = app.server.scene.add_line_segments
    monkeypatch.setattr(app.server.scene, "add_line_segments",
                        lambda name, *a, **k: (added.append((name, k.get("colors"))), real(name, *a, **k))[1])
    app.g_splat_tile_layout.value = "rectangles"  # boxes (covisibility, the default, draws outlines)
    app.g_splat_tile_photos.value = 30  # ~80 images -> 3 planned cells
    app.g_tiles_view.value = True
    t0 = _time.time()
    while not any(n == "/tiles/boxes" for n, _ in added) and _time.time() - t0 < 30:
        _time.sleep(0.1)
    boxes = [c for n, c in added if n == "/tiles/boxes"][-1]
    assert len(boxes) == 12 * 3 and "planned" in app.g_dense_md.content
    app.g_tiles_view.value = False
    assert "tiles" not in app.g_dense_md.content
    app.g_splat_tile_photos.value = 500


def test_set_up_tiles(app):
    """Splat tab → Set up tiles: tiles.json written without training; Train tiles offers each tile."""
    import time as _time
    from modelforge import splat_tiles
    app.g_splat_tile_photos.value = 30
    app._setup_tiles_clicked(None)
    t0 = _time.time()
    while app.running and _time.time() - t0 < 60:
        _time.sleep(0.1)
    state = splat_tiles.load(app.path)
    assert state and all(t["status"] == "pending" for t in state["tiles"])
    assert tuple(app.g_splat_train_tiles.options) == ("all",) + tuple(f"tile {k + 1}" for k in range(len(state["tiles"])))
    assert "set up tiles" in app.g_history.content
    app.g_splat_train_tiles.value = "tile 2"
    shutil.rmtree(os.path.join(app.path, splat_tiles.DIR))
    app._sync_tile_choice()
    assert app.g_splat_train_tiles.options == ("all",) and app.g_splat_train_tiles.value == "all"
    app.g_splat_tile_photos.value = 500


def test_pivot_navigation(app):
    """Ctrl + click: the ray from the camera meets the sparse points (not a stray one in front of them);
    zooming close to the pivot carries it forward along the view."""
    pts = app.snap.points[np.all(np.isfinite(app.snap.points), axis=1)]
    target = np.median(pts, axis=0)
    origin = target + np.array([0.0, 0.0, 50.0])
    hit = app.ray_hit(origin, target - origin)
    assert hit is not None and np.linalg.norm(hit[:2] - target[:2]) < 3.0 and hit[2] < origin[2]
    app.snap.points = np.vstack([app.snap.points, origin + np.array([0.0, 0.0, -5.0])])  # one stray point
    try:
        assert np.linalg.norm(app.ray_hit(origin, target - origin) - hit) < 1e-6  # alone it does not count
    finally:
        app.snap.points = app.snap.points[:-1]

    class Camera:
        position, look_at = (0.0, 0.0, 10.0), (0.0, 0.0, 9.9)

    class Client:
        client_id, camera = 31, Camera()
    c = Client()
    near = max(0.3, 0.01 * app._scene_scale())
    app.g_pivot_carry.value = True
    app._carry_pivot(c)
    assert np.allclose(c.camera.look_at, (0.0, 0.0, 10.0 - 8 * near))  # moved ahead along the view
    c.camera.look_at = (0.0, 0.0, 9.9)
    app.g_pivot_carry.value = False
    app._carry_pivot(c)
    assert np.allclose(c.camera.look_at, (0.0, 0.0, 9.9))


def test_move_speed(app, monkeypatch):
    """View → Move speed reaches the page as the hidden element the patched viser bundle reads, and is saved."""
    saved = {}
    monkeypatch.setattr(app, "_save_setting", lambda k, v: saved.update({k: v}))
    app._set_move_speed(12.5)
    assert 'id="mf-move-speed" data-speed="12.5"' in app.g_speed_html.content and saved == {"move_speed": 12.5}


def test_many_cameras_are_one_wireframe_per_layer(app, monkeypatch):
    """Above MAX_FRUSTA cameras the cameras and prior cameras are one line-segment node each (8 segments per
    camera), with no frustum nodes; the selection recolours it, hiding a layer removes its node."""
    from modelforge.gui import app as app_mod
    scene = app.server.scene
    monkeypatch.setattr(app_mod, "MAX_FRUSTA", 10)
    app.g_cams.value = app.g_priors.value = True
    app._draw_cameras()
    assert not app.cam_handles and not app.prior_handles
    lines = scene.get_handle_by_name("/cams_lines")
    assert len(lines.points) == 8 * int(app.snap.cams.valid.sum())
    assert scene.get_handle_by_name("/priors_lines") is not None
    app.select(0)
    assert tuple(scene.get_handle_by_name("/cams_lines").colors[0, 0]) == app_mod.SELECTED_COLOR
    app.g_priors.value = False
    app._draw_cameras()
    assert scene.get_handle_by_name("/priors_lines") is None
    monkeypatch.setattr(app_mod, "MAX_FRUSTA", 2000)  # few cameras again: frustum nodes
    app._draw_cameras()
    assert app.cam_handles and scene.get_handle_by_name("/cams_lines") is None


def test_far_plane_follows_the_scene(app):
    """The far clipping plane grows with the scene (viser's 1000 m cut large flights), never below 1000 m."""
    from modelforge.gui.app import FAR_EXTENTS
    app._set_far(frame=(np.zeros(3), 2000.0))
    assert app.server.initial_camera.far == FAR_EXTENTS * 2000.0
    app._set_far(frame=(np.zeros(3), 10.0))
    assert app.server.initial_camera.far == 1000.0


def test_exclude_photos_panel(app):
    """Sparse tab → Exclude photos: the Inspect selection toggles, a time range adds, the View greys them, the
    status asks for an export (the user asked, 2026-10-28)."""
    from modelforge import exclude
    from modelforge.gui.exclude import COLOR
    ex = app.exclude
    app.select(2)
    assert ex.pick() is True and ex.mask[2]
    assert "excluded" in app.g_inspect_md.content and "Apply" in ex.g_md.content
    assert tuple(app._camera_colors()[2]) == COLOR
    t = app.project.time
    ex.g_range.value = (float(t[5]), float(t[7]))
    assert ex.g_range_md.content.startswith("→ 3 photos")
    ex.add_range()
    assert int(ex.mask.sum()) == 4 and exclude.names(app.path) == {app.project.names[k] for k in (2, 5, 6, 7)}
    ex.edit("Undo last")
    assert int(ex.mask.sum()) == 1
    assert ex.pick(2) is False and not ex.mask.any()
    ex.edit("Include all")
    assert not os.path.exists(os.path.join(app.path, exclude.FILE))


def test_ground_grid_under_the_path(app):
    """View → Ground grid: under the lowest camera, over the path, a round cell size; off removes it."""
    app.g_grid.value = True
    app._draw_grid()
    g = app.server.scene.get_handle_by_name("/grid")
    pts = app._grid_path()
    assert g is not None and g.position[2] < pts[:, 2].min()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    assert g.width >= hi[0] - lo[0] and g.height >= hi[1] - lo[1]
    m = app.grid_cell / 10 ** np.floor(np.log10(app.grid_cell))
    assert round(m, 6) in (1, 2, 5, 10)
    app.g_grid.value = False
    app._draw_grid()
    assert app.server.scene.get_handle_by_name("/grid") is None
    app.g_grid.value = True


def test_pose_graph_lines_between_verified_pairs(app):
    """View → Pose graph: one line per verified pair of database.db at the snapshot's cameras; between passes is a
    subset; Min inliers filters; off removes it."""
    import sqlite3
    db = os.path.join(app.path, "database.db")
    if not os.path.exists(os.path.join(REAL, "database.db")):
        pytest.skip("needs a matched project")
    if not os.path.exists(db):
        os.symlink(os.path.abspath(os.path.join(REAL, "database.db")), db)  # only read
    con = sqlite3.connect(db)
    want = con.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
    con.close()
    app.load_snapshot("refined")
    i, j, n = app.project.pose_graph()
    assert len(i) == want and np.all(i < j) and n.min() > 0
    app.g_graph.value, app.g_graph_edges.value, app.g_graph_min.value = True, "all", 15
    app._draw_graph()
    v = app.snap.cams.valid
    h = app.server.scene.get_handle_by_name("/pose_graph")
    assert h is not None and len(h.points) == int((v[i] & v[j]).sum())
    assert np.allclose(h.points[0, 0], app.snap.cams.centers[i[v[i] & v[j]][0]], atol=1e-3)
    app.g_graph_min.value = int(np.median(n))
    app._draw_graph()
    assert len(app.server.scene.get_handle_by_name("/pose_graph").points) < len(h.points) or n.min() == n.max()
    app.g_graph_edges.value = "between passes"
    app.g_graph_min.value = 15
    app._draw_graph()
    g = app.server.scene.get_handle_by_name("/pose_graph")
    assert g is None or len(g.points) <= len(h.points)
    assert "verified pairs" in app.g_graph_md.content
    app.g_graph.value = False
    app._draw_graph()
    assert app.server.scene.get_handle_by_name("/pose_graph") is None


def test_live_snapshot_shows_in_the_dropdown(app, monkeypatch):
    """A snapshot a live refine loaded joins the Snapshot list and shows as picked, without a second load (the
    dropdown had kept "priors" only while the View showed the refine's model)."""
    app.g_snapshot.options = ("priors",)
    app.load_snapshot("priors")  # as after a project opened before SfM
    with app.lock:
        app.snap = app.project.load_snapshot("refined")
        loads = []
        monkeypatch.setattr(app, "load_snapshot", lambda name: loads.append(name))
        app._show_snapshot_name("refined")
    assert "refined" in app.g_snapshot.options and app.g_snapshot.value == "refined" and not loads
    monkeypatch.undo()
    app.g_snapshot.value = "priors"  # a real pick still loads
    assert app.snap.name == "priors"
    app.refresh_snapshot_list(select="main")

"""GUI dense/splat layers and dense stage runs, on a temp project that links the real
priors/images/sparse model and carries synthetic dense outputs and a synthetic splat."""

import os
import shutil
import socket
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
    assert state() == {"Splat": "●", "Mesh": "○", "Dense": "○", "Points": "○", "Cams": "○", "Thumbs": "·",
                       "Resid": state()["Resid"], "Traj": state()["Traj"], "Plan": state()["Plan"], "Tiles": "○", "Orbit": "○",
                       "Center": "⌖", "Fly": "○"}  # points, cameras, thumbs off
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
    """Fly: the view runs along the cameras to the end, looking through them; field of view and up come
    back afterwards; the position slider moves the view."""
    import contextlib
    import time

    class Camera:
        position, look_at, up_direction, fov = (50.0, 0.0, 30.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0

    class Client:
        client_id, camera = 21, Camera()

        def atomic(self):
            return contextlib.nullcontext()

    c = Client()
    path = app._flight_path()
    app.g_fly_speed.value, app.g_fly_loop.value, app.g_fly_fov.value = 20.0, False, True
    app.g_fly_pos.value = 0.0
    app.toggle_fly(c)
    assert app.g_fly.label == "Stop flying" and "● Fly" in app._quick_labels
    time.sleep(0.3)
    assert np.isclose(c.camera.fov, app.fov)  # the real camera's field of view while flying
    t0 = time.time()
    while app._flights and time.time() - t0 < path.duration / 20 + 10:
        time.sleep(0.1)
    assert not app._flights and app.g_fly.label == "Fly the camera path" and "finished" in app.g_fly_md.content
    end, R, _ = path.pose(path.duration)
    assert np.allclose(c.camera.position, end)  # at the last camera, looking along it
    look = np.asarray(c.camera.look_at) - end
    assert np.allclose(look / np.linalg.norm(look), R[:, 2], atol=1e-6)
    assert c.camera.fov == 1.0 and np.allclose(c.camera.up_direction, (0, 0, 1))  # restored
    app.g_fly_pos.value = 0.0
    app._fly_scrub(c)  # the slider, while not flying: the view jumps there
    assert np.allclose(c.camera.position, path.centers[0])
    app.toggle_fly(c)  # and Stop works mid-flight
    time.sleep(0.2)
    app.toggle_fly(c)
    time.sleep(0.2)
    assert not app._flights


def test_tile_boundaries(app, monkeypatch):
    """View → Tile boundaries: planned cells (no tiled run) drawn as boxes with labels; off removes them."""
    import time as _time
    added = []
    real = app.server.scene.add_line_segments
    monkeypatch.setattr(app.server.scene, "add_line_segments",
                        lambda name, *a, **k: (added.append((name, k.get("colors"))), real(name, *a, **k))[1])
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

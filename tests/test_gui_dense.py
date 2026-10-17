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
    for name in ("poses.json", "images", "sparse"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))  # stages write to it
    make_workspace(p)
    dense.fusion(p)
    dense.mesh(p, "delaunay")
    dense.preview(p, "delaunay", max_faces=2000)
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
    assert "faces" in app.g_dense_md.content
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


def test_tab_defaults(app):
    assert dense.QUALITY[gapp.DENSE_QUALITY[app.g_dense_quality.value]] == dense.QUALITY["low"]
    assert app.g_splat_disk.value is True and app.g_splat_down.value == 1


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
    assert Client.sent[-1][0].endswith("_mesh-delaunay.obj")  # untextured: one file, sent as is
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

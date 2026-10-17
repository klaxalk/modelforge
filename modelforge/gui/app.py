"""modelforge web GUI (viser): ingest videos, run stages live, inspect snapshots and products.

Usage (in the dev container; port 8080 is published by compose.yaml):
    python -m modelforge.gui.app [data/site] [--root data] [--port 8080]
Projects are folders under --root (Projects tab: open / create / clean). A project folder
may be new or empty: the Ingest tab builds it from one or more videos.
"""

import argparse
import os
import threading
import time
import traceback

import numpy as np
import plotly.graph_objects as go
import viser
from plotly.subplots import make_subplots

from modelforge import dense, pipeline, projects, splat
from modelforge.export import products
from modelforge.gui import data as gdata
from modelforge.gui.data import Project
from modelforge.gui.videos import VideosPanel, upload_progress
from modelforge.gui.ingest import IngestPanel, source_color
from modelforge.gui.projects import ProjectsPanel

DENSE_QUALITY = {"low (640 px, fast)": "low", "medium (1200 px)": "medium", "high (full res)": "high",
                 "max (COLMAP defaults)": "max"}  # label -> dense.QUALITY preset
DENSE_QUALITY_DEFAULT = next(k for k, v in DENSE_QUALITY.items() if v == dense.DEFAULT_QUALITY)

STAGE_HINTS = {
    "features": "SIFT keypoints for every image into features.db (upright, up to 8192 per image, one shared "
                "camera). A few minutes on CPU.",
    "pairs": "Choose image pairs to match from the priors: time neighbours within each source plus views whose "
             "look-at points overlap (catches revisits and links sources). Seconds.",
    "match": "Match the selected pairs and verify them geometrically into database.db. The slowest SfM stage "
             "on CPU (~12 min for 2400 pairs).",
    "refine": "Start every camera at its prior, then 3 rounds of triangulate + bundle adjustment with "
              "self-calibrating 6-DoF priors. Cameras move live in the View tab. ~1.5-3 min.",
    "export": "Write transforms.json + sparse_pc.ply (nerfstudio dataset, ENU world) from sparse/refined. Seconds.",
    "undistort": "Undistorted images + PINHOLE model into dense/ (COLMAP MVS workspace). Seconds.",
    "stereo": "PatchMatch depth and normal maps for every image. CUDA only; the long one (~10 GB at full res).",
    "fusion": "Fuse depth maps into dense/fused.ply (+ .vis for meshing).",
    "mesh": "Mesh the fused points (method above) and write a simplified preview for the View tab.",
    "texture": "Texture the mesh from the images: dense/textured-<method>/mesh.ply + texture.png.",
    "splat": "nerfstudio splatfacto: train, export, convert to ENU (splat/splat_enu.ply).",
    "all": "Full pipeline: sparse → splat → dense.",
}
NEXT_STEP = {
    "all": "View tab: refined cameras, mesh and splat; Export tab for the files",
    "sfm": "View tab → snapshot **refined**; next: Dense tab (GPU) or Splat tab",
    "features": "next: **pairs**", "pairs": "next: **match**", "match": "next: **refine**",
    "refine": "View tab shows **refined**; next: **export**",
    "export": "transforms.json written; next: Dense tab (GPU) or Splat tab",
    "dense": "View tab → Dense / splat; Export tab for the mesh",
    "undistort": "next: **stereo**", "stereo": "next: **fusion**", "fusion": "next: **mesh**",
    "mesh": "View tab → Mesh; next: **texture**", "texture": "textured mesh written; Export tab",
    "splat": "View tab → Dense / splat → **Splat (ENU)**; Export tab",
}
LOG_LINES = 40

COLOR_BY = ("position deviation", "rotation deviation", "reprojection error", "time", "source", "uniform")
SOURCE_SWATCH = ("🟦", "🟨", "🟩", "🟥", "🟢", "🟪", "🩷", "🔵")  # approximates gui.ingest.PALETTE
UNITS = {"position deviation": "m", "rotation deviation": "deg", "reprojection error": "px", "time": "s"}
PRIOR_COLOR = (150, 150, 150)
SELECTED_COLOR = (255, 255, 255)
_ANCHORS = np.array([[48, 18, 59], [70, 134, 251], [27, 229, 181], [164, 252, 60],
                     [251, 185, 56], [228, 70, 11], [122, 4, 3]], float)  # turbo-like


def _fmt_s(sec):
    sec = int(round(sec))
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def _download_payload(paths):
    """(filename, bytes) for the browser: the file itself, or a .zip when the export has several files."""
    if len(paths) == 1:
        with open(paths[0], "rb") as f:
            return os.path.basename(paths[0]), f.read()
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.basename(p))
    return os.path.splitext(os.path.basename(paths[0]))[0] + ".zip", buf.getvalue()


def colormap(values, vmax=None):
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    if vmax is None:
        vmax = float(np.percentile(v[finite], 95)) if finite.any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    x = np.clip(np.where(finite, v, 0) / vmax, 0, 1) * (len(_ANCHORS) - 1)
    lo = np.floor(x).astype(int).clip(0, len(_ANCHORS) - 2)
    f = (x - lo)[:, None]
    rgb = (_ANCHORS[lo] * (1 - f) + _ANCHORS[lo + 1] * f).astype(np.uint8)
    rgb[~finite] = (90, 90, 90)
    return rgb, vmax


def rot_dev_deg(Ra, Rb):
    c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class App:
    def __init__(self, project_path, port=8080, root=None):
        self.path = project_path
        self.root = root or os.path.dirname(os.path.abspath(project_path))
        self.project = Project(project_path) if os.path.exists(os.path.join(project_path, "poses.json")) else None
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_width="large")
        self.lock = threading.RLock()
        self.cam_handles, self.prior_handles = {}, {}
        self.selected = None
        self.snap = None
        self.running = False
        self.fov, self.aspect = np.radians(53.0), 16 / 9
        self._last_live = 0.0
        self._run_id = 0
        self._job = {"name": "", "t0": 0.0, "label": "", "stage_t0": 0.0, "detail": ""}
        self._splat_job = None
        self._build_gui()
        if self.project:
            names = self.project.snapshot_names()
            self.load_snapshot("refined" if "refined" in names else names[-1])
        self._refresh_dense()
        self.server.on_client_connect(self._on_connect)
        threading.Thread(target=self._watch_uploads, daemon=True).start()

    # ---------------------------------------------------------------- GUI
    def _build_gui(self):
        gui = self.server.gui
        self.g_header = gui.add_markdown(self._header())
        # run state above the tabs, so it stays visible on every tab
        self.g_state = gui.add_markdown("⚪ **idle**")
        self.g_progress = gui.add_progress_bar(value=0.0, visible=False)
        self.g_buttons = {}
        b = gui.add_button("Run full pipeline", icon=viser.Icon.PLAYER_TRACK_NEXT_FILLED, color="green",
                           hint="Sparse → splat → dense in one go, with the settings of the tabs below (and the "
                                "Ingest plan: images are extracted first if the project has none yet). Stages that "
                                "need a GPU are skipped without one.")
        b.on_click(lambda _: self.run_stage("all"))
        self.g_buttons["all"] = b
        # Two panels: the right (main) one runs the methods, the left one holds everything else.
        self.left = gui.add_panel()
        tabs = gui.add_tab_group()
        with self.left.add_tab("Projects", viser.Icon.FOLDERS):
            self.projects = ProjectsPanel(self, gui)
        with self.left.add_tab("Videos", viser.Icon.MOVIE):
            self.videos = VideosPanel(self, gui)
        with self.left.add_tab("Ingest", viser.Icon.VIDEO):
            self.ingest = IngestPanel(self, gui)

        with self.left.add_tab("View", viser.Icon.EYE):
            names = tuple(self.project.snapshot_names()) if self.project else ("priors",)
            self.g_snapshot = gui.add_dropdown(
                "Snapshot", options=names, hint="priors = telemetry poses; round_N_tri = after triangulation of "
                "round N; round_N = after its bundle adjustment; refined = final model.")
            self.g_snapshot.on_update(lambda _: self.load_snapshot(self.g_snapshot.value))
            self.g_stats = gui.add_markdown("")
            with gui.add_folder("Layers"):
                self.g_points = gui.add_checkbox("Points", initial_value=True,
                                                 hint="Sparse 3D points of the snapshot.")
                self.g_point_size = gui.add_slider("Point size", min=0.01, max=0.5, step=0.01, initial_value=0.08)
                self.g_cams = gui.add_checkbox("Cameras", initial_value=True,
                                               hint="Camera frustums of the snapshot; click one to inspect it.")
                self.g_thumbs = gui.add_checkbox("Thumbnails", initial_value=True,
                                                 hint="Image thumbnails in the frustums (slower with many cameras).")
                self.g_priors = gui.add_checkbox("Prior cameras", initial_value=False,
                                                 hint="Grey frustums at the telemetry poses.")
                self.g_corr = gui.add_checkbox("Prior → refined", initial_value=True,
                                               hint="Red segments from each prior position to the estimated one.")
                self.g_traj = gui.add_checkbox("Trajectories", initial_value=True,
                                               hint="Grey: prior path; blue: estimated path.")
                self.g_scale = gui.add_slider("Camera size", min=0.2, max=5.0, step=0.1, initial_value=1.0)
            with gui.add_folder("Dense / splat"):
                self.g_dense_pts = gui.add_checkbox("Dense points", initial_value=False, disabled=True,
                                                    hint="dense/fused.ply (subsampled to 1.5M points). Enabled once "
                                                         "fusion has run.")
                self.g_mesh = gui.add_dropdown("Mesh", options=("off",), initial_value="off", disabled=True,
                                               hint="Simplified preview of the dense mesh (≤300k faces).")
                self.g_splat = gui.add_checkbox("Splat (ENU)", initial_value=False, disabled=True,
                                                hint="splat/splat_enu.ply from scripts/splat, in the same frame "
                                                     "as the cameras and mesh.")
                self.g_dense_md = gui.add_markdown("")
                self.g_dense_pts.on_update(lambda _: self._draw_dense_points())
                self.g_mesh.on_update(lambda _: self._draw_mesh())
                self.g_splat.on_update(lambda _: self._draw_splat())
            self.g_color = gui.add_dropdown("Color cameras by", options=COLOR_BY, initial_value="position deviation",
                                           hint="Deviation from the prior, mean reprojection error, capture "
                                                "time or source (video piece).")
            self.g_legend = gui.add_markdown("")
            for h in (self.g_points, self.g_point_size):
                h.on_update(lambda _: self._draw_points())
            for h in (self.g_cams, self.g_thumbs, self.g_priors, self.g_scale):
                h.on_update(lambda _: self._draw_cameras())
            self.g_color.on_update(lambda _: self._apply_colors())
            for h in (self.g_corr, self.g_traj):
                h.on_update(lambda _: self._draw_lines())

        with self.left.add_tab("Inspect", viser.Icon.PHOTO):
            self.g_inspect_md = gui.add_markdown("Click a camera to inspect it.")
            self.g_gain = gui.add_slider("Residual ×", min=1.0, max=50.0, step=1.0, initial_value=10.0,
                                         hint="Magnification of the red reprojection-residual lines.")
            self.g_gain.on_update(lambda _: self.select(self.selected) if self.selected is not None else None)
            self.g_image = gui.add_image(np.zeros((54, 96, 3), np.uint8), label="observations")

        with self.left.add_tab("Plots", viser.Icon.CHART_LINE):
            self.g_plot = gui.add_plotly(go.Figure(), aspect=0.9)

        with tabs.add_tab("Sparse", viser.Icon.CAMERA):
            gui.add_markdown("Pose refinement on the CPU: features → pairs → match → refine → export "
                             "(nerfstudio dataset).")
            b = gui.add_button("Run SfM (all)", icon=viser.Icon.PLAYER_TRACK_NEXT,
                               hint="Runs features → pairs → match → refine → export in sequence. "
                                    "About 15-20 min on CPU for 300 images; matching dominates.")
            b.on_click(lambda _: self.run_stage("sfm"))
            self.g_buttons["sfm"] = b
            for stage in pipeline.SFM_STAGES:
                b = gui.add_button(f"Run {stage}" + (" (live)" if stage == "refine" else ""),
                                   hint=STAGE_HINTS[stage])
                b.on_click(lambda _, s=stage: self.run_stage(s))
                self.g_buttons[stage] = b

        with tabs.add_tab("Dense", viser.Icon.GRID_DOTS):
            gui.add_markdown("COLMAP MVS. PatchMatch stereo: " + (
                "CUDA available" if dense.has_cuda() else
                "**needs CUDA** - run the GUI in the CUDA container (`MODELFORGE_GPU=1 scripts/dev`)"))
            self.g_dense_quality = gui.add_dropdown(
                "Quality", options=tuple(DENSE_QUALITY), initial_value=DENSE_QUALITY_DEFAULT,
                hint="Speed vs detail (image size for undistort, PatchMatch effort for stereo; set when those "
                     "stages run). low: 640 px, fast PatchMatch (default, minutes). medium: 1200 px. high: full "
                     "resolution. max: COLMAP defaults at full resolution (hours for 300 images).")
            self.g_mesh_method = gui.add_dropdown(
                "Mesh method", options=dense.MESH_METHODS + ("both",), initial_value="delaunay",
                hint="delaunay: visibility-based, keeps open surfaces (default). poisson: watertight, "
                     "closes holes and may balloon over open areas.")
            b = gui.add_button("Run dense (all)", icon=viser.Icon.PLAYER_TRACK_NEXT, disabled=not dense.has_cuda(),
                               hint="undistort → stereo → fusion → mesh → texture on the refined poses. "
                                    "Needs CUDA. Time depends on Quality.")
            b.on_click(lambda _: self.run_stage("dense"))
            self.g_buttons["dense"] = b
            for stage in pipeline.DENSE_STAGES:
                b = gui.add_button(f"Run {stage}", disabled=(stage == "stereo" and not dense.has_cuda()),
                                   hint=STAGE_HINTS[stage])
                b.on_click(lambda _, s=stage: self.run_stage(s))
                self.g_buttons[stage] = b

        with tabs.add_tab("Splat", viser.Icon.SPARKLES):
            why = splat.available()
            gui.add_markdown("nerfstudio splatfacto: " + ("ready (GPU + nerfstudio found) · viewer on "
                                                          f"http://localhost:{splat.VIEWER_PORT} while training"
                                                          if why is None else f"**unavailable**: {why}"))
            self.g_splat_method = gui.add_dropdown(
                "Method", options=splat.METHODS, initial_value="splatfacto",
                hint="splatfacto: default. splatfacto-big: more Gaussians and detail, more GPU memory and time.")
            self.g_splat_iters = gui.add_number(
                "Iterations", initial_value=splat.DEFAULT_ITERATIONS, min=100, step=1000,
                hint="Training steps. 30000 is splatfacto's default (~20-40 min on a recent GPU); "
                     "use a few thousand for a quick check.")
            self.g_splat_disk = gui.add_checkbox(
                "Cache images on disk", initial_value=True,
                hint="On: images stay on disk and are streamed by worker processes, so large datasets fit in "
                     "memory. Off: all images are preloaded into GPU memory (nerfstudio's default; it falls back "
                     "to CPU memory above 500 images), a little faster for small projects.")
            self.g_splat_down = gui.add_number(
                "Downsample factor", initial_value=1, min=1, max=16, step=1,
                hint="Train on images shrunk by this factor (2 = half width and height): faster and less memory, "
                     "less detail. The shrunk copies go to images_<factor>/ (an intermediate, removed by Clean).")
            self.g_splat_live = gui.add_checkbox(
                "Live preview", initial_value=True,
                hint="Show the splat in the View tab while it trains, refreshed at every checkpoint (base colours, "
                     "at most 500k Gaussians). The finished splat replaces it at the end.")
            self.g_splat_every = gui.add_number(
                "Preview every (steps)", initial_value=1000, min=100, step=100,
                hint="Checkpoint interval while previewing (nerfstudio's default is 2000). Each checkpoint is "
                     "written to disk (~0.75 GB per million Gaussians), so very small values slow training down.")
            self.g_splat_status = gui.add_markdown("")
            b = gui.add_button("Train + export splat", icon=viser.Icon.SPARKLES, disabled=why is not None,
                               hint="ns-train on transforms.json + sparse_pc.ply, then ns-export and conversion "
                                    "to splat/splat_enu.ply in the project's ENU frame (aligned with the "
                                    "cameras and the mesh).")
            b.on_click(lambda _: self.run_stage("splat"))
            self.g_buttons["splat"] = b
            self.g_splat_stop = gui.add_button("Stop splat training", icon=viser.Icon.PLAYER_STOP, disabled=True,
                                               hint="Terminate ns-train / ns-export. Nothing is converted; "
                                                    "checkpoints written so far stay in nerfstudio/.")
            self.g_splat_stop.on_click(lambda _: self._splat_job.stop() if self._splat_job else None)
            gui.add_button("Reload dense / splat", hint="Re-scan the project for fused points, meshes and "
                           "splat/splat_enu.ply, e.g. after scripts/splat finished.").on_click(
                lambda _: self._refresh_dense())

        with self.left.add_tab("Export", viser.Icon.DOWNLOAD):
            self.g_export_md = gui.add_markdown("")
            with gui.add_folder("Mesh"):
                self.g_export_mesh = gui.add_dropdown(
                    "Mesh", options=("none",), initial_value="none", disabled=True,
                    hint="Which dense mesh to export (enabled once the mesh stage has run).")
                self.g_export_mesh_fmt = gui.add_dropdown(
                    "Format", options=products.MESH_FORMATS, initial_value="glb",
                    hint="glb: one textured glTF file (Blender, web viewers, Cesium). obj: .obj + .mtl + texture. "
                         "Both Y-up as those tools expect. ply: COLMAP's files as they are, in ENU (z up).")
                self.g_export_textured = gui.add_checkbox(
                    "Textured", initial_value=True,
                    hint="Use the textured mesh (texture stage) if it exists; off: the vertex-coloured mesh.")
                self.g_export_buttons = {}
                b = gui.add_button("Export mesh", icon=viser.Icon.FILE_EXPORT, disabled=True,
                                   hint="Write the mesh into PROJECT/export/ (+ export/georef.json).")
                b.on_click(lambda e: self._export("mesh", e.client, download=False))
                self.g_export_buttons["mesh"] = b
                b = gui.add_button("Download mesh", icon=viser.Icon.DOWNLOAD, disabled=True,
                                   hint="Export, then send it to this browser (OBJ and textured PLY as a .zip).")
                b.on_click(lambda e: self._export("mesh", e.client, download=True))
                self.g_export_buttons["mesh_dl"] = b
            with gui.add_folder("Splat"):
                self.g_export_splat_fmt = gui.add_dropdown(
                    "Format", options=products.SPLAT_FORMATS, initial_value="ply",
                    hint="ply: standard 3DGS PLY with all SH bands (SuperSplat, Postshot, gsplat). splat: compact "
                         "32 bytes per Gaussian, colour only (antimatter15-style web viewers). Both in ENU (z up).")
                b = gui.add_button("Export splat", icon=viser.Icon.FILE_EXPORT, disabled=True,
                                   hint="Write the splat into PROJECT/export/ (+ export/georef.json).")
                b.on_click(lambda e: self._export("splat", e.client, download=False))
                self.g_export_buttons["splat"] = b
                b = gui.add_button("Download splat", icon=viser.Icon.DOWNLOAD, disabled=True,
                                   hint="Export, then send it to this browser.")
                b.on_click(lambda e: self._export("splat", e.client, download=True))
                self.g_export_buttons["splat_dl"] = b

        with gui.add_folder("History"):
            self.g_history = gui.add_markdown("_nothing run in this session_")
        with gui.add_folder("Log", expand_by_default=False):
            self.g_log = gui.add_markdown("")
        self._place_panels()

    def _place_panels(self):
        self.left.dock_left()
        self.left.set_width(420)
        self.server.gui.main_panel.dock_right()

    def _header(self):
        if self.project is None:
            return f"### modelforge\n`{self.path}` · no images yet"
        n_src = len(set(self.project.source.tolist()))
        return (f"### modelforge\n`{self.path}` · {len(self.project.names)} images"
                + (f" from {n_src} sources" if n_src > 1 else ""))

    def load_project(self):
        """(Re)load the open project after extraction: new images, no snapshots yet."""
        with self.lock:
            self._clear_scene()
            self.project = Project(self.path)
            self._show_project()
        self.projects.refresh()

    def open_project(self, path):
        """Switch the whole GUI to another (new or existing) project folder."""
        if self.running or self.ingest.running:
            raise RuntimeError("a job is running: wait for it to finish before switching projects")
        with self.lock:
            self.path = path
            self._clear_scene()
            self.project = Project(path) if os.path.exists(os.path.join(path, "poses.json")) else None
            self._show_project()
            self._history_lines, self._log_lines = [], []
            self.g_history.content, self.g_log.content = "_nothing run in this session_", ""
            self.g_state.content = "⚪ **idle**"
            self.g_inspect_md.content = "Click a camera to inspect it."
        self.ingest.set_project(path)
        self.projects.refresh()
        for client in self.server.get_clients().values():
            self._on_connect(client)

    def _watch_uploads(self):
        """Progress of browser uploads still arriving (viser delivers them only when complete)."""
        while True:
            p = upload_progress(self.server)
            if p:  # once complete, the upload handlers write the result
                self.videos.g_msg.content = self.ingest.g_upload_md.content = "⏳ receiving " + " · ".join(p)
            time.sleep(0.5)

    def _clear_scene(self):
        for h in list(self.cam_handles.values()) + list(self.prior_handles.values()):
            h.remove()
        self.cam_handles, self.prior_handles, self.selected, self.snap = {}, {}, None, None
        for name in ("/points", "/corrections", "/traj", "/dense"):
            self.server.scene.remove_by_name(name)

    def _show_project(self):
        self.g_header.content = self._header()
        if self.project is None:
            self.g_snapshot.options = ("priors",)
            self.g_stats.content = self.g_legend.content = ""
            self.g_plot.figure = go.Figure()
        else:
            names = tuple(self.project.snapshot_names())
            self.g_snapshot.options = names
            self.load_snapshot("refined" if "refined" in names else names[-1])
        self._refresh_dense()

    def _on_connect(self, client):
        if self.project is None:
            prev = self.ingest.prev
            if not prev:
                return
            centers = np.concatenate([x["track"] for x in prev["sources"]])
        else:
            c = self.snap.cams if self.snap else self.project.prior_cams
            centers = c.centers[c.valid]
        mid = centers.mean(0)
        extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(mid + np.array([0.0, -0.6, 0.5]) * extent)
        client.camera.look_at = tuple(mid)

    # ------------------------------------------------------------- loading
    def load_snapshot(self, name):
        if self.project is None:
            return
        with self.lock:
            self.snap = self.project.load_snapshot(name)
            if self.snap.intrinsics:
                intr = self.snap.intrinsics
                fy = intr["params"][1] if intr["model"] in ("OPENCV", "PINHOLE") else intr["params"][0]
                self.fov = 2 * np.arctan(intr["height"] / 2 / fy)
                self.aspect = intr["width"] / intr["height"]
            self._draw_points()
            self._draw_cameras()
            self._draw_lines()
            self._update_stats()
            self._update_plots()
            if self.selected is not None:
                self.select(self.selected)

    def refresh_snapshot_list(self, select=None):
        names = tuple(self.project.snapshot_names())
        self.g_snapshot.options = names
        if select in names:
            self.g_snapshot.value = select  # triggers load_snapshot

    # ------------------------------------------------------------- metrics
    def deviations(self, cams=None):
        cams = cams or self.snap.cams
        pc = self.project.prior_cams
        pos = np.linalg.norm(cams.centers - pc.centers, axis=1)
        rot = rot_dev_deg(cams.R_wc, pc.R_wc)
        return pos, rot

    def color_values(self):
        mode = self.g_color.value
        pos, rot = self.deviations()
        if mode == "position deviation":
            return pos
        if mode == "rotation deviation":
            return rot
        if mode == "reprojection error":
            return self.snap.reproj_error if self.snap.reproj_error is not None else np.full(len(pos), np.nan)
        if mode == "time":
            return self.project.time
        if mode == "source":
            return self.project.source.astype(float)
        return None

    # ------------------------------------------------------------- drawing
    def _draw_points(self):
        s = self.snap
        pts, cols = s.points, s.colors
        if len(pts):
            c = self.project.prior_cams.centers
            mid, extent = c.mean(0), float(np.linalg.norm(c.max(0) - c.min(0)))
            keep = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts - mid, axis=1) < 2.0 * extent + 200.0)
            pts, cols = pts[keep], cols[keep]
        if not len(pts):
            pts, cols = np.zeros((1, 3)), np.zeros((1, 3), np.uint8)
        self.server.scene.add_point_cloud("/points", points=pts.astype(np.float32), colors=cols,
                                          point_size=self.g_point_size.value, precision="float32",
                                          visible=bool(self.g_points.value and len(s.points)))

    def _draw_cameras(self):
        with self.lock:
            s, n = self.snap, len(self.project.names)
            colors = self._camera_colors()
            scale = 0.6 * self.g_scale.value
            for i in range(n):
                show = self.g_cams.value and s.cams.valid[i]
                color = SELECTED_COLOR if i == self.selected else tuple(int(c) for c in colors[i])
                h = self.server.scene.add_camera_frustum(
                    f"/cams/{i}", fov=self.fov, aspect=self.aspect, scale=scale, color=color,
                    image=self.project.thumbnail(i) if (show and self.g_thumbs.value) else None,
                    wxyz=s.cams.wxyz(i) if s.cams.valid[i] else (1, 0, 0, 0),
                    position=s.cams.centers[i] if s.cams.valid[i] else (0, 0, 0), visible=bool(show))
                h.on_click(lambda _, i=i: self.select(i))
                self.cam_handles[i] = h
                pc = self.project.prior_cams
                ph = self.server.scene.add_camera_frustum(
                    f"/priors/{i}", fov=self.fov, aspect=self.aspect, scale=scale * 0.8, color=PRIOR_COLOR,
                    wxyz=pc.wxyz(i), position=pc.centers[i], visible=bool(self.g_priors.value))
                self.prior_handles[i] = ph

    def _draw_lines(self, cams=None):
        cams = cams or self.snap.cams
        pc = self.project.prior_cams
        v = cams.valid
        if self.g_corr.value and self.snap.name != "priors" and v.any():
            seg = np.stack([pc.centers[v], cams.centers[v]], axis=1).astype(np.float32)
            self.server.scene.add_line_segments("/corrections", seg, colors=(255, 80, 80), thickness=2.0,
                                                thickness_units="screen")
        else:
            self.server.scene.add_line_segments("/corrections", np.zeros((0, 2, 3), np.float32), colors=(0, 0, 0))
        if self.g_traj.value:
            p = pc.centers
            self.server.scene.add_line_segments("/traj/prior", np.stack([p[:-1], p[1:]], 1).astype(np.float32),
                                                colors=PRIOR_COLOR, thickness=1.5, thickness_units="screen")
            c = cams.centers
            ok = v[:-1] & v[1:]
            self.server.scene.add_line_segments("/traj/estimate",
                                                np.stack([c[:-1][ok], c[1:][ok]], 1).astype(np.float32),
                                                colors=(70, 134, 251), thickness=2.0, thickness_units="screen")
        else:
            for name in ("/traj/prior", "/traj/estimate"):
                self.server.scene.add_line_segments(name, np.zeros((0, 2, 3), np.float32), colors=(0, 0, 0))

    def _update_stats(self):
        s = self.snap
        pos, rot = self.deviations()
        v = s.cams.valid
        lines = [f"**{s.name}** · {int(v.sum())}/{len(v)} cameras · {len(s.points):,} points"]
        if s.name != "priors" and v.any():
            lines.append(f"prior deviation: pos median {np.median(pos[v]):.2f} m (p90 {np.percentile(pos[v], 90):.2f}),"
                         f" rot median {np.median(rot[v]):.2f}° (p90 {np.percentile(rot[v], 90):.2f})")
        if len(s.point_error):
            lines.append(f"point error: median {np.median(s.point_error):.2f} px, "
                         f"p90 {np.percentile(s.point_error, 90):.2f} px")
        if s.intrinsics:
            p = s.intrinsics["params"]
            lines.append(f"camera {s.intrinsics['model']}: f = {p[0]:.1f}, {p[1]:.1f} px")
        pm = self._prior_model(s.name)
        if pm:
            a, r = pm["abs_sigmas"], pm["rel_sigmas"]
            lines.append(f"prior σ abs: {a['pos_horizontal_m']:.2f} m h, {a['pos_vertical_m']:.2f} m v, "
                         f"{a['roll_pitch_deg']:.2f}° r/p, {a['yaw_deg']:.2f}° yaw · rel: {r['rot_deg']:.3f}°, "
                         f"{r['pos_m']:.3f} m + {r['pos_per_m'] * 100:.2f} % of baseline")
            lines.append(f"boresight {pm['boresight_norm_deg']:.2f}° · time offset {pm['dt_ms']:.1f} ms · "
                         f"rejected priors: {len(pm['abs_rejected'])} abs, {pm['rel_rejected']} rel")
            for k, o in pm.get("source_offsets_m", {}).items():
                lines.append(f"source {int(k) + 1} offset: {np.linalg.norm(o):.2f} m "
                             f"(E {o[0]:+.2f}, N {o[1]:+.2f}, U {o[2]:+.2f})")
        self.g_stats.content = "  \n".join(lines)

    def _prior_model(self, snapshot):
        """Calibrated prior model after the given snapshot, from report.json (None if unknown)."""
        refine_rep = self.project.report().get("refine", {})
        if snapshot == "refined":
            return refine_rep.get("prior_model")
        rnd = refine_rep.get(snapshot, {}) if snapshot.startswith("round_") and not snapshot.endswith("_tri") else {}
        pm = rnd.get("prior_model", {})
        return None if pm.get("skipped") else pm.get("model")

    def _update_plots(self):
        s = self.snap
        pos, rot = self.deviations()
        v = s.cams.valid
        t = self.project.time
        col = ["rgb(%d,%d,%d)" % source_color(k) for k in self.project.source[v]]
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09,
                            subplot_titles=("position deviation from prior [m]",
                                            "rotation deviation from prior [deg]", "point reprojection error [px]"))
        fig.add_trace(go.Scatter(x=t[v], y=pos[v], mode="markers", marker=dict(size=4, color=col), name="pos"), 1, 1)
        fig.add_trace(go.Scatter(x=t[v], y=rot[v], mode="markers", marker=dict(size=4, color=col), name="rot"), 2, 1)
        if len(s.point_error):
            fig.add_trace(go.Histogram(x=np.clip(s.point_error, 0, 5), nbinsx=50, name="err"), 3, 1)
        fig.update_xaxes(title_text="time [s] (colour = source)", row=2, col=1)
        fig.update_layout(showlegend=False, margin=dict(l=40, r=10, t=30, b=30), height=640)
        self.g_plot.figure = fig

    # ------------------------------------------------------------- inspect
    def select(self, i):
        prev, self.selected = self.selected, i
        for j in (prev, i):
            if j is not None and j in self.cam_handles:
                self._recolor(j)
        name = self.project.names[i]
        img, st = self.project.inspect_image(self.snap, i, residual_gain=self.g_gain.value)
        self.g_image.image = img
        pos, rot = self.deviations()
        src = self.project.source_names[self.project.source[i]] if self.project.source[i] < len(
            self.project.source_names) else ""
        md = [f"**{name}** · {src} t = {self.project.pts[i]:.2f} s"]
        if self.snap.cams.valid[i] and self.snap.name != "priors":
            md.append(f"prior deviation: {pos[i]:.2f} m, {rot[i]:.2f}°")
        if st:
            md.append(f"{st['observations']} / {st['keypoints']} keypoints observed · "
                      f"error median {st['median_err_px']:.2f} px, p90 {st['p90_err_px']:.2f} px")
            md.append(f"green = observation, red = residual ×{self.g_gain.value:g}, grey = unmatched")
        self.g_inspect_md.content = "  \n".join(md)

    def _camera_colors(self):
        if self.g_color.value == "source":
            names = self.project.source_names
            self.g_legend.content = "  \n".join(f"{SOURCE_SWATCH[k % len(SOURCE_SWATCH)]} {k + 1}. {n}"
                                                 for k, n in enumerate(names))
            return np.array([source_color(k) for k in self.project.source], np.uint8)
        vals = self.color_values()
        n = len(self.project.names)
        if vals is None:
            self.g_legend.content = ""
            return np.tile(np.array([[70, 134, 251]], np.uint8), (n, 1))
        colors, vmax = colormap(vals)
        self.g_legend.content = f"blue → red: 0 … {vmax:.2f} {UNITS.get(self.g_color.value, '')} (p95); grey = n/a"
        return colors

    def _apply_colors(self):
        colors = self._camera_colors()
        for i, h in self.cam_handles.items():
            h.color = SELECTED_COLOR if i == self.selected else tuple(int(c) for c in colors[i])

    def _recolor(self, j):
        c = SELECTED_COLOR if j == self.selected else tuple(int(x) for x in self._camera_colors()[j])
        self.cam_handles[j].color = c

    # ------------------------------------------------------------- running
    def run_stage(self, stage):
        if self.running or self.ingest.running:
            self.g_state.content += "  \n_busy: wait for the running job to finish_"
            return
        if self.project is None and stage != "all":  # the full pipeline extracts them itself
            self.g_state.content = "⚪ **no images yet**: extract them in the Ingest tab first"
            return
        self.running = True
        self._run_id = getattr(self, "_run_id", 0) + 1
        for b in self.g_buttons.values():
            b.disabled = True
        self.g_progress.visible = True
        self.g_progress.value = 0.0
        threading.Thread(target=self._run, args=(stage, self._run_id), daemon=True).start()

    def _run(self, stage, run_id):
        project, t0 = self.path, time.time()
        self._job = {"name": stage, "t0": t0, "label": stage, "stage_t0": t0, "detail": ""}
        threading.Thread(target=self._ticker, args=(run_id,), daemon=True).start()
        pipeline.LISTENERS.append(self._on_log)
        final = None
        try:
            if stage == "all" and self.project is None:
                self._begin_stage("extract", 0, 1)
                self.ingest.extract_now()
                self._history(f"✅ **extract** · {len(self.project.names)} images")
            report = pipeline.Report(project)
            pri = self.project.priors
            if stage == "all":
                seq, skipped = list(pipeline.SFM_STAGES), []
                if splat.available() is None:
                    seq.append("splat")
                else:
                    skipped.append(f"splat ({splat.available()})")
                if dense.has_cuda():
                    seq += list(pipeline.DENSE_STAGES)
                else:
                    skipped.append("dense (no CUDA)")
                for sk in skipped:
                    self._history(f"⚪ **skipped** {sk}")
            elif stage == "sfm" or stage in pipeline.SFM_STAGES:
                seq = pipeline.SFM_STAGES if stage == "sfm" else (stage,)
            elif stage == "splat":
                seq = ("splat",)
            else:
                seq = pipeline.DENSE_STAGES if stage == "dense" else (stage,)
            m = self.g_mesh_method.value
            methods = dense.MESH_METHODS if m == "both" else (m,)
            for k, st in enumerate(seq):
                self._begin_stage(st, k, len(seq))
                ts = time.time()
                if st == "features":
                    pipeline.stage_features(project, report)
                elif st == "pairs":
                    pipeline.stage_pairs(project, report, pri, self.project.meta)
                elif st == "match":
                    pipeline.stage_match(project, report)
                elif st == "refine":
                    pipeline.stage_refine(project, report, pri, progress=self._on_refine_event)
                elif st == "export":
                    pipeline.stage_export(project, report)
                elif st == "splat":
                    self._run_splat(project)
                    report = pipeline.Report(project)  # splat.run wrote its stats
                else:
                    pipeline.stage_dense(project, report, st, methods, DENSE_QUALITY[self.g_dense_quality.value])
                self._history(f"✅ **{st}** · {time.time() - ts:.0f} s · {self._stage_summary(st, report.data)}")
            if stage in ("dense", "splat", "all") or stage in pipeline.DENSE_STAGES:
                self._refresh_dense()
            self.refresh_snapshot_list(select="refined" if stage in ("refine", "sfm", "all") else None)
            final = (f"✅ **{stage} finished** in {_fmt_s(time.time() - t0)} · {time.strftime('%H:%M:%S')}  \n"
                     f"{NEXT_STEP.get(stage, '')}")
        except Exception as e:  # surface failures in the GUI instead of killing the thread silently
            traceback.print_exc()
            label = self._job["label"]
            self._history(f"❌ **{label}** failed: `{e}`")
            final = (f"❌ **{label} failed** after {_fmt_s(time.time() - t0)}: `{e}`  \n"
                     "details in the Log folder and the server console")
        finally:
            pipeline.LISTENERS.remove(self._on_log)
            with self.lock:  # the ticker writes under the same lock and checks `running`
                self.running = False
                self.g_state.content = final or "❌ **stopped**"
            unavailable = {"stereo": not dense.has_cuda(), "dense": not dense.has_cuda(),
                           "splat": splat.available() is not None}
            for name, b in self.g_buttons.items():
                b.disabled = unavailable.get(name, False)
            self.g_progress.visible = False

    def _run_splat(self, project):
        self._splat_job = splat.Job()
        self.g_splat_stop.disabled = False
        iters = int(self.g_splat_iters.value)

        self._live_msg = ""

        def progress(step, pct):
            self.g_progress.value = pct
            self._job["detail"] = f"training step {step:,} / {iters:,} ({pct:.1f} %)" + (
                f"  \n{self._live_msg}" if self._live_msg else "")

        def on_line(m):
            self._on_log(m)
            if m.startswith("splat:"):  # also on the server console, for diagnosing the preview
                print(m, flush=True)
        try:
            live = self.g_splat_live.value
            if live:  # the live layer stands in for the previous result while training
                self.server.scene.remove_by_name("/dense/splat")
                self._splat_live_on = True
                self.g_splat.disabled, self.g_splat.value = False, True  # View tab toggle drives the live layer
                self.g_splat_status.content = (f"live preview: first snapshot at step "
                                               f"{int(self.g_splat_every.value):,}, then in the View tab")
            splat.run(project, self.g_splat_method.value, iters, on_line=on_line, on_progress=progress,
                      job=self._splat_job, cache="disk" if self.g_splat_disk.value else "gpu",
                      downscale=int(self.g_splat_down.value), preview_every=int(self.g_splat_every.value),
                      on_snapshot=self._on_splat_snapshot if live else None)
            self.g_splat.value = True  # show the finished splat (drawn by _refresh_dense after the run)
        finally:
            self.g_splat_stop.disabled = True
            self._splat_job = None
            self._splat_live_on = False
            self.server.scene.remove_by_name("/dense/splat_live")

    def _on_splat_snapshot(self, step, npz, transforms):
        """A new training checkpoint: redraw the live splat in the View tab."""
        step, n, arrays = gdata.load_live_splat(npz, transforms)
        self._live_splat = arrays
        self._draw_live_splat()
        msg = f"live splat: step {step:,}, {n:,} gaussians" + (f" ({len(arrays[0]):,} shown)" if len(arrays[0]) < n
                                                                 else "")
        self.g_dense_md.content = msg
        self.g_splat_status.content = msg + " · View tab → Dense / splat → Splat"
        self._live_msg = msg + " (View tab)"
        self._on_log(f"splat: live preview at step {step:,} ({n:,} gaussians)")
        print(f"splat: live preview at step {step:,} ({n:,} gaussians)", flush=True)

    def _draw_live_splat(self):
        a = getattr(self, "_live_splat", None)
        if a is not None and self.g_splat.value:
            c, cov, rgb, op = a
            self.server.scene.add_gaussian_splats("/dense/splat_live", centers=c, covariances=cov, rgbs=rgb,
                                                  opacities=op)
        else:
            self.server.scene.remove_by_name("/dense/splat_live")

    def _begin_stage(self, st, k, n):
        j = self._job
        j["label"] = f"{st} ({k + 1}/{n})" if n > 1 else st
        j["stage_t0"], j["detail"] = time.time(), ""
        if n > 1:
            self.g_progress.value = 100.0 * k / n
        self._on_log(f"--- {st} started")

    def _ticker(self, run_id):
        """Rewrites the state line every second until this run ends (no stale 'running' after it)."""
        while True:
            with self.lock:
                if not self.running or self._run_id != run_id:
                    return
                j = self._job
                now = time.time()
                self.g_state.content = (
                    f"⏳ **running {j['label']}** · stage {_fmt_s(now - j['stage_t0'])} · "
                    f"total {_fmt_s(now - j['t0'])}" + (f"  \n{j['detail']}" if j["detail"] else ""))
            time.sleep(1.0)

    def _on_log(self, msg):
        lines = getattr(self, "_log_lines", [])
        for line in str(msg).splitlines():
            lines.append(f"{time.strftime('%H:%M:%S')} {line[:240]}" + ("…" if len(line) > 240 else ""))
        self._log_lines = lines[-LOG_LINES:]
        self.g_log.content = "```\n" + "\n".join(self._log_lines) + "\n```"

    def _history(self, line):
        h = getattr(self, "_history_lines", [])
        h.append(f"{time.strftime('%H:%M:%S')} {line}")
        self._history_lines = h[-30:]
        self.g_history.content = "  \n".join(reversed(self._history_lines))

    @staticmethod
    def _stage_summary(st, rep):
        r = rep.get(st, {}) if st in pipeline.SFM_STAGES else rep.get("dense", {}).get(st, {})
        try:
            if st == "features":
                return f"{r['images']} images, {r['keypoints']:,} keypoints"
            if st == "pairs":
                return f"{r['num_pairs']:,} pairs"
            if st == "match":
                return f"{r['verified_pairs']:,} verified pairs, {r['inlier_matches']:,} inlier matches"
            if st == "refine":
                f = r["final"]
                e, d = f["point_error_px"], f["prior_deviation"]["position_m"]
                return (f"{f['reg_images']} cameras, {f['points3D']:,} points, error median {e['median']} px "
                        f"(p90 {e['p90']}), prior deviation median {d['median']} m (max {d['max']})")
            if st == "export":
                return f"{r['frames']} frames → transforms.json"
            if st == "fusion":
                return f"{r.get('points', r)}"
            if st == "splat":
                r = rep["splat"]
                return (f"{r['num_gaussians']:,} gaussians, {r['method']} {r['iterations']:,} it, "
                        f"training {_fmt_s(r['train_seconds'])}")
        except (KeyError, TypeError):
            pass
        text = str(r)
        return text[:160] + ("…" if len(text) > 160 else "")

    def _on_refine_event(self, ev):
        phase, rnd, rounds = ev["phase"], ev.get("round", 0), ev.get("rounds", 1)
        if phase == "triangulate":
            self._job["detail"] = f"round {rnd + 1}/{rounds}: triangulating …"
            self.g_progress.value = 100.0 * rnd / rounds
        elif phase == "ba":
            now = time.time()
            if now - self._last_live < 0.25:  # throttle scene updates
                return
            self._last_live = now
            self._job["detail"] = (f"round {rnd + 1}/{rounds}: bundle adjustment iteration {ev['iteration']} "
                                   f"(max 500), cost {ev['cost']:.4g}")
            self._update_cameras_live(ev["rec"])
        elif phase in ("round_done", "done"):
            with self.lock:
                name = f"round_{rnd}" if phase == "round_done" else "refined"
                self.snap = self.project.snapshot_from_reconstruction(name, ev["rec"])
                self._draw_points()
                self._draw_cameras()
                self._draw_lines()
                self._update_stats()
                self._update_plots()
            self.g_progress.value = 100.0 * (rnd + 1) / rounds if phase == "round_done" else 100.0

    def _update_cameras_live(self, rec):
        """Move existing frustums to the poses Ceres is currently evaluating (cheap: no re-adding)."""
        cams = self.project.poses_from_reconstruction(rec)
        for i, h in self.cam_handles.items():
            if cams.valid[i]:
                h.position = cams.centers[i]
                h.wxyz = cams.wxyz(i)
        self._draw_lines(cams)

    # ------------------------------------------------------------- dense / splat
    def _refresh_dense(self):
        self.dense = gdata.dense_artifacts(self.path)
        meshes = tuple(self.dense["meshes"])
        self.g_dense_pts.disabled = self.dense["fused"] is None
        self.g_splat.disabled = self.dense["splat"] is None
        self.g_mesh.options = ("off",) + meshes
        self.g_mesh.disabled = not meshes
        have = [k for k, v in (("dense points", self.dense["fused"]), ("splat", self.dense["splat"])) if v] + \
            [f"mesh ({m})" for m in meshes]
        self.g_dense_md.content = "available: " + (", ".join(have) if have else "nothing yet (Dense / Splat tabs)")
        self._draw_dense_points()
        self._draw_mesh()
        self._draw_splat()
        self._refresh_export()

    # ------------------------------------------------------------- export
    def _refresh_export(self, message=None):
        have = products.available(self.path)
        meshes = tuple(have["meshes"])
        self.g_export_mesh.options = meshes or ("none",)
        self.g_export_mesh.disabled = not meshes
        busy = getattr(self, "_exporting", False)
        for k, b in self.g_export_buttons.items():
            b.disabled = busy or not (meshes if k.startswith("mesh") else have["splat"])
        mesh_list = ", ".join(m + (" (textured)" if v["textured"] else "") for m, v in have["meshes"].items())
        lines = [f"mesh: {mesh_list}" if meshes else "mesh: none yet (Dense tab)",
                 "splat: splat/splat_enu.ply" if have["splat"] else "splat: none yet (Splat tab)"]
        out = products.out_dir(self.path)
        if os.path.isdir(out):
            files = sorted(f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f)))
            lines.append(f"`export/`: {len(files)} files ({projects.human(projects.size(out))})")
        if message:
            lines.append(message)
        self.g_export_md.content = "  \n".join(lines)

    def _export(self, what, client, download):
        if getattr(self, "_exporting", False):
            return
        self._exporting = True
        self._refresh_export(f"⏳ exporting {what} …")

        def work():
            msg = None
            try:
                t0 = time.time()
                if what == "mesh":
                    paths = products.export_mesh(self.path, self.g_export_mesh.value, self.g_export_mesh_fmt.value,
                                                 self.g_export_textured.value)
                else:
                    paths = products.export_splat(self.path, self.g_export_splat_fmt.value)
                rel = [os.path.relpath(p, self.path) for p in paths]
                msg = f"✅ wrote `{rel[0]}`" + (f" (+{len(rel) - 1} files)" if len(rel) > 1 else "") + \
                    f" in {time.time() - t0:.1f} s"
                self._history(f"✅ **export {what}** · `{rel[0]}`")
                if download and client is not None:
                    name, data = _download_payload(paths)
                    client.send_file_download(name, data)
                    msg += f" · sent `{name}` ({len(data) / 1e6:.1f} MB) to the browser"
            except Exception as e:
                traceback.print_exc()
                msg = f"❌ export {what} failed: `{e}`"
            finally:
                self._exporting = False
                self._refresh_export(msg)
        threading.Thread(target=work, daemon=True).start()

    def _draw_dense_points(self):
        if self.g_dense_pts.value and self.dense["fused"]:
            pts, cols, n = gdata.load_fused(self.dense["fused"])
            self.server.scene.add_point_cloud("/dense/points", points=pts, colors=cols,
                                              point_size=self.g_point_size.value * 0.5, precision="float32")
            self.g_dense_md.content = f"dense points: {len(pts):,} of {n:,} shown"
        else:
            self.server.scene.remove_by_name("/dense/points")

    def _draw_mesh(self):
        method = self.g_mesh.value
        if method != "off" and method in self.dense["meshes"]:
            mesh = gdata.load_mesh(self.dense["meshes"][method])
            self.server.scene.add_mesh_trimesh("/dense/mesh", mesh)
            self.g_dense_md.content = f"mesh ({method}): {len(mesh.faces):,} faces"
        else:
            self.server.scene.remove_by_name("/dense/mesh")

    def _draw_splat(self):
        if getattr(self, "_splat_live_on", False):  # training: the checkbox shows / hides the live layer
            self._draw_live_splat()
            return
        if self.g_splat.value and self.dense["splat"]:
            c, cov, rgb, op = gdata.load_splat(self.dense["splat"])
            self.server.scene.add_gaussian_splats("/dense/splat", centers=c, covariances=cov, rgbs=rgb, opacities=op)
            self.g_dense_md.content = f"splat: {len(c):,} gaussians"
        else:
            self.server.scene.remove_by_name("/dense/splat")

    def serve_forever(self):
        while True:
            time.sleep(3600)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", nargs="?",
                    help="project folder to open (new or existing); default: the newest project under --root")
    ap.add_argument("--root", default="data", help="folder holding the projects (default data/)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if args.project:
        path = os.path.abspath(args.project)
    else:
        found = projects.list_projects(root)
        path = found[0] if found else os.path.join(root, "project_1")
    app = App(path, args.port, root=root)
    print(f"modelforge GUI on http://localhost:{args.port}", flush=True)
    app.serve_forever()


if __name__ == "__main__":
    main()

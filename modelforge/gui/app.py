"""modelforge web GUI (viser): ingest videos, run stages live, inspect snapshots and products.

Usage (in the dev container; port 8080 is published by compose.yaml):
    python -m modelforge.gui.app [data/site] [--root data] [--port 8080]
Projects are folders under --root (Projects tab: open / create / clean). A project folder
may be new or empty: the Ingest tab builds it from one or more videos.
"""

import argparse
import json
import os
import threading
import time
import traceback

import numpy as np
import viser

from modelforge import dense, jobs, pipeline, projects, semantic, splat, splat_mesh, splat_plan
from modelforge.dji import ingest
from modelforge.export import products
from modelforge.sfm import chunked, features, reconstruct
from modelforge.gui import data as gdata
from modelforge.gui.data import Project
from modelforge.gui.hardware import HardwarePanel
from modelforge.gui.videos import VideosPanel, upload_progress
from modelforge.gui.ingest import IngestPanel, source_color
from modelforge.gui.projects import ProjectsPanel
from modelforge.gui.semantic import SemanticPanel
from modelforge.gui.subset import SubsetPanel

SFM_BUTTONS = ("features", "pairs", "match", "refine", "reconstruct", "export")  # sparse stages of both kinds

DENSE_QUALITY = {"superfast (320 px)": "superfast", "low (640 px, fast)": "low", "medium (1200 px)": "medium",
                 "high (full res)": "high", "max (COLMAP defaults)": "max"}  # label -> dense.QUALITY preset
DENSE_QUALITY_DEFAULT = next(k for k, v in DENSE_QUALITY.items() if v == dense.DEFAULT_QUALITY)

QUICK_SHORT = {"Cameras": "Cams", "Residuals": "Resid", "Trajectory": "Traj"}  # quick bar labels (two short rows)
STAGE_HINTS = {
    "features": "SIFT keypoints for every image into features.db (upright, up to 8192 per image, one shared "
                "camera). A few minutes on CPU.",
    "pairs": "Choose image pairs to match from the priors: time neighbours within each source plus views whose "
             "look-at points overlap (catches revisits and links sources). Seconds.",
    "match": "Match the selected pairs and verify them geometrically into database.db. The slowest SfM stage "
             "on CPU (~12 min for 2400 pairs).",
    "refine": "Start every camera at its prior, then 3 rounds of triangulate + bundle adjustment with "
              "self-calibrating 6-DoF priors. Cameras move live in the View tab. ~1.5-3 min for 300 images. "
              "From 2500 images (BA mode auto): calibration on a subset, chunks, a pose graph, a polish sweep.",
    "reconstruct": "Videos without metadata: incremental SfM (COLMAP mapper) from the images alone, then the model "
                   "is oriented up and scaled (no metric scale). Live count of registered images; Stop works.",
    "export": "Write transforms.json + sparse_pc.ply (nerfstudio dataset, ENU world) from sparse/refined. Seconds.",
    "undistort": "Undistorted images + PINHOLE model into dense/ (COLMAP MVS workspace). Seconds.",
    "stereo": "PatchMatch depth and normal maps for every image. CUDA only; the long one (~10 GB at full res).",
    "fusion": "Fuse depth maps into dense/fused.ply (+ .vis for meshing).",
    "mesh": "Mesh the fused points (method above) and write a simplified preview for the View tab.",
    "texture": "Texture the mesh from the images: dense/textured-<method>/mesh.ply + texture.png.",
    "splat": "nerfstudio splatfacto: train, export, convert to ENU (splat/splat_enu.ply).",
    "semantic": "C-RADIOv3 features of the photos projected onto the splat's Gaussians (the Search tab).",
    "analyze": "Overlap analysis of all extracted images (features + sequential matches) for the keyframe "
               "selection in Images used.",
    "splat-resume": "Continue the paused or stopped splat training from its last checkpoint, with its own "
                    "settings (method, steps, schedule): optimizer state and learning-rate schedules carry on.",
    "splatmesh": "Mesh from the splat: depth rendered at every camera, fused into a TSDF, marching cubes "
                 "(splatmesh/mesh.ply, vertex colours). Seconds to a minute on the GPU.",
    "all": "Full pipeline: sparse → splat → splat mesh (→ COLMAP dense if ticked in the Dense tab).",
}
NEXT_STEP = {
    "all": "View tab: refined cameras, mesh and splat; Export tab for the files",
    "sfm": "View tab → snapshot **refined**; next: Dense tab (GPU) or Splat tab",
    "features": "next: **pairs**", "pairs": "next: **match**", "match": "next: **refine**",
    "refine": "View tab shows **refined**; next: **export**",
    "reconstruct": "View tab shows **refined**; next: **export**",
    "export": "transforms.json written; next: Dense tab (GPU) or Splat tab",
    "dense": "View tab → Dense / splat; Export tab for the mesh",
    "undistort": "next: **stereo**", "stereo": "next: **fusion**", "fusion": "next: **mesh**",
    "mesh": "View tab → Mesh; next: **texture**", "texture": "textured mesh written; Export tab",
    "splat": "View tab → Dense / splat → **Splat (ENU)**; Export tab",
    "splat-resume": "View tab → Dense / splat → **Splat (ENU)**; Export tab",
    "analyze": "Sparse tab → Images used → Selection **keyframes**, then Use these images",
    "semantic": "Search tab (left): type a query, **Search**",
    "splatmesh": "View tab → Dense / splat → Mesh **splat**; Export tab",
}
LOG_LINES = 200

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
        self.server = viser.ViserServer(host="0.0.0.0", port=port, label="ModelForge")
        self.server.scene.set_up_direction("+z")
        self.dark = self._load_settings().get("dark_mode", False)
        self.server.gui.configure_theme(control_width="large", dark_mode=self.dark)
        self.lock = threading.RLock()
        self.cam_handles, self.prior_handles = {}, {}
        self.selected = None
        self.snap = None
        self.running = False
        self.fov, self.aspect = np.radians(53.0), 16 / 9
        self._last_live = 0.0
        self._run_id = 0
        self._job = {"name": "", "t0": 0.0, "label": "", "stage_t0": 0.0, "detail": ""}
        self._token = None  # jobs.Token of the running job (the Stop button cancels it)
        self._orbits = {}  # client id -> threading.Event that stops its orbit
        self._flights = {}  # client id -> {"stop": Event, "jump": None or path time to jump to}
        self._fly_guard = False  # set while the flight moves the position slider (not a user scrub)
        self._build_gui()
        self.subset.refresh()
        self._apply_kind()
        self._set_aspect()
        if self.project:
            names = self.project.snapshot_names()
            self.load_snapshot("refined" if "refined" in names else names[-1])
        self._refresh_dense()
        self._auto_show_sparse()
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
                           hint="Sparse → splat → mesh from the splat in one go (COLMAP's dense too if ticked in the Dense "
                                "tab), with the settings of the tabs below (and the "
                                "Ingest plan: images are extracted first if the project has none yet). Stages that "
                                "need a GPU are skipped without one.")
        b.on_click(lambda _: self.run_stage("all"))
        self.g_buttons["all"] = b
        self.g_stop = gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP, color="red", disabled=True,
                                     hint="Stop the running job (a stage, the full pipeline, splat training or an "
                                          "extraction). Subprocess stages (stereo, meshing, texturing) and "
                                          "bundle adjustment stop at once; feature extraction, matching, "
                                          "triangulation and fusion stop when their current call returns. Splat "
                                          "training saves a checkpoint at the current step, then exports that "
                                          "partial splat (shown and exportable; Splat tab → Resume continues).")
        self.g_stop.on_click(lambda _: self.stop())
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
                "round N; round_N = after its bundle adjustment; calib_N = the calibration subset of a chunked refine; "
                "posegraph = a chunked refine after the pose graph and polish; "
                "refined = final model.")
            self.g_snapshot.on_update(lambda _: self.load_snapshot(self.g_snapshot.value))
            self.g_stats = gui.add_markdown("")
            with gui.add_folder("Layers"):
                self.g_points = gui.add_checkbox("Points", initial_value=False,
                                                 hint="Sparse 3D points of the snapshot.")
                self.g_point_size = gui.add_slider("Point size", min=0.01, max=0.5, step=0.01, initial_value=0.08)
                self.g_cams = gui.add_checkbox("Cameras", initial_value=False,
                                               hint="Camera frustums of the snapshot; click one to inspect it.")
                self.g_thumbs = gui.add_checkbox("Thumbnails", initial_value=False,
                                                 hint="Image thumbnails in the frustums (slower with many cameras).")
                self.g_priors = gui.add_checkbox("Prior cameras", initial_value=False,
                                                 hint="Grey frustums at the telemetry poses.")
                self.g_corr = gui.add_checkbox("Prior → refined", initial_value=True,
                                               hint="Red segments from each prior position to the estimated one.")
                self.g_traj = gui.add_checkbox("Trajectories", initial_value=True,
                                               hint="Grey: prior path; blue: estimated path.")
                self.g_plan = gui.add_checkbox(
                    "Flight plan", initial_value=self.ingest.g_show.value,
                    hint="The Ingest plan: each source's camera track (in its colour inside the range, grey "
                         "outside) and dots at the planned image positions. Same as the Ingest tab's "
                         "\"Show flight plan in 3D\" and the quick bar's Plan.")
                self.g_scale = gui.add_slider("Camera size", min=0.2, max=5.0, step=0.1, initial_value=1.0)
            with gui.add_folder("Dense / splat"):
                self.g_dense_pts = gui.add_checkbox("Dense points", initial_value=False, disabled=True,
                                                    hint="dense/fused.ply (subsampled to 1.5M points). Enabled once "
                                                         "fusion has run.")
                self.g_mesh = gui.add_dropdown("Mesh", options=("off",), initial_value="off", disabled=True,
                                               hint="The dense mesh: textured once the texture stage has run, else "
                                                    "the grey simplified preview (≤300k faces).")
                self.g_mesh_tex = gui.add_checkbox("Mesh texture", initial_value=True,
                                                   hint="Show the textured mesh (colour from the images). Off: the "
                                                        "grey simplified preview, lighter on the browser. COLMAP's "
                                                        "meshers write geometry only, so the colour comes from the "
                                                        "texture stage.")
                self.g_splat = gui.add_checkbox("Splat (ENU)", initial_value=True, disabled=True,
                                                hint="splat/splat_enu.ply, in the same frame as the cameras and "
                                                     "mesh; shown as soon as training finishes (and live while it "
                                                     "trains).")
                self.g_splat_all = gui.add_checkbox(
                    "All gaussians", initial_value=True,
                    hint=f"Show every Gaussian of the splat. Off: a random subset of at most "
                         f"{gdata.SPLAT_SHOWN:,} (finished splat) or {gdata.LIVE_SPLAT_SHOWN:,} (live preview while "
                         f"training), which keeps the browser responsive. Millions of Gaussians need a strong GPU "
                         f"in the browser and take a while to send.")
                self.g_dense_md = gui.add_markdown("")
                self.g_splat_all.on_update(lambda _: self._splat_all_changed())
                self.g_dense_pts.on_update(lambda _: self._draw_dense_points())
                self.g_mesh.on_update(lambda _: self._draw_mesh())
                self.g_mesh_tex.on_update(lambda _: self._draw_mesh())
                for h in (self.g_dense_pts, self.g_mesh, self.g_splat):  # keep the quick bar in step
                    h.on_update(lambda _: self._sync_quickbar())
                self.g_splat.on_update(lambda _: self._draw_splat())
            self.g_color = gui.add_dropdown("Color cameras by", options=COLOR_BY, initial_value="position deviation",
                                           hint="Deviation from the prior, mean reprojection error, capture "
                                                "time or source (video piece).")
            self.g_legend = gui.add_markdown("")
            gui.add_button("Center the view", icon=viser.Icon.FOCUS_CENTERED,
                           hint="Look at the whole scene (cameras and sparse points) from the south, above it. Use it "
                                "when the view has wandered off. Also in the quick bar (⌖ Center).").on_click(
                lambda e: self.recenter(e.client))
            with gui.add_folder("Orbit"):
                self.g_orbit = gui.add_button(
                    "Start orbit", icon=viser.Icon.ROTATE_360,
                    hint="Circle the camera around the point you are looking at, keeping your distance and "
                         "height; press again to stop. You can still zoom, tilt and pan with the mouse: the orbit "
                         "pauses while you drag and continues from the new view. Only moves the view of the "
                         "browser that pressed it.")
                self.g_orbit.on_click(lambda e: self.toggle_orbit(e.client))
                self.g_orbit_speed = gui.add_slider("Speed (°/s)", min=1, max=90, step=1, initial_value=12,
                                                    hint="Orbit speed; can be changed while orbiting.")
            with gui.add_folder("Fly"):
                self.g_fly = gui.add_button(
                    "Fly the camera path", icon=viser.Icon.PLANE,
                    hint="Replay the flight: the view follows the cameras of the shown snapshot in capture order "
                         "(position and orientation interpolated smoothly between them), looking through the "
                         "camera. Press again to stop. Dragging with the mouse pauses it; it resumes on the path "
                         "a second later. Only moves the view of the browser that pressed it.")
                self.g_fly.on_click(lambda e: self.toggle_fly(e.client))
                self.g_fly_speed = gui.add_slider(
                    "Speed (m/s)", min=0.1, max=20.0, step=0.1, initial_value=0.5,
                    hint="Speed along the camera path (scene units per second for videos without metadata). "
                         "Turns on the spot count 0.02 m per degree, so they take time too; jumps between videos "
                         "are shortened to 2 m. Can be changed while flying.")
                self.g_fly_pos = gui.add_slider(
                    "Position (%)", min=0.0, max=100.0, step=0.1, initial_value=0.0,
                    hint="Where on the path the flight is. Drag it to jump there (also while not flying: the view "
                         "moves to that point); Fly starts from here.")
                self.g_fly_pos.on_update(lambda e: self._fly_scrub(e.client))
                self.g_fly_fov = gui.add_checkbox(
                    "Camera's field of view", initial_value=True,
                    hint="Use the real camera's field of view while flying, so the view matches the images; your "
                         "own field of view comes back when the flight stops.")
                self.g_fly_loop = gui.add_checkbox("Loop", initial_value=True,
                                                   hint="Start over at the end instead of stopping.")
                self.g_fly_md = gui.add_markdown("")
            for h in (self.g_points, self.g_point_size):
                h.on_update(lambda _: self._draw_points())
            for h in (self.g_points, self.g_cams, self.g_thumbs, self.g_corr, self.g_traj):
                h.on_update(lambda _: self._sync_quickbar())
            for h in (self.g_cams, self.g_thumbs, self.g_priors, self.g_scale):
                h.on_update(lambda _: self._draw_cameras())
            self.g_color.on_update(lambda _: self._apply_colors())
            for h in (self.g_corr, self.g_traj):
                h.on_update(lambda _: self._draw_lines())
            self.g_plan.on_update(lambda _: self._set_plan_visible(self.g_plan.value))
            self.ingest.g_show.on_update(lambda _: self._set_plan_visible(self.ingest.g_show.value))

        with self.left.add_tab("Search", viser.Icon.SEARCH):  # semantic search, its own tab after View
            self.semantic = SemanticPanel(self, gui)

        with self.left.add_tab("Inspect", viser.Icon.PHOTO):
            self.g_inspect_md = gui.add_markdown("Click a camera to inspect it.")
            self.g_gain = gui.add_slider("Residual ×", min=1.0, max=50.0, step=1.0, initial_value=10.0,
                                         hint="Magnification of the red reprojection-residual lines.")
            self.g_gain.on_update(lambda _: self.select(self.selected) if self.selected is not None else None)
            self.g_image = gui.add_image(np.zeros((54, 96, 3), np.uint8), label="observations")

        with tabs.add_tab("Sparse", viser.Icon.CAMERA):
            self.g_sparse_md = gui.add_markdown("")
            self.subset = SubsetPanel(self, gui)
            self.g_buttons["analyze"] = self.subset.g_analyze
            b = gui.add_button("Run SfM (all)", icon=viser.Icon.PLAYER_TRACK_NEXT,
                               hint="Runs features → pairs → match → refine → export in sequence. "
                                    "About 15-20 min on CPU for 300 images; matching dominates.")
            b.on_click(lambda _: self.run_stage("sfm"))
            self.g_buttons["sfm"] = b
            self.g_ba_iters = gui.add_number(
                "BA iterations (max per round)", initial_value=500, min=0, step=50,
                hint="Cap on bundle-adjustment iterations in each of the 3 refine rounds. BA stops by itself once "
                     "the cameras settle (seconds on 80 images), so the cap rarely matters. 0 = no BA: poses stay at "
                     "the priors and the focal length at its start value (2x off), which leaves ~15 px reprojection "
                     "error instead of ~0.4 px and hurts dense and splat.")
            self.g_ba_mode = gui.add_dropdown(
                "BA mode", options=pipeline.BA_MODES, initial_value="auto",
                hint="global: one bundle adjustment over all images per round. chunked (for thousands of images): "
                     "the intrinsics and the prior model are calibrated on 300 images spread over the flight; then "
                     "each chunk of consecutive images is refined on its own (its matches only, so memory does not "
                     "grow with the project), a pose graph (cameras only, no points) joins the chunks under the "
                     "priors, and one sweep of windows polishes all images. auto: chunked from 2500 images on "
                     "(global BA pulls revisits of a long flight together better: on 2000 images 0.05 m between "
                     "passes vs 0.24 m chunked, but 67 vs 24 min).")
            self.g_chunk_size = gui.add_number(
                "Chunk size (images)", initial_value=chunked.Chunked.size, min=100, step=100,
                hint="Images per chunk in a chunked refine; consecutive chunks share the overlap.")
            self.g_chunk_overlap = gui.add_number(
                "Chunk overlap (images)", initial_value=chunked.Chunked.overlap, min=10, step=10,
                hint="Images shared by consecutive chunks: they tie the chunks together in the pose graph.")
            self.g_polish = gui.add_number(
                "Polish sweeps", initial_value=chunked.Chunked.polish_sweeps, min=0, max=3, step=1,
                hint="Chunked refine: sweeps of 300-image windows over all images after the pose graph (each window "
                     "adjusted with all its points, the rest held fixed). They also bring in the matches between "
                     "chunks (revisits). 0 = none.")
            self.g_skip = gui.add_button(
                "Skip refinement", icon=viser.Icon.PLAYER_SKIP_FORWARD, disabled=True,
                hint="While refine runs: keep the poses as they are now, skip the remaining rounds, and go on "
                     "(final triangulation, then export / splat / dense in the full pipeline).")
            self.g_skip.on_click(lambda _: self.skip_refine())
            self.g_skip_round = gui.add_button(
                "Skip this BA round", icon=viser.Icon.PLAYER_TRACK_NEXT, disabled=True,
                hint="While refine runs: end the current bundle adjustment where it is (e.g. when the cameras no "
                     "longer move in the 3D view) and go on with the next round as usual. In a chunked refine it "
                     "ends the current chunk round or polish window. Can be pressed again in later rounds.")
            self.g_skip_round.on_click(lambda _: self.skip_ba_round())
            with gui.add_folder("Features", expand_by_default=False):
                self.g_extractor = gui.add_dropdown(
                    "Extractor", options=tuple(features.EXTRACTORS), initial_value="sift",
                    hint=" ".join(f"{k}: {v}." for k, v in features.EXTRACTORS.items())
                    + " Learned features (ALIKED, LightGlue) need COLMAP built with ONNX, which this image is not.")
                self.g_max_features = gui.add_number(
                    "Max features", initial_value=8192, min=500, max=65536, step=500,
                    hint="Per image. More helps high-resolution or detailed scenes; fewer is faster.")
                self.g_upright = gui.add_checkbox(
                    "Upright", initial_value=True,
                    hint="One keypoint orientation: right when the horizon stays level (DJI gimbal; default on), "
                         "wrong for handheld footage that rolls (videos without metadata; default off).")
            with gui.add_folder("Matching", expand_by_default=False):
                self.g_pairs = gui.add_dropdown(
                    "Pairs", options=tuple(pipeline.PAIR_MODES), initial_value="auto",
                    hint="Which images are matched. " + " ".join(f"{k}: {v}." for k, v in pipeline.PAIR_MODES.items()))
                self.g_neighbors = gui.add_number(
                    "Neighbours", initial_value=10, min=1, max=100, step=1,
                    hint="For sequential pairs: how many following images each one is matched with.")
                self.g_ratio = gui.add_slider(
                    "Ratio test", min=0.5, max=0.95, step=0.01, initial_value=0.8,
                    hint="Lowe's ratio: lower is stricter (fewer, cleaner matches); higher keeps more on repetitive "
                         "or low-texture scenes. COLMAP's default is 0.8.")
                self.g_guided = gui.add_checkbox(
                    "Guided matching", initial_value=False,
                    hint="A second pass that re-matches along the estimated two-view geometry: more inliers, slower.")
                self.g_min_inliers = gui.add_number(
                    "Min inliers", initial_value=15, min=5, max=500, step=5,
                    hint="Geometric verification: consistent matches a pair needs to count (COLMAP's default 15).")
            self.g_mapper = gui.add_dropdown(
                "Mapper", options=tuple(reconstruct.MAPPERS), initial_value="incremental",
                hint="SfM for videos without metadata. "
                     + " ".join(f"{k}: {v}." for k, v in reconstruct.MAPPERS.items())
                     + " Incremental shows each registered image; global and hierarchical show their stages.")
            for stage in SFM_BUTTONS:  # refine (DJI) or reconstruct (no metadata): _apply_kind shows one
                b = gui.add_button(f"Run {stage}" + (" (live)" if stage == "refine" else ""),
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
                "Iterations", initial_value=0, min=0, step=1000,
                hint="Training steps; 0 = auto: ~20 views per training image, at least 30000 (splatfacto's "
                     "default) and at most 300000. The learning-rate decay and the densification schedule "
                     "follow the value. Use a few thousand for a quick check.")
            self.g_splat_cache = gui.add_dropdown(
                "Image cache", options=("auto",) + tuple(reversed(splat.CACHE_MODES)), initial_value="auto",
                hint="Where the training photos are kept. gpu: decoded in GPU memory (costs GPU memory the "
                     "Gaussians then lack: 471 4K photos take 11.7 GB). cpu: decoded once in RAM. disk: decoded "
                     "in parallel workers at every step (only ~6 % slower on karlak; the only option when the "
                     "photos fit nowhere, e.g. thousands of 4K photos). auto: gpu if the photos take < 1/4 of the "
                     "GPU memory, else cpu if twice their size (nerfstudio keeps two copies) fits in half the free "
                     "RAM, else disk.")
            self.g_splat_down = gui.add_dropdown(
                "Downsample factor", options=("auto",) + tuple(str(f) for f in splat.DOWNSCALES),
                initial_value="auto",
                hint="Train on images shrunk by this factor (2 = half width and height): faster and less memory. "
                     "auto: to ~2K width, i.e. 2 for 4K photos, 1 up to 2.7K. On karlak (471 4K photos) full "
                     "resolution grew no more Gaussians and was no sharper, even viewed at 4K, in 2.4× the time. "
                     "The shrunk copies go to images_<factor>/ (an intermediate, removed by Clean).")
            self.g_splat_strategy = gui.add_dropdown(
                "Densification", options=("auto",) + splat.STRATEGIES, initial_value="auto",
                hint="How Gaussians are added. default: split / clone / prune by gradient, with the opacity reset "
                     "period scaled to the image count (splatfacto's fixed 3000 steps blocks densification "
                     "entirely from ~2900 training images). mcmc: a fixed budget of Gaussians relocated where "
                     "needed. auto = default: on karlak mcmc built up semi-transparent fog in front of some "
                     "viewpoints and smudged detail; default densification did not, with half the Gaussians in "
                     "a fraction of the time. Pick mcmc to hold a hard Gaussian cap (GPU memory).")
            self.g_splat_maxg = gui.add_number(
                "Max gaussians (M)", initial_value=0.0, min=0.0, step=0.5,
                hint="mcmc budget in millions; 0 = auto: what the scene needs at the downsample factor, "
                     "within the GPU memory estimate (~0.35M per GB above 4 GB).")
            self.g_splat_camopt = gui.add_dropdown(
                "Camera pose refinement", options=("auto",) + splat.CAMERA_OPT, initial_value="auto",
                hint="splatfacto's camera optimizer: corrects small pose errors while training (rotation + "
                     "translation per image). auto = SO3xR3. off = keep the refined poses as they are.")
            self.g_splat_plan = gui.add_markdown("")
            for g in (self.g_splat_method, self.g_splat_iters, self.g_splat_down, self.g_splat_strategy,
                      self.g_splat_maxg, self.g_splat_camopt):
                g.on_update(lambda _: self._refresh_splat_plan())
            self.g_splat_lidar = gui.add_checkbox(
                "Start from LiDAR points", initial_value=True,
                hint="ROS 2 bag projects: the splat's initial cloud (sparse_pc.ply) gets the LiDAR points, coloured "
                     "from the images, besides the SfM points, so Gaussians start on the true surfaces even where "
                     "the images lack texture. Re-exports the dataset before training (seconds).")
            self.g_splat_depth = gui.add_number(
                "LiDAR depth weight", initial_value=splat.DEFAULT_LIDAR_DEPTH, min=0.0, max=10.0, step=0.05,
                hint="ROS 2 bag projects: weight of the LiDAR depth loss (splatfacto-lidar). Every training step "
                     "projects the LiDAR points into the training image and penalizes the relative difference "
                     "between the splat's rendered depth and the LiDAR depth, so Gaussians stay on the measured "
                     "surfaces, also where the images lack texture. 0 = off (plain splatfacto). The depth error "
                     "is reported in the log and History.")
            self.g_splat_live = gui.add_checkbox(
                "Live preview", initial_value=True,
                hint="Show the splat in the View tab while it trains, refreshed at every checkpoint (base colours, "
                     "at most 500k Gaussians). The finished splat replaces it at the end.")
            self.g_splat_every = gui.add_number(
                "Preview every (steps)", initial_value=500, min=100, step=100,
                hint="Checkpoint interval while previewing (nerfstudio's default is 2000). Each checkpoint is "
                     "written to disk (~0.75 GB per million Gaussians), so very small values slow training down.")
            self.g_splat_status = gui.add_markdown("")
            b = gui.add_button("Train + export splat", icon=viser.Icon.SPARKLES, disabled=why is not None,
                               hint="ns-train on transforms.json + sparse_pc.ply, then ns-export and conversion "
                                    "to splat/splat_enu.ply in the project's ENU frame (aligned with the "
                                    "cameras and the mesh).")
            b.on_click(lambda _: self.run_stage("splat"))
            self.g_buttons["splat"] = b
            self.g_splat_pause = gui.add_button(
                "Pause training", icon=viser.Icon.PLAYER_PAUSE, disabled=True,
                hint="Save a checkpoint at the current step and end training without exporting (frees the GPU "
                     "at once; the live preview stays). Resume continues later from that step. Stop (top) also "
                     "keeps the checkpoint but exports the partial splat.")
            self.g_splat_pause.on_click(lambda _: self.pause_splat())
            b = gui.add_button("Resume training", icon=viser.Icon.PLAYER_PLAY, disabled=True,
                               hint=STAGE_HINTS["splat-resume"] + " Available after Pause or Stop, while "
                                    "transforms.json is unchanged.")
            b.on_click(lambda _: self.run_stage("splat-resume"))
            self.g_buttons["splat-resume"] = b
            gui.add_button("Reload dense / splat", hint="Re-scan the project for fused points, meshes and "
                           "splat/splat_enu.ply, e.g. after scripts/splat finished.").on_click(
                lambda _: self._refresh_dense())

        with tabs.add_tab("Splat mesh", viser.Icon.CUBE):
            gui.add_markdown(
                "An alternative to the Dense tab's COLMAP mesh: a mesh **from the finished splat**. Depth and "
                "colour are rendered from the splat at every camera, fused into a truncated signed distance "
                "field (TSDF) and meshed with marching cubes. Needs a splat (Splat tab) and the GPU; takes "
                "seconds to a minute. The mesh has vertex colours and lies in the same frame as the cameras, "
                "the splat and the COLMAP mesh.")
            self.g_sm_quality = gui.add_dropdown(
                "Quality", options=tuple(splat_mesh.QUALITY), initial_value="medium",
                hint="Voxel size as pixels at the median rendered depth: coarse 8, medium 4, fine 2 (at the render "
                     "width). Finer = more detail, more faces and memory; on the test bag fine gave 3x the faces "
                     "for about the same accuracy.")
            self.g_sm_width = gui.add_number(
                "Render width (px)", initial_value=960, min=160, max=4096, step=160,
                hint="Width of the depth images rendered from the splat (never larger than the images). Wider = "
                     "finer automatic voxels and slower fusion.")
            self.g_sm_depth = gui.add_number(
                "Max depth (m, 0 = auto)", initial_value=0.0, min=0.0, step=5.0,
                hint="Depth beyond this is not fused, which keeps the far background and sky floaters out of the "
                     "mesh. 0 = 3 x the median rendered depth.")
            b = gui.add_button("Build mesh from splat", icon=viser.Icon.CUBE,
                               disabled=splat_mesh.available() is not None,
                               hint="Render depth from splat/splat_enu.ply at every camera, fuse (TSDF), mesh: "
                                    "splatmesh/mesh.ply + a simplified preview for the View tab (Mesh: splat).")
            b.on_click(lambda _: self.run_stage("splatmesh"))
            self.g_buttons["splatmesh"] = b

        with tabs.add_tab("Dense", viser.Icon.GRID_DOTS):
            gui.add_markdown("COLMAP MVS. PatchMatch stereo: " + (
                "CUDA available" if dense.has_cuda() else
                "**needs CUDA** - run the GUI in the CUDA container (`MODELFORGE_GPU=1 scripts/dev`)"))
            self.g_dense_in_all = gui.add_checkbox(
                "Include in Run full pipeline", initial_value=False,
                hint="Off: the full pipeline's dense result is the mesh from the splat (Splat mesh tab). On: COLMAP's "
                     "MVS (undistort → stereo → fusion → mesh → texture) runs after it as well.")
            self.g_dense_quality = gui.add_dropdown(
                "Quality", options=tuple(DENSE_QUALITY), initial_value=DENSE_QUALITY_DEFAULT,
                hint="Speed vs detail (image size for undistort, PatchMatch effort for stereo; set when those "
                     "stages run). superfast: 320 px, fast PatchMatch (default, the quickest, coarse mesh). low: "
                     "640 px, fast PatchMatch. medium: 1200 px. high: full resolution. max: COLMAP defaults at full "
                     "resolution (hours for 300 images).")
            self.g_dense_lidar = gui.add_checkbox(
                "Use LiDAR points", initial_value=True,
                hint="ROS 2 bag projects: merge the LiDAR cloud into the fused points before meshing (each point "
                     "with the cameras that see it, occlusion-checked against the depth maps), so surfaces without "
                     "texture still get geometry. The image-only cloud stays as dense/fused_mvs.ply.")
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

        with self.left.add_tab("Settings", viser.Icon.SETTINGS):
            self.g_dark = gui.add_checkbox("Dark mode", initial_value=self.dark,
                                           hint="Dark or light theme for the whole GUI (panels, 3D view and plots), "
                                                "for every open browser. Remembered across restarts.")
            self.g_dark.on_update(lambda _: self.set_dark(self.g_dark.value))
        with gui.add_folder("History", expand_by_default=False):
            self.g_history = gui.add_markdown("_nothing run in this session_")
        # bottom windows: the log and the hardware monitor, floating at the bottom of the 3D view (viser
        # docks panels only to the left / right edges or above / below another panel, so "bottom" is a float)
        # quick bar: the View tab's visibility toggles + orbit, floating at the top of the 3D view
        self.quick = gui.add_panel()
        self._quick_tab = self.quick.add_tab("Quick view", viser.Icon.EYE)
        self._quick_group, self._quick_labels = None, None
        self.log_panel = gui.add_panel()
        with self.log_panel.add_tab("Log", viser.Icon.TERMINAL_2):
            gui.add_markdown(f"_last {LOG_LINES} lines of the running job, newest first_")
            self.g_log = gui.add_markdown("")
        # the hardware monitor is its own window right of the log: a floating panel cannot be split
        # into columns (viser splits only docked panels), so two windows sit side by side
        self.hw_panel = gui.add_panel()
        with self.hw_panel.add_tab("Hardware", viser.Icon.CPU):
            self.hardware = HardwarePanel(self, gui, plot_aspect=0.3)  # the bottom window is low
        self._place_panels()

    def _place_panels(self):
        self.left.dock_left()
        self.left.set_width(420)
        self.server.gui.main_panel.dock_right()
        self.log_panel.float(x=8, y=-8, width=760, height=260)  # negative y: measured from the bottom
        self.log_panel.minimize()  # folded by default; one click opens it
        self.hw_panel.float(x=776, y=-8, width=480, height=260)
        self.hw_panel.minimize()
        # viser gives every button of a group the same width (the longest label: ~72 px for "· Thumbs") and wraps
        # the rest onto further rows, stretching a short last row; 12 buttons at 6 per row give an even 2 x 6 grid
        # (the user found one row of 13 too wide, ~1250 px)
        per_row = (len(self._quick_items()) + 1) // 2
        self.quick.float(x=8, y=8, width=24 + 72 * per_row, height=136)

    # ------------------------------------------------------------- settings
    def _settings_path(self):
        return os.path.join(self.root, ".modelforge_gui.json")

    def _load_settings(self):
        try:
            with open(self._settings_path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_setting(self, key, value):
        settings = {**self._load_settings(), key: value}  # read before "w" empties the file
        try:
            with open(self._settings_path(), "w") as f:
                json.dump(settings, f)
        except OSError:
            pass

    def _remember_project(self):
        """The project to open at the next start (main: last_project), relative to the root."""
        rel = os.path.relpath(os.path.abspath(self.path), self.root)
        if not rel.startswith(".."):
            self._save_setting("last_project", rel)

    def set_dark(self, dark):
        """Switch the theme (all clients) and remember the choice (the Hardware plot follows at its next update)."""
        self.dark = bool(dark)
        self.server.gui.configure_theme(control_width="large", dark_mode=self.dark)
        self._save_setting("dark_mode", self.dark)

    def plot_template(self):
        return "plotly_dark" if self.dark else "plotly_white"

    def _kind(self):
        return ingest.load_kind(self.path)

    def _sparse_kind(self):
        """pipeline.sparse_kind: a ROS 2 bag of images only reconstructs like a video."""
        return pipeline.sparse_kind(self.path)

    def _apply_kind(self):
        """Sparse tab: refine + BA controls for DJI projects, reconstruct for videos without metadata."""
        video = self._sparse_kind() == "video"
        for name, show in (("refine", not video), ("reconstruct", video)):
            self.g_buttons[name].visible = show
        self.g_ba_iters.visible = self.g_skip.visible = self.g_skip_round.visible = not video
        for g in (self.g_ba_mode, self.g_chunk_size, self.g_chunk_overlap, self.g_polish):
            g.visible = not video
        self.g_mapper.visible = video
        self.g_dense_lidar.visible = self.g_splat_lidar.visible = self.g_splat_depth.visible = \
            self._kind() == "rosbag"
        self.g_upright.value = self._kind() == "dji"  # the kind's default: only a gimbal keeps the horizon level
        self.g_sparse_md.content = (
            ("Incremental SfM from the images alone (" + ("ROS 2 bag without camera poses" if self._kind() == "rosbag"
                                                          else "video without metadata")
             + "): features → pairs → match → reconstruct → export. Arbitrary frame and scale, oriented up.")
            if video else
            "Pose refinement from the bag's TF / odometry priors (camera_info intrinsics fixed): features → pairs → "
            "match → refine → export; SfM points are checked against the LiDAR." if self._kind() == "rosbag" else
            "Pose refinement from the DJI priors: features → pairs → match → refine → export (nerfstudio dataset).")
        if video and self.g_color.value in ("position deviation", "rotation deviation"):
            self.g_color.value = "time"

    def _header(self):
        kind = ingest.KINDS[self._kind()].split(" (")[0]
        if self.project is None:
            return f"### modelforge\n`{self.path}` · {kind} · no images yet"
        n_src = len(set(self.project.source.tolist()))
        return (f"### modelforge\n`{self.path}` · {kind} · {len(self.project.names)} images"
                + (f" from {n_src} sources" if n_src > 1 else ""))

    def load_project(self):
        """(Re)load the open project after extraction: new images, no snapshots yet."""
        with self.lock:
            self._clear_scene()
            self.project = Project(self.path)
            self._show_project()
        self.projects.refresh()

    def refresh_after_images_changed(self):
        """Another image set (Images used): the old run's live preview and History no longer apply."""
        self._splat_live_on = False
        self.server.scene.remove_by_name("/dense/splat_live")
        self._refresh_dense()

    def open_project(self, path):
        """Switch the whole GUI to another (new or existing) project folder."""
        if self.running or self.ingest.running:
            raise RuntimeError("a job is running: wait for it to finish before switching projects")
        with self.lock:
            self.path = path
            self._clear_scene()
            self._splat_live_on = False  # a paused run's live preview belongs to the old project
            self.server.scene.remove_by_name("/dense/splat_live")
            self.project = Project(path) if os.path.exists(os.path.join(path, "poses.json")) else None
            self._show_project()
            self._history_lines, self._log_lines = [], []
            self.g_history.content, self.g_log.content = "_nothing run in this session_", ""
            self.g_state.content = "⚪ **idle**"
            self.g_inspect_md.content = "Click a camera to inspect it."
        self.ingest.set_project(path)
        self.projects.refresh()
        self._remember_project()
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
        self._drawn = {}
        self._dense_lines = {}

    def _set_aspect(self):
        """Frustum aspect from the project's images (portrait phone videos are taller than wide)."""
        if self.project is not None and self.project.names:
            from PIL import Image
            try:
                with Image.open(os.path.join(self.path, "images", self.project.names[0])) as im:
                    self.aspect = im.width / im.height
            except OSError:
                pass

    def _show_project(self):
        self.g_header.content = self._header()
        self.subset.refresh()
        self._apply_kind()
        self._set_aspect()
        if self.project is None:
            self.g_snapshot.options = ("priors",)
            self.g_stats.content = self.g_legend.content = ""
        else:
            names = tuple(self.project.snapshot_names())
            self.g_snapshot.options = names
            self.load_snapshot("refined" if "refined" in names else names[-1])
        self._refresh_dense()
        self._auto_show_sparse()

    def _auto_show_sparse(self, max_cameras=2000):
        """Points and Cameras start off (thousands of frustums hide a splat); a project with nothing else to
        show (no splat, mesh or dense points, e.g. right after SfM) gets them on, cameras up to max_cameras."""
        d = self.dense
        if self.project is None or d["splat"] is not None or d["meshes"] or d["fused"] is not None:
            return
        self.g_points.value = True
        if len(self.project.names) <= max_cameras:
            self.g_cams.value = True

    def _on_connect(self, client):
        self.recenter(client)

    def _view_frame(self):
        """(centre, extent) of what there is to see: the cameras (or the Ingest flight plan) and the sparse
        points, over their 2-98 % range so stray far points do not count; None without anything."""
        if self.project is None:
            prev = self.ingest.prev
            if not prev:
                return None
            pts = np.concatenate([x["track"] for x in prev["sources"]])
        else:
            c = self.snap.cams if self.snap else self.project.prior_cams
            pts = c.centers[c.valid]
            if self.snap is not None and len(self.snap.points):
                p = self.snap.points
                p = p[np.random.default_rng(0).choice(len(p), min(len(p), 20000), replace=False)]
                pts = np.concatenate([pts, p]) if len(pts) else p
        pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
        if not len(pts):
            return None
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        return (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 1e-3)

    def glide_to(self, client, position, look_at, seconds=1.2, rate_hz=30.0):
        """Move a browser's camera smoothly (smoothstep) to position, looking at look_at, z up. Stops its
        orbit / flight; returns at once (the glide runs in a thread)."""
        if client is None:
            return
        self._stop_fly(client)
        stop = self._orbits.pop(client.client_id, None)
        if stop is not None:
            stop.set()
            self.g_orbit.label = "Stop orbit" if self._orbits else "Start orbit"
        p1, l1 = np.asarray(position, float), np.asarray(look_at, float)

        def run():
            try:
                p0, l0 = np.array(client.camera.position, float), np.array(client.camera.look_at, float)
                n = max(1, int(seconds * rate_hz))
                for k in range(1, n + 1):
                    a = k / n
                    a = a * a * (3 - 2 * a)
                    with client.atomic():
                        client.camera.up_direction = (0.0, 0.0, 1.0)
                        client.camera.position = tuple(p0 + (p1 - p0) * a)
                        client.camera.look_at = tuple(l0 + (l1 - l0) * a)
                    time.sleep(1.0 / rate_hz)
            except Exception:  # the browser went away
                pass
        threading.Thread(target=run, daemon=True).start()

    def recenter(self, client=None):
        """Center button (and on connect / project switch): look at the scene from the south, above it,
        far enough to see all of it. client None = every browser. Stops that browser's orbit / flight."""
        frame = self._view_frame()
        if frame is None:
            return
        mid, extent = frame
        clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in clients:
            self._stop_fly(c)
            stop = self._orbits.pop(c.client_id, None)
            if stop is not None:
                stop.set()
            try:
                with c.atomic():
                    c.camera.up_direction = (0.0, 0.0, 1.0)
                    c.camera.position = tuple(mid + np.array([0.0, -0.6, 0.5]) * extent)
                    c.camera.look_at = tuple(mid)
            except Exception:  # the browser went away
                pass
        self.g_orbit.label = "Stop orbit" if self._orbits else "Start orbit"

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
            if not self.project.has_priors:  # no metadata: the reconstructed cameras bound the scene
                c = s.cams.centers[s.cams.valid] if s.cams.valid.any() else np.zeros((1, 3))
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
                if not pc.valid[i]:  # video without metadata: no prior pose to show
                    continue
                ph = self.server.scene.add_camera_frustum(
                    f"/priors/{i}", fov=self.fov, aspect=self.aspect, scale=scale * 0.8, color=PRIOR_COLOR,
                    wxyz=pc.wxyz(i), position=pc.centers[i], visible=bool(self.g_priors.value))
                self.prior_handles[i] = ph

    def _draw_lines(self, cams=None):
        if cams is None and self.snap is None:  # a toggle while the project is still loading
            return
        cams = cams or self.snap.cams
        pc = self.project.prior_cams
        v = cams.valid & pc.valid  # no priors (video without metadata): no correction lines
        if self.g_corr.value and self.snap.name != "priors" and v.any():
            seg = np.stack([pc.centers[v], cams.centers[v]], axis=1).astype(np.float32)
            self.server.scene.add_line_segments("/corrections", seg, colors=(255, 80, 80), thickness=2.0,
                                                thickness_units="screen")
        else:
            self.server.scene.add_line_segments("/corrections", np.zeros((0, 2, 3), np.float32), colors=(0, 0, 0))
        if self.g_traj.value:
            p, pv = pc.centers, pc.valid[:-1] & pc.valid[1:]
            self.server.scene.add_line_segments("/traj/prior", np.stack([p[:-1][pv], p[1:][pv]], 1).astype(np.float32),
                                                colors=PRIOR_COLOR, thickness=1.5, thickness_units="screen")
            c, cv = cams.centers, cams.valid
            ok = cv[:-1] & cv[1:]
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
        if s.name != "priors" and v.any() and self.project.has_priors:
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
            for k, d in pm.get("drift_m", {}).items():
                if d["max"] >= 0.05:
                    lines.append(f"source {int(k) + 1} position drift: up to {d['max']:.2f} m "
                                 f"(height range {d['range_z']:.2f} m, {d['knots']} knots)")
        self.g_stats.content = "  \n".join(lines)

    def _prior_model(self, snapshot):
        """Calibrated prior model after the given snapshot, from report.json (None if unknown)."""
        refine_rep = self.project.report().get("refine", {})
        if snapshot == "refined":
            return refine_rep.get("prior_model")
        rnd = refine_rep.get(snapshot, {}) if snapshot.startswith(("round_", "calib_")) and not snapshot.endswith("_tri") else {}
        pm = rnd.get("prior_model", {})
        return None if pm.get("skipped") else pm.get("model")

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
        if self.snap.cams.valid[i] and self.snap.name != "priors" and self.project.has_priors:
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
        if self.snap is None:  # nothing drawn yet (e.g. the kind switched the colouring at startup)
            return
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
        self.semantic.unload(stage)  # reconstructions get the memory; the text model loads again on demand
        self._token = jobs.Token()
        self.g_stop.disabled = False
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
            sfm = pipeline.sfm_stages(self._sparse_kind())
            if stage == "all":
                seq, skipped = list(sfm), []
                if self.subset.keyframes_selected():  # Images used = keyframes: choose them first
                    seq = ["analyze", "keyframes"] + seq
                if splat.available() is None:
                    seq += ["splat", "splatmesh"]  # the dense result is the mesh from the splat
                    if self.semantic.g_in_all.value:  # Search tab: a searchable splat at the end
                        if semantic.available() is None:
                            seq.append("semantic")
                        else:
                            skipped.append(f"semantic features ({semantic.available()})")
                else:
                    skipped.append(f"splat and splat mesh ({splat.available()})")
                if self.g_dense_in_all.value:  # COLMAP's MVS only when asked (Dense tab)
                    if dense.has_cuda():
                        seq += list(pipeline.DENSE_STAGES)
                    else:
                        skipped.append("COLMAP dense (no CUDA)")
                for sk in skipped:
                    self._history(f"⚪ **skipped** {sk}")
            elif stage == "sfm" or stage in SFM_BUTTONS:
                seq = sfm if stage == "sfm" else (stage,)
            elif stage in ("splat", "splatmesh", "splat-resume", "analyze", "semantic"):
                seq = (stage,)
            else:
                seq = pipeline.DENSE_STAGES if stage == "dense" else (stage,)
            m = self.g_mesh_method.value
            methods = dense.MESH_METHODS if m == "both" else (m,)
            for k, st in enumerate(seq):
                self._token.check()
                self._begin_stage(st, k, len(seq))
                ts = time.time()
                if st == "features":
                    pipeline.stage_features(project, report, self._sparse_kind(), self.g_extractor.value,
                                            int(self.g_max_features.value), bool(self.g_upright.value),
                                            progress=self._on_features_progress)
                elif st == "reconstruct":
                    pipeline.stage_reconstruct(project, report, progress=self._on_reconstruct_event,
                                               token=self._token, mapper=self.g_mapper.value)
                elif st == "pairs":
                    pipeline.stage_pairs(project, report, pri, self.project.meta, self.g_pairs.value,
                                         int(self.g_neighbors.value))
                elif st == "match":
                    pipeline.stage_match(project, report, float(self.g_ratio.value), bool(self.g_guided.value),
                                         int(self.g_min_inliers.value), token=self._token,
                                         progress=lambda k, n: self._on_count_progress("match", "pair", k, n))
                elif st == "refine":
                    self._skip, self._skip_round = threading.Event(), threading.Event()
                    self.g_skip.disabled = self.g_skip_round.disabled = False
                    try:
                        pipeline.stage_refine(project, report, pri, progress=self._on_refine_event, token=self._token,
                                              max_iterations=int(self.g_ba_iters.value), skip=self._skip,
                                              skip_round=self._skip_round,
                                              ba_mode=self.g_ba_mode.value,
                                              chunks=chunked.Chunked(size=int(self.g_chunk_size.value),
                                                                     overlap=int(self.g_chunk_overlap.value),
                                                                     polish_sweeps=int(self.g_polish.value)))
                    finally:
                        self.g_skip.disabled = self.g_skip_round.disabled = True
                elif st == "export":
                    pipeline.stage_export(project, report)
                elif st == "analyze":
                    from modelforge.sfm import overlap
                    if stage == "all" and overlap.is_current(project):  # the full pipeline reuses a current one
                        self._history("⚪ **analyze** · the overlap analysis of these images exists: reused")
                        continue
                    overlap.analyze(project, kind=self._sparse_kind(), token=self._token, log=self._on_log,
                                    progress=self._on_analysis_progress)
                    self.subset.refresh()
                elif st == "semantic":
                    semantic.run(project, on_line=self._on_log, job=self._token,
                                 on_progress=lambda ph, k, n: self._on_count_progress(
                                     "semantic", "PCA sample: photo" if ph == "pca" else "photo", k, n))
                    self.semantic.refresh()
                elif st == "keyframes":  # the full pipeline with Images used = keyframes
                    n, total = self.subset.apply_keyframes()
                    report, pri = pipeline.Report(project), self.project.priors  # the new image set
                    self._history(f"✅ **keyframes** ({self.subset.g_level.value}) · {n} of {total} images")
                    continue
                elif st in ("splat", "splat-resume"):
                    self._run_splat(project, resume=st == "splat-resume")
                    report = pipeline.Report(project)  # splat.run wrote its stats
                    self._refresh_dense()  # show the finished splat now, not only when the whole run ends
                elif st == "splatmesh":
                    self._run_splat_mesh(project)
                    report = pipeline.Report(project)
                    self._refresh_dense()
                    if "splat" in self.g_mesh.options:
                        self.g_mesh.value = "splat"  # show the new mesh in the View tab
                else:
                    pipeline.stage_dense(project, report, st, methods, DENSE_QUALITY[self.g_dense_quality.value],
                                         token=self._token, on_progress=self._on_stereo_progress,
                                         lidar=bool(self.g_dense_lidar.value))
                    if st in ("fusion", "mesh", "texture"):
                        self._refresh_dense()  # points / mesh appear as soon as they exist
                self._history(f"✅ **{st}** · {time.time() - ts:.0f} s · {self._stage_summary(st, report.data)}")
            if stage in ("dense", "splat", "splatmesh", "all") or stage in pipeline.DENSE_STAGES:
                self._refresh_dense()
            self.refresh_snapshot_list(select="refined" if stage in ("refine", "reconstruct", "sfm", "all") else None)
            if stage in ("refine", "reconstruct", "sfm"):
                self._auto_show_sparse()  # nothing else to look at yet
            self._refresh_splat_plan()  # new cameras: new suggestion
            final = (f"✅ **{stage} finished** in {_fmt_s(time.time() - t0)} · {time.strftime('%H:%M:%S')}  \n"
                     f"{NEXT_STEP.get(stage, '')}")
        except splat.Paused as e:
            self._history(f"⏸ **splat** {e}")
            final = (f"⏸ **splat {e}** · {time.strftime('%H:%M:%S')}  \n"
                     "Splat tab → **Resume training** continues from there")
        except splat.StoppedPartial as e:
            self._refresh_dense()  # the partial splat replaces the live preview
            self._history(f"⏹ **splat** {e}")
            final = (f"⏹ **splat {e}** · {time.strftime('%H:%M:%S')}  \n"
                     "View tab → Dense / splat → **Splat**; Export tab")
        except jobs.Cancelled:
            label = self._job["label"]
            self._history(f"⏹ **{label}** stopped")
            final = (f"⏹ **stopped** during {label} after {_fmt_s(time.time() - t0)} · {time.strftime('%H:%M:%S')}"
                     "  \nthat stage's outputs may be incomplete: run it again")
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
                           "splat": splat.available() is not None,
                           "splat-resume": splat.available() is not None or splat.resume_info(project) is None,
                           "splatmesh": splat_mesh.available() is not None}
            for name, b in self.g_buttons.items():
                b.disabled = unavailable.get(name, False)
            self.g_progress.visible = False
            self.g_stop.disabled = not self.ingest.running
            self.g_splat_pause.disabled = True
            self._sync_resume()

    def pause_splat(self):
        """Pause button: like Stop (the trainer saves a checkpoint and ends), but splat.run skips the
        export and leaves the live preview; Resume continues."""
        if self.running and getattr(self, "_splat_pause_event", None) is not None:
            self._splat_pause_event.set()
            self.g_splat_pause.disabled = True
            self.stop()

    def _sync_resume(self):
        """Resume button and status line from splat/resume.json (after a pause / stop)."""
        info = splat.resume_info(self.path) if self.path else None
        b = self.g_buttons.get("splat-resume")
        if b is not None and not self.running:
            b.disabled = info is None or splat.available() is not None
        if info and not self.running:
            self.g_splat_status.content = (
                f"⏸ {info['method']} stopped at step {info['step']:,} of "
                f"{info['iterations'] or splat.DEFAULT_ITERATIONS:,}"
                " · **Resume training** continues with its settings (×" + str(info["downscale"]) + ")")

    def stop(self):
        """Stop button: cancel the running job and / or an extraction (in a thread: killing a
        subprocess may wait a few seconds)."""
        tokens = [t for t in (self._token if self.running else None,
                              self.ingest.token if self.ingest.running else None) if t is not None]
        if not tokens:
            return
        self.g_stop.disabled = True
        self._on_log("--- stop requested")
        for t in tokens:
            threading.Thread(target=t.cancel, daemon=True).start()

    # ------------------------------------------------------------ quick bar
    def _quick_items(self):
        """(name, available, on) for the quick bar, mirroring the View tab."""
        d = getattr(self, "dense", None) or {"splat": None, "meshes": {}, "fused": None}
        return (("Splat", d["splat"] is not None or getattr(self, "_splat_live_on", False), self.g_splat.value),
                ("Mesh", bool(d["meshes"]), self.g_mesh.value != "off"),
                ("Dense", d["fused"] is not None, self.g_dense_pts.value),
                ("Points", self.project is not None, self.g_points.value),
                ("Cameras", self.project is not None, self.g_cams.value),
                ("Thumbs", self.project is not None and self.g_cams.value, self.g_thumbs.value),
                ("Residuals", self.project is not None, self.g_corr.value),  # View: Prior → refined
                ("Trajectory", self.project is not None, self.g_traj.value),  # View: camera / prior paths
                ("Plan", bool(self.ingest.prev), self.ingest.g_show.value),
                ("Orbit", True, bool(self._orbits)),
                ("Center", True, False),
                ("Fly", self.project is not None, bool(self._flights)))

    def _sync_quickbar(self):
        """Rebuild the quick bar's row when a state changes (button groups cannot relabel in place):
        ● on, ○ off, · nothing to show yet."""
        labels = tuple(f"⌖ {name}" if name == "Center" else
                       f"{'●' if on and avail else '○' if avail else '·'} {QUICK_SHORT.get(name, name)}"
                       for name, avail, on in self._quick_items())
        if labels == self._quick_labels:
            return
        self._quick_labels = labels
        if self._quick_group is not None:
            self._quick_group.remove()
        with self._quick_tab:
            g = self.server.gui.add_button_group(
                "", options=labels, order=1.0,
                hint="Splat, Mesh, Dense (fused points), Points (sparse), Cams (cameras), Thumbs (their "
                     "thumbnails, with the cameras on), Resid (red prior → refined segments), Traj (trajectories), "
                     "Plan (the Ingest flight plan); Orbit, Center (look at the whole scene), Fly (the camera "
                     "path). ● on, ○ off, · nothing yet. Same as the View tab.")
        long = {v: k for k, v in QUICK_SHORT.items()}

        def click(e):
            short = e.target.value.split(" ", 1)[1]
            self._quick_click(long.get(short, short), e.client)
        g.on_click(click)
        self._quick_group = g

    def _quick_click(self, name, client):
        avail = {n: a for n, a, _ in self._quick_items()}
        if not avail.get(name):
            return
        if name == "Splat":
            self.g_splat.value = not self.g_splat.value
        elif name == "Mesh":
            meshes = tuple(self.dense["meshes"])
            if self.g_mesh.value != "off":
                self._last_mesh, self.g_mesh.value = self.g_mesh.value, "off"
            else:
                last = getattr(self, "_last_mesh", None)
                self.g_mesh.value = last if last in meshes else meshes[0]
        elif name == "Points":
            self.g_points.value = not self.g_points.value
        elif name == "Cameras":
            self.g_cams.value = not self.g_cams.value
        elif name == "Thumbs":
            self.g_thumbs.value = not self.g_thumbs.value
        elif name == "Residuals":
            self.g_corr.value = not self.g_corr.value
        elif name == "Trajectory":
            self.g_traj.value = not self.g_traj.value
        elif name == "Dense":
            self.g_dense_pts.value = not self.g_dense_pts.value
        elif name == "Plan":
            self._set_plan_visible(not self.ingest.g_show.value)
        elif name == "Orbit":
            self.toggle_orbit(client)
        elif name == "Fly":
            self.toggle_fly(client)
        elif name == "Center":
            self.recenter(client)
        self._sync_quickbar()

    def _set_plan_visible(self, on):
        """The flight plan layer (drawn by the Ingest tab): keep its two checkboxes in step and redraw."""
        for box in (self.ingest.g_show, self.g_plan):
            if box.value != on:
                box.value = on  # its on_update calls back here with the same value: no-op
        self.ingest._draw_scene()
        self._sync_quickbar()

    def toggle_orbit(self, client):
        """Start / stop orbiting this client's camera around its current look-at point (about +z)."""
        if client is None:
            return
        stop = self._orbits.pop(client.client_id, None)
        if stop is not None:
            stop.set()
        else:
            self._stop_fly(client)  # one camera driver per browser
            stop = threading.Event()
            self._orbits[client.client_id] = stop
            threading.Thread(target=self._orbit, args=(client, stop), daemon=True).start()
        self.g_orbit.label = "Stop orbit" if self._orbits else "Start orbit"
        self._sync_quickbar()

    def _orbit(self, client, stop, rate_hz=30.0, hold_s=0.6):
        """Each tick rotates the browser's *current* camera about its look-at point (+z axis), so mouse
        zoom / tilt / pan stick and the orbit continues from there. While the user drags (the camera
        differs from every pose sent recently) rotation pauses, resuming hold_s after the last change."""
        cam = client.camera
        sent = []  # recent (position, look_at) we set: the browser may still echo them back
        hold_until, t_prev = 0.0, time.time()
        try:
            while not stop.wait(1.0 / rate_hz):  # no step after Stop
                now = time.time()
                dt, t_prev = now - t_prev, now
                pos, center = np.array(cam.position, float), np.array(cam.look_at, float)
                eps = 1e-3 * max(np.linalg.norm(pos - center), 1e-3)
                ours = any(np.linalg.norm(pos - p) < eps and np.linalg.norm(center - c) < eps for p, c in sent)
                if sent and not ours:  # moved by the mouse
                    hold_until = now + hold_s
                    sent.clear()
                if now < hold_until:
                    continue
                if ours:  # an echo, often of an older step: continue from the latest pose sent, never go back
                    pos, center = sent[-1]
                else:  # adopting the browser's camera (start, or after a drag): its late echoes are ours too
                    sent.append((pos, center))
                a = np.radians(float(self.g_orbit_speed.value)) * dt
                c, s = np.cos(a), np.sin(a)
                off = pos - center
                new = center + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]])
                with client.atomic():
                    cam.position = new
                    cam.look_at = center
                sent = (sent + [(new, center)])[-15:]
        except Exception:  # the browser went away
            pass
        finally:
            if self._orbits.get(client.client_id) is stop:
                self._orbits.pop(client.client_id, None)
                self.g_orbit.label = "Stop orbit" if self._orbits else "Start orbit"
                self._sync_quickbar()

    # ------------------------------------------------------------ fly the camera path
    def _flight_path(self):
        cams = self.snap.cams if self.snap is not None else self.project.prior_cams
        return gdata.FlightPath(cams, self.project.time)

    def _look_distance(self, path):
        """How far ahead the look-at point sits (the pivot for the mouse after a flight): the median depth
        of the sparse points in front of the cameras, else a tenth of the path's extent."""
        pts = self.snap.points if self.snap is not None else np.zeros((0, 3))
        if len(pts):
            pts = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 2000), replace=False)]
            ks = np.linspace(0, len(path.index) - 1, min(len(path.index), 30)).astype(int)
            depths = []
            for k in ks:
                _, R, _ = path.pose(path.t[k])
                z = (pts - path.centers[k]) @ R[:, 2]
                if (z > 0).any():
                    depths.append(np.median(z[z > 0]))
            if depths:
                return float(np.median(depths))
        return max(1.0, 0.1 * float(np.ptp(path.centers, axis=0).max()))

    def _fly_set(self, client, path, t, dist):
        """Put the browser's camera on the path at time t: the camera's position, its view direction (look-at
        dist ahead) and its up (-y of the OpenCV camera, so roll is reproduced). Returns (position, image)."""
        pos, R, i = path.pose(t)
        with client.atomic():
            client.camera.position = pos
            client.camera.look_at = pos + R[:, 2] * dist
            client.camera.up_direction = -R[:, 1]
        return pos, i

    def toggle_fly(self, client):
        if client is None:
            return
        if self._stop_fly(client):
            return
        stop = self._orbits.pop(client.client_id, None)  # one camera driver per browser
        if stop is not None:
            stop.set()
            self.g_orbit.label = "Stop orbit" if self._orbits else "Start orbit"
        try:
            path = self._flight_path()
        except (ValueError, AttributeError) as e:
            self.g_fly_md.content = f"⚪ {e}"
            return
        state = {"stop": threading.Event(), "jump": None}
        self._flights[client.client_id] = state
        threading.Thread(target=self._fly, args=(client, state, path), daemon=True).start()
        self.g_fly.label = "Stop flying"
        self._sync_quickbar()

    def _stop_fly(self, client):
        state = self._flights.pop(client.client_id, None) if client is not None else None
        if state is None:
            return False
        state["stop"].set()
        return True

    def _fly_scrub(self, client):
        """The position slider moved: jump there (a flying browser at its next tick, else at once)."""
        if self._fly_guard or client is None or self.project is None:
            return
        state = self._flights.get(client.client_id)
        try:
            path = self._flight_path()
        except (ValueError, AttributeError):
            return
        t = path.duration * float(self.g_fly_pos.value) / 100.0
        if state is not None:
            state["jump"] = t
            return
        try:
            _, i = self._fly_set(client, path, t, self._look_distance(path))
            self.g_fly_md.content = f"at {t:.1f} / {path.duration:.1f} m · `{self.project.names[i]}`"
        except Exception:  # the browser went away
            pass

    def _fly(self, client, state, path, rate_hz=30.0, hold_s=1.0):
        """Advance along the path at the set speed (m/s). A camera that is none of the last poses sent (the
        browser echoes them late) means the user is dragging: the flight pauses until hold_s after that."""
        cam, stop = client.camera, state["stop"]
        saved_fov, saved_up, set_fov = None, None, False
        try:
            saved_fov, saved_up = float(cam.fov), np.array(cam.up_direction, float)
            dist = self._look_distance(path)
            t = path.duration * float(self.g_fly_pos.value) / 100.0
            if t >= path.duration - 1e-6:
                t = 0.0
            if self.g_fly_fov.value:
                cam.fov, set_fov = float(self.fov), True
            sent, hold_until = [], 0.0
            t_prev = last_ui = time.time()
            while not stop.is_set():
                now = time.time()
                dt, t_prev = now - t_prev, now
                if state["jump"] is not None:
                    t, state["jump"], sent = state["jump"], None, []
                pos_now = np.array(cam.position, float)
                if sent and not any(np.linalg.norm(pos_now - p) < 1e-3 * max(dist, 1.0) for p in sent):
                    hold_until, sent = now + hold_s, []  # dragged by the mouse
                if now >= hold_until:
                    t += dt * float(self.g_fly_speed.value)
                    if t > path.duration:
                        if not self.g_fly_loop.value:
                            self._fly_set(client, path, path.duration, dist)
                            break
                        t = 0.0
                    pos, i = self._fly_set(client, path, t, dist)
                    sent = (sent + [pos])[-15:]
                    if now - last_ui > 0.25:  # the slider and status, a few times a second
                        last_ui = now
                        self._fly_guard = True
                        try:
                            self.g_fly_pos.value = round(100.0 * t / path.duration, 1)
                        finally:
                            self._fly_guard = False
                        self.g_fly_md.content = (f"flying · {t:.1f} / {path.duration:.1f} m · "
                                                 f"`{self.project.names[i]}`")
                stop.wait(1.0 / rate_hz)
        except Exception:  # the browser went away
            pass
        finally:
            try:
                if set_fov:
                    cam.fov = saved_fov
                if saved_up is not None:
                    cam.up_direction = saved_up
            except Exception:
                pass
            if self._flights.get(client.client_id) is state:
                self._flights.pop(client.client_id, None)
            if not self._flights:
                self.g_fly.label = "Fly the camera path"
                if not stop.is_set():
                    self.g_fly_md.content = "flight finished" + (" (loop is off)" if not self.g_fly_loop.value else "")
            self._sync_quickbar()

    def skip_ba_round(self):
        """Skip-round button: end the running BA where it is; the next round runs as usual."""
        ev = getattr(self, "_skip_round", None)
        if ev is not None and self.running:
            ev.set()
            self._job["detail"] = "ending this bundle adjustment round, then on to the next"
            self._on_log("--- skip of the current BA round requested")

    def skip_refine(self):
        """Skip button: end the running BA where it is and continue after refine."""
        skip = getattr(self, "_skip", None)
        if skip is not None and not skip.is_set():
            skip.set()
            self.g_skip.disabled = True
            self._job["detail"] = "skipping: finishing this round's outputs, then the final triangulation"
            self._on_log("--- refine skip requested")

    def _on_stereo_progress(self, view, views, pass_index):
        passes = ("photometric", "geometric")
        self._job["detail"] = f"PatchMatch {passes[min(pass_index, 1)]} pass: view {view} / {views}"
        self.g_progress.value = 100.0 * (pass_index * views + view) / (2 * views)

    def _run_splat_mesh(self, project):
        def progress(i, n):
            self.g_progress.value = 100.0 * i / n
            self._job["detail"] = f"fusing view {i} / {n}"
        depth = float(self.g_sm_depth.value)
        stats = splat_mesh.run(project, self.g_sm_quality.value, int(self.g_sm_width.value),
                               max_depth=depth if depth > 0 else None, on_line=self._on_log, on_progress=progress,
                               job=self._token)
        return stats

    def _splat_survey(self, project):
        """splat_plan.survey of the project, cached by its camera files' times."""
        key = (project,) + tuple(os.path.getmtime(os.path.join(project, f)) if os.path.exists(os.path.join(project, f))
                                 else 0 for f in ("transforms.json", "poses.json", "sparse_pc.ply"))
        if getattr(self, "_survey", (None, None))[0] != key:
            self._survey = (key, splat_plan.survey(project))
        return self._survey[1]

    def _splat_settings(self, project):
        """Splat tab values with the auto ones filled in from the flight (splat_plan.suggest)."""
        return splat_plan.suggest(self._splat_survey(project), iterations=int(self.g_splat_iters.value),
                                  downscale=self.g_splat_down.value, strategy=self.g_splat_strategy.value,
                                  max_gaussians=int(float(self.g_splat_maxg.value) * 1e6),
                                  camera_opt=self.g_splat_camopt.value, method=self.g_splat_method.value)

    def _refresh_splat_plan(self):
        """The Splat tab's plan line, computed in the background (the survey reads every camera)."""
        project = self.path

        def work():
            try:
                p = self._splat_settings(project)
            except (OSError, ValueError, KeyError) as e:
                text = f"plan: not available yet ({e})"
            else:
                text = ("**Plan** (auto values from the cameras + sparse points): ×" + str(p["downscale"]) + ", "
                        f"{p['iterations']:,} steps, {p['strategy']}"
                        + (f" ≤{p['max_gaussians'] / 1e6:.1f}M" if p["max_gaussians"] else "")
                        + f", camera {p['camera_opt']}  \n" + "  \n".join("· " + n for n in p["notes"]))
            if project == self.path:
                self.g_splat_plan.content = text
        threading.Thread(target=work, daemon=True).start()

    def _run_splat(self, project, resume=False):
        """Train (or, resume=True, continue the paused / stopped run with its own settings)."""
        info = splat.resume_info(project) if resume else None
        if resume and info is None:
            raise RuntimeError("nothing to resume (no checkpoint, or transforms.json changed since)")
        if info:
            iters, down = info["iterations"] or splat.DEFAULT_ITERATIONS, info["downscale"]
            self._on_log(f"splat: resuming {info['method']} from step {info['step']:,} of {iters:,}")
        else:
            plan = self._splat_settings(project)
            iters, down = plan["iterations"], plan["downscale"]
            self._on_log("splat plan: " + "; ".join(plan["notes"]))

        self._live_msg = ""

        def progress(step, pct):
            self.g_progress.value = pct
            self._job["detail"] = f"training step {step:,} / {iters:,} ({pct:.1f} %)" + (
                f"  \n{self._live_msg}" if self._live_msg else "")

        def on_line(m):
            self._on_log(m)
            if m.startswith("splat:"):  # also on the server console, for diagnosing the preview
                print(m, flush=True)

        def prepare(done, total):  # images_<factor>/ before training (thousands of images: minutes)
            self.g_progress.value = 100.0 * done / total
            self._job["detail"] = f"downscaling images ×{down}: {done:,} / {total:,}"
        self._splat_pause_event = pause = threading.Event()
        self.g_splat_pause.disabled = False
        keep_live = False
        try:
            if self._kind() == "rosbag" and not info:  # the initial cloud with or without the LiDAR points
                pipeline.stage_export(project, pipeline.Report(project), lidar=bool(self.g_splat_lidar.value))
            weight = float(self.g_splat_depth.value)
            lidar_depth = weight if self._kind() == "rosbag" and weight > 0 and splat.has_lidar(project) else None
            live = self.g_splat_live.value
            if not live:  # a paused run's preview may still be up
                self._splat_live_on = False
                self.server.scene.remove_by_name("/dense/splat_live")
            if live:  # the live layer stands in for the previous result while training
                self.server.scene.remove_by_name("/dense/splat")
                self._unchanged("splat", None)
                self._splat_live_on = True
                self.g_splat.disabled, self.g_splat.value = False, True  # View tab toggle drives the live layer
                self.g_splat_status.content = (f"live preview: first snapshot at step "
                                               f"{int(self.g_splat_every.value):,}, then in the View tab")
            common = dict(on_line=on_line, on_progress=progress, job=self._token,
                          preview_every=int(self.g_splat_every.value), pause=pause,
                          on_snapshot=self._on_splat_snapshot if live else None, on_prepare=prepare)
            if info:
                splat.resume(project, **common)
            else:
                cache = self.g_splat_cache.value
                if cache == "auto":
                    cache, gb = splat.auto_cache(project, down)
                    self._on_log(f"splat: image cache {cache} ({gb:.1f} GB of decoded photos)")
                splat.run(project, self.g_splat_method.value, iters, cache=cache,
                          downscale=down, lidar_depth=lidar_depth, schedule=plan, **common)
            self.g_splat.value = True  # show the finished splat (drawn by _refresh_dense after the run)
        except splat.Paused:
            keep_live = live  # paused: the last live snapshot stays on screen (the Splat toggle drives it)
            raise
        finally:
            self._splat_pause_event = None
            self.g_splat_pause.disabled = True
            if not keep_live:
                self._splat_live_on = False
                self.server.scene.remove_by_name("/dense/splat_live")

    def _on_splat_snapshot(self, step, npz, transforms):
        """A new training checkpoint: redraw the live splat in the View tab."""
        self._live_src = (npz, transforms)
        step, n, arrays = gdata.load_live_splat(npz, transforms, self._splat_limit(gdata.LIVE_SPLAT_SHOWN))
        self._live_splat = arrays
        self._draw_live_splat()
        msg = f"live splat: step {step:,}, {n:,} gaussians" + (
            f" ({len(arrays[0]):,} shown; View → All gaussians for all)" if len(arrays[0]) < n else "")
        self._dense_status("splat", msg)
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

    def _on_count_progress(self, stage, unit, done, total, name=""):
        """'<unit> k / n' (+ name) on the state line; the bar too when the stage runs alone."""
        self._job["detail"] = f"{unit} {done:,} / {total:,}" + (f" · `{name}`" if name else "")
        if self._job.get("label") == stage:
            self.g_progress.value = 100.0 * done / max(total, 1)

    def _on_analysis_progress(self, phase, done, total, name):
        self._on_count_progress("analyze", "features: image" if phase == "features" else "matching: pair",
                                done, total, name)

    def _on_features_progress(self, done, total, name):
        """Feature extraction: image k / n and the image just finished, on the state line and the bar."""
        self._job["detail"] = f"image {done:,} / {total:,} · `{name}`" if name else f"image {done:,} / {total:,}"
        if self._job.get("label") == "features":  # the stage alone: the bar is this stage's
            self.g_progress.value = 100.0 * done / max(total, 1)

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
                verb = "⏹ **stopping**" if self._token is not None and self._token.cancelled else "⏳ **running**"
                self.g_state.content = (
                    f"{verb} **{j['label']}** · stage {_fmt_s(now - j['stage_t0'])} · "
                    f"total {_fmt_s(now - j['t0'])}" + (f"  \n{j['detail']}" if j["detail"] else ""))
            time.sleep(1.0)

    def _on_log(self, msg):
        lines = getattr(self, "_log_lines", [])
        for line in str(msg).splitlines():
            lines.append(f"{time.strftime('%H:%M:%S')} {line[:240]}" + ("…" if len(line) > 240 else ""))
        self._log_lines = lines[-LOG_LINES:]
        self.g_log.content = "```\n" + "\n".join(reversed(self._log_lines)) + "\n```"  # newest on top

    def _history(self, line):
        h = getattr(self, "_history_lines", [])
        h.append(f"{time.strftime('%H:%M:%S')} {line}")
        self._history_lines = h[-30:]
        self.g_history.content = "  \n".join(reversed(self._history_lines))

    def _stage_summary(self, st, rep):
        r = rep.get(st, {}) if st in SFM_BUTTONS else rep.get("dense", {}).get(st, {})
        try:
            if st == "features":
                return (f"{r['images']} images, {r['keypoints']:,} keypoints"
                        + (f" ({r['extractor']}{', upright' if r.get('upright') else ''})" if "extractor" in r else ""))
            if st == "pairs":
                return f"{r['num_pairs']:,} pairs ({r.get('mode', 'auto')})"
            if st == "match":
                return f"{r['verified_pairs']:,} verified pairs, {r['inlier_matches']:,} inlier matches"
            if st == "refine":
                f = r["final"]
                if f["points3D"] < 100:  # nothing for dense / splat to start from
                    inliers = rep.get("match", {}).get("inlier_matches", 0)
                    if inliers >= 200 * max(f["reg_images"], 1):  # the images match well: the priors are wrong
                        hint = ("for a ROS 2 bag check Camera frame / **Extrinsic RPY** (e.g. yaw 180: the image's "
                                "top the other way)" if self._kind() == "rosbag" else
                                "check the yaw source and the telemetry lead (Ingest tab)")
                        return (f"⚠️ {f['reg_images']} cameras, **{f['points3D']} 3D points** although the images "
                                f"match well ({inliers:,} inlier matches): the pose priors do not fit the images, "
                                f"e.g. a camera orientation off by a fixed rotation; {hint}")
                    return (f"⚠️ {f['reg_images']} cameras but only **{f['points3D']} 3D points**: too little overlap "
                            "between the images (Images used: keep more of them?); dense and splat need points")
                e, d = f["point_error_px"], f["prior_deviation"]["position_m"]
                mode = " (chunked)" if r.get("ba", {}).get("mode") == "chunked" else ""
                its = [v["ba"]["iterations"] for k, v in sorted(r.items(), key=lambda kv: kv[0])
                       if k.startswith("round_") and isinstance(v, dict) and "ba" in v]
                cap = r.get("ba", {}).get("max_iterations")
                if its and cap:
                    mode += f" · BA iterations {', '.join(map(str, its))} / {cap}"
                drifts = [v["drift"] for k, v in sorted(r.items()) if isinstance(v, dict) and v.get("drift")]
                drifts += r.get("chunked", {}).get("drift") or []
                if drifts:  # the last estimate: revisits of a long flight brought together
                    dm = max((d.get("drift_max_m", 0.0) for d in drifts[-1]["sources"].values()), default=0.0)
                    if dm >= 0.05:
                        mode += f" · position drift up to {dm:.2f} m removed (revisits merged)"
                return (f"{f['reg_images']} cameras, {f['points3D']:,} points, error median {e['median']} px "
                        f"(p90 {e['p90']}), prior deviation median {d['median']} m (max {d['max']}){mode}")
            if st == "reconstruct":
                e = r["point_error_px"]
                return (f"{r.get('mapper', 'incremental')}: "
                        f"{r['reg_images']}/{r['images']} images registered, {r['points3D']:,} points, error median "
                        f"{e.get('median')} px, f = {r['camera']['params'][0]:.0f} px, up from {r['frame']['up']}"
                        + (f", **{len(r['models'])} models** (kept the largest)" if len(r["models"]) > 1 else ""))
            if st == "export":
                return f"{r['frames']} frames → transforms.json"
            if st == "fusion":
                return f"{r.get('points', r)}"
            if st == "semantic":
                r = rep["semantic"]
                return (f"{r['seen']:,} of {r['gaussians']:,} Gaussians from {r['photos']} photos ({r['version']}, "
                        f"{r['width']} px), {_fmt_s(r['seconds'])} · Search tab")
            if st == "splatmesh":
                r = rep["splat_mesh"]
                return (f"{r['faces']:,} faces from {r['views']} views, voxel {100 * r['voxel_m']:.1f} cm "
                        f"({r['quality']}), max depth {r['max_depth_m']:.0f} m")
            if st == "splat":
                r = rep["splat"]
                err = (r.get("lidar_depth") or {}).get("error")
                return (f"{r['num_gaussians']:,} gaussians, {r['method']} {r['iterations']:,} it, "
                        f"training {_fmt_s(r['train_seconds'])}"
                        + (f", LiDAR depth error median {err['median_pct']:.1f} % / p90 {err['p90_pct']:.1f} %"
                           if err else ""))
        except (KeyError, TypeError):
            pass
        text = str(r)
        return text[:160] + ("…" if len(text) > 160 else "")

    def _on_reconstruct_event(self, ev):
        if ev.get("phase") == "log":  # global / hierarchical: their current step
            self._job["detail"] = f"{self.g_mapper.value} mapper: {ev['line']}"
            return
        n = len(self.project.names) if self.project else 0
        k = ev.get("registered", 0)
        self._job["detail"] = f"registered {k} / {n} images"
        if n:
            self.g_progress.value = 100.0 * min(k, n) / n

    def _on_refine_event(self, ev):
        phase, rnd, rounds = ev["phase"], ev.get("round", 0), ev.get("rounds", 1)
        stage = ev.get("stage")
        what = {"calib": "calibration round", "chunk": f"chunk {ev.get('chunk', 0) + 1}/{ev.get('chunks', 1)}, round",
                "polish": "polish"}.get(stage, "round")
        step = f"{what} {rnd + 1}/{rounds}" if stage != "polish" else what
        if "chunks" in ev:  # chunked: the chunks fill the bar up to 85 %
            frac = (ev["chunk"] + (rnd + (phase == "ba") * 0.5) / max(1, rounds)) / ev["chunks"]
            self.g_progress.value = 85.0 * min(1.0, frac if phase != "chunk_done" else (ev["chunk"] + 1) / ev["chunks"])
        if phase == "chunk_done":
            return
        if phase == "posegraph":
            self._job["detail"] = f"pose graph: {ev['cameras']} cameras, {ev['edges']:,} edges …"
            self.g_progress.value = 86.0
        elif phase == "drift":
            self._job["detail"] = (f"drift pass {ev['pass'] + 1} (≤ {ev['passes']}): triangulating all images and "
                                   "measuring revisits …")
            self.g_progress.value = 87.0
        elif phase == "triangulate":
            self._job["detail"] = f"{step}: triangulating …"
            if "chunks" not in ev:
                self.g_progress.value = 88.0 if stage == "polish" else 100.0 * rnd / rounds
        elif phase == "ba":
            now = time.time()
            if now - self._last_live < 0.25:  # throttle scene updates
                return
            self._last_live = now
            win = f"window {ev['window'] + 1}/{ev['windows']}, " if ev.get("windows") else ""
            cap = int(self.g_ba_iters.value)
            self._job["detail"] = (f"{step}: bundle adjustment {win}iteration {ev['iteration']}"
                                   + (f" / {cap}" if cap else "") + f", cost {ev['cost']:.4g}")
            if stage == "polish" and ev.get("windows"):
                self.g_progress.value = 88.0 + 10.0 * (ev["window"] + 1) / ev["windows"]
            self._update_cameras_live(ev["rec"])
        elif phase in ("round_done", "done"):
            if ev.get("ba"):  # the round's final iteration count, e.g. 120 / 500 (SETTLED)
                ba = ev["ba"]
                msg = f"{step}: BA {ba['iterations']} / {ev.get('max_iterations')} iterations ({ba['termination']})"
                self._job["detail"] = msg
                self._on_log(msg)
            with self.lock:
                name = ev.get("name", f"round_{rnd}") if phase == "round_done" else "refined"
                # the pipeline writes the snapshot before the event: reading the files is much faster
                # than converting the Reconstruction (8000 images: 8 s instead of 4.5 min under the lock)
                if os.path.isdir(os.path.join(self.path, "sparse", name)):
                    self.snap = self.project.load_snapshot(name)
                else:
                    self.snap = self.project.snapshot_from_reconstruction(name, ev["rec"])
                self._draw_points()
                self._draw_cameras()
                self._draw_lines()
                self._update_stats()
            if stage not in ("polish",):
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
        self._dense_status("available", "available: " + (", ".join(have) if have else
                                                         "nothing yet (Dense / Splat tabs)"))
        self._draw_dense_points()
        self._draw_mesh()
        self._draw_splat()
        self._refresh_export()
        self._sync_quickbar()
        self._refresh_splat_plan()
        self._sync_resume()
        self.semantic.refresh()

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

    def _dense_status(self, layer, text):
        """One status line per layer under View → Dense / splat (layers no longer overwrite each other)."""
        lines = getattr(self, "_dense_lines", {})
        self._dense_lines = lines
        if text:
            lines[layer] = text
        else:
            lines.pop(layer, None)
        order = ("available", "points", "mesh", "splat")
        self.g_dense_md.content = "  \n".join(lines[k] for k in order if k in lines)

    def _unchanged(self, layer, key):
        """True if `layer` is already drawn from the same inputs (refreshes during a run don't re-send
        tens of MB to the browser); records `key` otherwise. key None: the layer is hidden."""
        drawn = getattr(self, "_drawn", {})
        self._drawn = drawn
        if key is not None and drawn.get(layer) == key:
            return True
        drawn[layer] = key
        return False

    @staticmethod
    def _file_key(path):
        return (path, os.path.getmtime(path)) if path and os.path.exists(path) else None

    def _draw_dense_points(self):
        key = (self._file_key(self.dense["fused"]), self.g_point_size.value) \
            if self.g_dense_pts.value and self.dense["fused"] else None
        if self._unchanged("points", key):
            return
        if key is not None:
            pts, cols, n = gdata.load_fused(self.dense["fused"])
            self.server.scene.add_point_cloud("/dense/points", points=pts, colors=cols,
                                              point_size=self.g_point_size.value * 0.5, precision="float32")
            self._dense_status("points", f"dense points: {len(pts):,} of {n:,} shown")
        else:
            self.server.scene.remove_by_name("/dense/points")
            self._dense_status("points", None)

    def _draw_mesh(self):
        method = self.g_mesh.value
        key = None
        if method != "off" and method in self.dense["meshes"]:
            key = (method, self._file_key(self.dense["meshes"][method]),
                   self._file_key(self.dense["textured"].get(method)), self.g_mesh_tex.value)
        if self._unchanged("mesh", key):
            return
        if key is not None:
            tex = self.dense["textured"].get(method) if self.g_mesh_tex.value else None
            mesh = gdata.load_textured_mesh(tex) if tex else None
            note = "textured"
            if mesh is None:
                mesh = gdata.load_mesh(self.dense["meshes"][method])
                note = ("from the splat, vertex colours" if method == "splat" else
                        "no texture yet: run the texture stage" if method not in self.dense["textured"] else
                        "untextured preview" if not self.g_mesh_tex.value else
                        f"textured mesh over {gdata.MAX_TEXTURED_FACES:,} faces: grey preview")
            self.server.scene.add_mesh_trimesh("/dense/mesh", mesh)
            self._dense_status("mesh", f"mesh ({method}): {len(mesh.faces):,} faces, {note}")
        else:
            self.server.scene.remove_by_name("/dense/mesh")
            self._dense_status("mesh", None)

    def _splat_limit(self, default):
        """How many Gaussians to send to the browser: None = all (View: All gaussians)."""
        return None if self.g_splat_all.value else default

    def _splat_all_changed(self):
        src = getattr(self, "_live_src", None)
        if getattr(self, "_splat_live_on", False) and src:  # reload the last live snapshot at the new size
            step, n, self._live_splat = gdata.load_live_splat(*src, self._splat_limit(gdata.LIVE_SPLAT_SHOWN))
        self._draw_splat()

    def _draw_splat(self):
        if getattr(self, "_splat_live_on", False):  # training: the checkbox shows / hides the live layer
            self._draw_live_splat()
            return
        key = ((self._file_key(self.dense["splat"]), self.g_splat_all.value)
               if self.g_splat.value and self.dense["splat"] else None)
        if self._unchanged("splat", key):
            return
        if key is not None:
            c, cov, rgb, op = gdata.load_splat(self.dense["splat"], self._splat_limit(gdata.SPLAT_SHOWN))
            total = gdata.splat_count(self.dense["splat"])
            self._splat_shown = (c, cov, rgb, op, gdata.splat_subset(total, self._splat_limit(gdata.SPLAT_SHOWN)))
            self.redraw_splat_decorated()
            self._dense_status("splat", f"splat: {total:,} gaussians" + (
                f" ({len(c):,} shown; View → All gaussians for all)" if len(c) < total else ""))
        else:
            self._splat_shown = None
            self.server.scene.remove_by_name("/dense/splat")
            self._dense_status("splat", None)

    def redraw_splat_decorated(self):
        """Send the loaded splat again with the semantic search applied (no file reload)."""
        shown = getattr(self, "_splat_shown", None)
        if shown is None or not self.g_splat.value or getattr(self, "_splat_live_on", False):
            return
        c, cov, rgb, op = self.semantic.decorate(*shown)
        self.server.scene.add_gaussian_splats("/dense/splat", centers=c, covariances=cov, rgbs=rgb, opacities=op)

    def serve_forever(self):
        while True:
            time.sleep(3600)


def last_project(root):
    """The project opened last (ROOT/.modelforge_gui.json "last_project") if it still exists, else the
    newest project under root, else ROOT/project_1."""
    try:
        with open(os.path.join(root, ".modelforge_gui.json")) as f:
            last = json.load(f).get("last_project")
    except (OSError, ValueError):
        last = None
    found = projects.list_projects(root)
    if last:
        path = os.path.abspath(os.path.join(root, last))
        if path in {os.path.abspath(p) for p in found}:
            return path
    return found[0] if found else os.path.join(root, "project_1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", nargs="?",
                    help="project folder to open (new or existing); default: the one opened last, else the newest "
                         "project under --root")
    ap.add_argument("--root", default="data", help="folder holding the projects (default data/)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if args.project:
        path = os.path.abspath(args.project)
    else:
        path = last_project(root)
    app = App(path, args.port, root=root)
    app._remember_project()  # here, not in App(): tests open projects too
    app.semantic.preload()  # the text model loads in the background, ready for the first search
    print(f"modelforge GUI on http://localhost:{args.port}", flush=True)
    app.serve_forever()


if __name__ == "__main__":
    main()

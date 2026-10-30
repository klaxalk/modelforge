"""GUI Ingest tab: pick videos, cut them into sources (time ranges), plan sampling, extract.

The project kind (ingest.json) picks the back end: modelforge.dji.ingest (DJI clips with telemetry)
or modelforge.video.ingest (videos without metadata: motion / time sampling, no telemetry views).

The plan lives in PROJECT/ingest.json (saved on every change) and is turned into
images + poses.json by modelforge.dji.ingest.run. While editing, the 3D view shows
each source's aligned flight track (bright inside the selected range) and the planned
image positions; a frame preview shows the video and telemetry at any time.
"""

import os
import threading
import time
import traceback
from dataclasses import asdict

import numpy as np
import viser

from modelforge import jobs, videos
from modelforge.dji import extract as ex
from modelforge.dji import ingest
from modelforge.rosbag import bag as rbag
from modelforge.rosbag import ingest as ros_ingest
from modelforge.video import ingest as video_ingest

SAMPLING = {"dji": ("distance", "time"), "video": video_ingest.SAMPLING, "rosbag": ros_ingest.SAMPLING}
NO_INFO, IMAGE_FRAME = "(none: estimate intrinsics)", "(image header frame)"
ALL_FOLDERS = "all folders"  # the Ingest tab's Folder: no filter


def _label(video):
    """A source's name in the Ingest tab: its folder and file name ("baracks/DJI_0483.MP4")."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(video)))
    return f"{parent}/{os.path.basename(video)}" if parent else os.path.basename(video)

PALETTE = [(70, 134, 251), (251, 185, 56), (27, 229, 181), (228, 70, 11), (164, 252, 60), (180, 90, 250),
           (250, 110, 180), (120, 200, 255)]
OUTSIDE_COLOR = (110, 110, 110)


def source_color(k):
    return PALETTE[k % len(PALETTE)]


def plan_dot_size(pts, largest=1.2):
    """The planned images' dot size (m): half the median step between a source's consecutive images, at most 1 %
    of the plan's extent and `largest` (large flights look as before; a small local scene is not filled by them)."""
    steps = [np.linalg.norm(np.diff(q, axis=0), axis=1) for q in pts if len(q) > 1]
    steps = np.concatenate(steps) if steps else np.zeros(0)
    steps = steps[steps > 0]
    size = 0.5 * float(np.median(steps)) if len(steps) else largest
    allp = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(allp) > 1:
        extent = float(np.ptp(allp, axis=0).max())
        if extent > 0:
            size = min(size, 0.01 * extent)
    return float(max(min(size, largest), 1e-3))


class IngestPanel:
    def __init__(self, app, gui):
        self.app, self.gui, self.server = app, gui, app.server
        self.path = app.path
        self._set_kind(ingest.load_kind(self.path))
        self.sources, self.settings = ingest.load_plan(self.path)
        if not self.sources:
            self.sources, self.settings = self._plan_from_poses()
        self.plan, self.prev = None, None
        self.running = False
        self.token = None  # jobs.Token of a running extraction (the Stop button cancels it)
        self.edited = False  # ingest.json is written only once the user changes the plan
        self._loading = False  # widget updates from set_project are not user edits
        self._lock = threading.RLock()  # source widgets: GUI callbacks vs. the planner thread
        self._dirty = threading.Event()
        self._preview_req = None
        self._pv_gen, self._pv_fresh = 0, True  # preview generation (a new project / source redraws); start at the source's start
        self._preview_lock = threading.Lock()
        self._source_handles = []
        self._build()
        threading.Thread(target=self._planner, daemon=True).start()
        threading.Thread(target=self._previewer, daemon=True).start()
        self.request_plan(edited=False)

    def _set_kind(self, kind):
        """Project kind -> back end with the same plan / extract interface (DJI or video without metadata)."""
        self.kind = kind
        self.be = {"video": video_ingest, "rosbag": ros_ingest}.get(kind, ingest)
        self._ros_bag = None  # bag whose topics fill the ROS 2 widgets

    def _apply_kind(self):
        """Show only the controls the kind uses (videos without metadata have no telemetry)."""
        poses = self.kind != "video"  # DJI and ROS bags carry poses; videos without metadata do not
        self.g_sampling.options = SAMPLING[self.kind]
        for h in (self.g_rotw, self.g_show):
            h.visible = poses
        self.g_yaw.visible = self.kind == "dji"
        self.g_ros.visible = self.kind == "rosbag"
        self.g_extract.label = "Extract images + priors" if poses else "Extract images"

    def set_project(self, path):
        """Show another project's plan (called by App.open_project)."""
        with self._lock:
            self.path = path
            self._set_kind(ingest.load_kind(path))
            self.sources, self.settings = ingest.load_plan(path)
            if not self.sources:
                self.sources, self.settings = self._plan_from_poses()
            self.plan, self.prev, self.edited = None, None, False
            self._loading = True
            try:
                st = self.settings
                self._apply_kind()
                self.g_sampling.value = st.sampling if st.sampling in SAMPLING[self.kind] else SAMPLING[self.kind][0]
                self.g_num.value = st.num_images
                self.g_spacing.value = st.spacing or 0.0
                self.g_rotw.value = -1.0 if st.rot_weight is None else st.rot_weight
                self.g_window.value = st.window
                self.g_yaw.value = st.yaw_source
            finally:
                self._loading = False
            self._rebuild_sources_locked()
        self.server.scene.remove_by_name("/ingest")
        self._clear_preview()
        self.g_status.content = ""
        self.g_plan_md.content = ""
        self._update_status()
        self.request_plan(edited=False)

    def _plan_from_poses(self):
        """Older single-video projects have no ingest.json: rebuild the plan from poses.json."""
        meta = self.app.project.meta if self.app.project else None
        if not meta or self.kind == "video":
            return [], (video_ingest.default_settings() if self.kind == "video" else ingest.Settings())
        s = meta.get("sampling", {})
        settings = ingest.Settings(sampling=s.get("mode", "distance"), num_images=len(meta["images"]),
                                   rot_weight=s.get("rot_weight_m_per_deg"), window=s.get("window", 0.5),
                                   yaw_source=meta.get("yaw_source", "aircraft"))
        srcs = meta.get("sources") or [{"video": meta.get("source"), "start": 0.0, "end": None}]
        out = []
        for d in srcs:
            v = d.get("video")
            if v and not os.path.exists(v):
                v = os.path.relpath(v, "/workspace") if v.startswith("/workspace/") else v
            if v:
                out.append(ingest.Source(v, d.get("start", 0.0), d.get("end")))
        return out, settings

    # ------------------------------------------------------------------ widgets
    def _build(self):
        gui = self.gui
        self.g_md = gui.add_markdown("")
        with gui.add_folder("Videos"):
            self.g_folder = gui.add_dropdown("Folder", options=(ALL_FOLDERS,), initial_value=ALL_FOLDERS,
                                             hint="Pick a folder first: the Video list then shows only its clips "
                                                  "(the number of clips in brackets).")
            self.g_folder.on_update(lambda _: self._filter_videos())
            self.g_video = gui.add_dropdown("Video", options=("(none found)",),
                                            hint="Videos of the library (data root, incl. data/videos/; project "
                                                 "folders are skipped; Videos tab to manage them). An .SRT sidecar "
                                                 "is optional: used if present for a quicker telemetry alignment.")
            self.g_video_path = gui.add_text("or path", initial_value="",
                                             hint="Any video path inside the container (repo = /workspace); "
                                                  "overrides the dropdown.")
            row = gui.add_button_group("", options=("Add as source", "Add all in folder", "Rescan"),
                                       hint="Add as source: the whole clip becomes a new source (narrow it with its "
                                            "range). Add all in folder: every clip of the chosen Folder becomes a "
                                            "source (those already in the sources are skipped). Rescan: look for "
                                            "new videos.")
            row.on_click(lambda e: {"Add as source": self.add_source,
                                    "Add all in folder": self.add_folder}.get(e.target.value, self.rescan)())
            from modelforge.gui import upload as stream_upload
            self.g_upload = gui.add_html(stream_upload.widget("videos:", hint=(
                "Upload a DJI video (.MP4 / .MOV), and optionally its .SRT sidecar, or a ROS bag file (ROS 1 .bag, "
                "ROS 2 .mcap) from this computer into the shared video library (data/videos/, managed in the Videos "
                "tab). The video carries all the metadata; the .SRT only gives a quicker, firmer telemetry alignment. "
                "A ROS 2 bag folder (metadata.yaml + .db3) goes through the Videos tab as a .zip. Streamed to disk, "
                "any size; an interrupted upload resumes (pick the same file again after a reload). An existing file "
                "is not overwritten.")))
            self.g_upload_md = gui.add_markdown("")
        self.g_ros = gui.add_folder("ROS bag")
        with self.g_ros:
            self.g_ros_md = gui.add_markdown("_add a bag as a source above_")
            self.g_ros_image = gui.add_dropdown("Image topic", options=("-",),
                                                hint="sensor_msgs/CompressedImage (preferred) or Image.")
            self.g_ros_info = gui.add_dropdown("Camera info", options=(NO_INFO,),
                                               hint="sensor_msgs/CameraInfo: known intrinsics, kept fixed in BA "
                                                    "(plumb_bob -> OPENCV, equidistant -> fisheye). None: estimated.")
            self.g_ros_pose = gui.add_dropdown("Pose source", options=ros_ingest.POSE_SOURCES,
                                               hint="tf: the camera pose from the TF tree (world -> camera frame) at "
                                                    "each image stamp. odometry: an Odometry topic (its frame -> its "
                                                    "child), chained with TF to the world and to the camera. none: "
                                                    "images only (no TF to the camera, no odometry): sampled by image "
                                                    "motion and reconstructed like a video without metadata.")
            self.g_ros_world = gui.add_dropdown("World frame", options=("-",),
                                                hint="The frame the poses are expressed in: the project's world.")
            self.g_ros_camera = gui.add_dropdown("Camera frame", options=(IMAGE_FRAME,),
                                                 hint="The camera's optical frame (x right, y down, z forward; ROS "
                                                      "*_optical_frame). Default: the image messages' frame_id. "
                                                      "When TF has no frame for the camera, pick the frame it is "
                                                      "mounted on (e.g. the body, x forward, z up) and use "
                                                      "Extrinsic RPY.")
            self.g_ros_use_rpy = gui.add_checkbox(
                "Use extrinsic RPY", initial_value=False,
                hint="The Camera frame is a mount frame (ROS body convention: x forward, y left, z up), not the "
                     "optical frame: the camera's orientation in it is given by Extrinsic RPY below.")
            self.g_ros_rpy = gui.add_vector3(
                "Extrinsic RPY [deg]", initial_value=(0.0, 0.0, 0.0), step=1.0,
                hint="Camera orientation in the Camera frame: roll about x, pitch about y, yaw about z, in that "
                     "order about the frame's fixed axes, starting from a camera looking forward with the image's "
                     "top up. 0, 0, 0 = forward; 0, 90, 0 = straight down (image top towards the front); "
                     "0, 30, 0 = forward, tilted 30° down. SfM refines the last few degrees.")
            self.g_ros_odom = gui.add_dropdown("Odometry topic", options=("-",),
                                               hint="nav_msgs/Odometry, for pose source odometry.")
            self.g_ros_clouds = gui.add_folder("Point clouds")
            with self.g_ros_clouds:
                self.g_ros_clouds_md = gui.add_markdown("")
            self.g_ros_voxel = gui.add_number("LiDAR voxel [m]", initial_value=0.05, min=0.0, step=0.01,
                                              hint="The merged cloud keeps one point per voxel (0 = all points).")
            self.g_ros_range = gui.add_number("LiDAR max range [m]", initial_value=60.0, min=1.0, step=5.0,
                                              hint="Returns farther from the sensor are dropped.")
            for h in (self.g_ros_image, self.g_ros_info, self.g_ros_pose, self.g_ros_world, self.g_ros_camera,
                      self.g_ros_odom, self.g_ros_voxel, self.g_ros_range, self.g_ros_use_rpy, self.g_ros_rpy):
                h.on_update(lambda _: self._ros_changed())
            self._cloud_boxes = {}
        self.g_sources = gui.add_folder("Sources")
        with self.g_sources:
            self.g_sources_md = gui.add_markdown("")
        with gui.add_folder("Sampling"):
            st = self.settings
            opts = SAMPLING[self.kind]
            self.g_sampling = gui.add_dropdown("Sampling", options=opts,
                                               initial_value=st.sampling if st.sampling in opts else opts[0],
                                               hint="distance (DJI): equal steps of travelled distance + "
                                                    "view-rotation credit (none while hovering). motion (videos "
                                                    "without metadata): equal steps of image motion (optical flow), "
                                                    "none while standing still. time: equal steps of video time.")
            self.g_num = gui.add_number("Images (all sources)", initial_value=st.num_images, min=1, step=1,
                                        hint="Total images over all sources, spread by the sampling curve "
                                             "(spacing = total / images). Without metadata SfM needs overlap: "
                                             "keep consecutive images well within one image width of motion.")
            self.g_spacing = gui.add_number("Spacing (0 = from images)", initial_value=st.spacing or 0.0, min=0.0,
                                            step=0.1, hint="Fixed step between images in m (distance) or s (time) "
                                                           "instead of a total image count. 0 = use Images.")
            self.g_rotw = gui.add_number("Rotation credit m/° (−1 = auto)",
                                         initial_value=-1.0 if st.rot_weight is None else st.rot_weight,
                                         min=-1.0, step=0.01,
                                         hint="Metres of travel one degree of view rotation counts as, so turns "
                                              "on the spot still get images. −1 = median look-at depth × π/180.")
            self.g_window = gui.add_slider("Sharpest-frame window", min=0.1, max=1.0, step=0.05,
                                           initial_value=st.window,
                                           hint="Fraction of the spacing searched around each target for the "
                                                "sharpest frame (Laplacian variance).")
            self.g_yaw = gui.add_dropdown("Yaw source", options=("auto", "aircraft", "gimbal"),
                                          initial_value=st.yaw_source,
                                          hint="Camera heading. auto (default): per clip, the gimbal yaw when it "
                                               "is a true heading (Mavic 4: its gimbal can yaw, so that is the "
                                               "camera's own heading), else the aircraft yaw (NEO2: the gimbal "
                                               "cannot yaw and reports yaw in an offset frame). The source "
                                               "summary shows which one each clip uses.")
            for h in (self.g_sampling, self.g_num, self.g_spacing, self.g_rotw, self.g_window, self.g_yaw):
                h.on_update(lambda _: self._settings_changed())
        self.g_fc = gui.add_folder("Flight alignment (several flights)")
        with self.g_fc:
            gui.add_markdown(
                "The flights' altitude offsets come from the drones' absolute altitude, which can be wrong (karlak3: "
                "240 m low). The check measures them from the images: ~40 photos per flight where the flights look at "
                "the same ground, matched across, refined with an offset per flight (a few minutes). Run full "
                "pipeline does it by itself before extracting.")
            self.g_fc_run = gui.add_button("Check flight alignment (images)", icon=viser.Icon.ARROWS_VERTICAL,
                                           hint="Extracts a few photos per flight into flight_check/ (not the "
                                                "project's images), matches them across the flights and refines "
                                                "with an offset per flight. Runs as a job (Stop works).")
            self.g_fc_run.on_click(lambda _: self.check_flights())
            self.g_fc_md = gui.add_markdown("")
            self.g_fc_use = gui.add_button("Use the measured altitude offsets", icon=viser.Icon.CHECK, disabled=True,
                                           hint="Puts the measured offsets into each source's Altitude offset "
                                                "(flights the check could not tie to the others keep theirs).")
            self.g_fc_use.on_click(lambda _: self.apply_flight_check())
        self.g_plan_md = gui.add_markdown("")
        with gui.add_folder("Frame preview"):
            self.g_pv_source = gui.add_dropdown("Source", options=("-",), hint="Source to preview.")
            self.g_pv_time = gui.add_slider("Time [s]", min=0.0, max=1.0, step=0.1, initial_value=0.0,
                                            hint="Video time; shows the frame, the aligned telemetry and a white "
                                                 "camera in the 3D view.")
            self.g_pv_image = gui.add_image(np.zeros((36, 64, 3), np.uint8))
            self.g_pv_md = gui.add_markdown("")
            row = gui.add_button_group("Set range", options=("Start here", "End here"),
                                       hint="Set the previewed source's range start or end to the current time.")
            row.on_click(lambda e: self._set_range_here(e.target.value == "Start here"))
            self.g_pv_source.on_update(lambda _: self._preview_source_changed())
            self.g_pv_time.on_update(lambda _: self.request_preview())
        self.g_show = gui.add_checkbox("Show flight plan in 3D", initial_value=True,
                                       hint="Tracks per source (bright inside the range, grey outside) and dots at "
                                            "the planned image positions.")
        self.g_show.on_update(lambda _: self._draw_scene())
        self.g_overwrite = gui.add_checkbox("Overwrite existing images", initial_value=False,
                                            hint="Old images and everything derived from them (databases, sparse, "
                                                 "dense, splat, report) move to previous/<time>/, nothing is deleted.")
        self.g_extract = gui.add_button("Extract images + priors", icon=viser.Icon.PHOTO,
                                        hint="Write images/, poses.json and poses.csv for the plan. Decodes each "
                                             "video once (pieces late in a video take longer). Then: Sparse tab.")
        self.g_extract.on_click(lambda _: self.extract())
        self.g_progress = gui.add_progress_bar(value=0.0, visible=False)
        self.g_status = gui.add_markdown("")
        self._apply_kind()
        self.rescan()
        self._rebuild_sources()

    def rescan(self):
        roots = [self.app.root, os.path.dirname(os.path.abspath(self.path)), os.path.join(os.getcwd(), "data")]
        found = []
        find = rbag.find_bags if self.kind == "rosbag" else ingest.find_videos
        for r in dict.fromkeys(roots):
            if os.path.isdir(r):
                found += [os.path.relpath(v) for v in find(r)]
        self._found = list(dict.fromkeys(found))
        counts = {}
        for v in self._found:
            d = os.path.dirname(v) or "."
            counts[d] = counts.get(d, 0) + 1
        old = getattr(self, "_folders", {}).get(self.g_folder.value)
        self._folders = {f"{d} ({n})": d for d, n in sorted(counts.items())}
        opts = (ALL_FOLDERS,) + tuple(self._folders)
        self.g_folder.options = opts
        # the same folder again after a rescan (its count may have changed)
        self.g_folder.value = next((o for o, d in self._folders.items() if d == old), ALL_FOLDERS)
        self._filter_videos()

    def _filter_videos(self):
        """The Video list: every clip (paths), or those of the chosen folder (file names)."""
        folder = getattr(self, "_folders", {}).get(self.g_folder.value)
        found = getattr(self, "_found", [])
        if folder is None:
            self._video_paths = {v: v for v in found}
        else:
            self._video_paths = {os.path.basename(v): v for v in found if (os.path.dirname(v) or ".") == folder}
        self.g_video.options = tuple(self._video_paths) or ("(none found)",)

    def _unusable(self, v):
        """Why `v` cannot be a source of this project, or None."""
        if not v or v == "(none found)" or not os.path.exists(v):
            return f"❌ video not found: `{v}`"
        if self.kind == "rosbag" and not rbag.is_bag(v):
            return f"❌ not a ROS bag (a ROS 1 .bag, a ROS 2 .mcap, or a folder with metadata.yaml): `{v}`"
        if self.kind != "rosbag" and rbag.is_bag(v):
            return f"❌ `{v}` is a ROS bag: it needs a project of kind ROS bag"
        return None

    def add_source(self, video=None):
        v = video or self.g_video_path.value.strip() or getattr(self, "_video_paths", {}).get(
            self.g_video.value, self.g_video.value)
        why = self._unusable(v)
        if why:
            self.g_status.content = why
            return
        self.sources.append(ingest.Source(v))
        self.g_video_path.value = ""
        self._sources_added(1)

    def add_folder(self):
        """Every clip of the chosen Folder as a source, each whole; clips already among the sources are skipped."""
        folder = getattr(self, "_folders", {}).get(self.g_folder.value)
        if folder is None:
            self.g_status.content = "❌ pick a Folder first (Add all in folder adds the clips of one folder)"
            return
        have = {os.path.abspath(s.video) for s in self.sources}
        clips = [v for v in getattr(self, "_video_paths", {}).values() if (os.path.dirname(v) or ".") == folder]
        new = [v for v in clips if os.path.abspath(v) not in have and not self._unusable(v)]
        self.sources += [ingest.Source(v) for v in new]
        skipped = len(clips) - len(new)
        self.g_status.content = (f"added {len(new)} clip(s) from `{folder}`"
                                 + (f" · {skipped} skipped (already a source or not usable)" if skipped else ""))
        if new:
            self._sources_added(len(new))

    def _sources_added(self, n):
        self._rebuild_sources()
        self._clear_preview()
        self.g_pv_source.value = self.g_pv_source.options[-n]  # the first new one's first frame, once decoded
        self.request_plan()

    def _rebuild_sources(self):
        """One folder per source; rebuilt on add/remove (range edits update in place)."""
        with self._lock:
            self._rebuild_sources_locked()

    def _rebuild_sources_locked(self):
        for h in self._source_handles:
            h.remove()
        self._source_handles = []
        self._source_widgets = []
        gui = self.gui
        with self.g_sources:
            for k, s in enumerate(self.sources):
                f = gui.add_folder(f"{k + 1}. {_label(s.video)}")
                with f:
                    md = gui.add_markdown("analysing …")
                    end = s.end if s.end is not None else max(s.start + 1.0, 1.0)
                    rng = gui.add_multi_slider("Range [s]", min=0.0, max=max(end, 1.0), step=0.5,
                                               initial_value=(s.start, end), min_range=0.5, disabled=True,
                                               hint="Video time range used from this source. Trim takeoff, landing "
                                                    "or near-horizontal views here; the track turns grey outside.")
                    en = gui.add_checkbox("Enabled", initial_value=s.enabled,
                                          hint="Untick to leave this source out of the plan without removing it.")
                    lead = gui.add_number("Lead [frames] (−1 = auto)", initial_value=-1.0 if s.lead is None else s.lead,
                                          min=-1.0, step=0.1,
                                          hint="How many frames the djmd telemetry runs ahead of the video. "
                                               "Auto = estimated from the SRT, or without one from the video's "
                                               "optical flow (21.6 / 21.5 on the NEO2 sample).")
                    alt = gui.add_text("Altitude offset [m]", initial_value="auto" if s.alt_offset is None
                                       else f"{s.alt_offset:g}",
                                       hint="Added to this flight's height above its takeoff (rel_alt), so flights "
                                            "from different takeoff heights line up. auto = the difference of the "
                                            "takeoff altitudes (abs_alt) against the first flight, refused when the "
                                            "takeoffs are close but abs_alt says tens of metres apart (it can be "
                                            "wrong: karlak3 read 240 m low). Or a number in metres; the refine "
                                            "corrects the rest (per-flight offset).")
                    btn = gui.add_button_group("", options=("Preview", "Add piece", "Remove"),
                                               hint="Preview: show this source in Frame preview. Add piece: another "
                                                    "range of the same video. Remove: drop the source.")
                rng.on_update(lambda _, k=k: self._range_changed(k))
                en.on_update(lambda _, k=k: self._enabled_changed(k))
                lead.on_update(lambda _, k=k: self._lead_changed(k))
                lead.visible = self.kind == "dji"
                alt.on_update(lambda _, k=k: self._alt_changed(k))
                alt.visible = self.kind == "dji"
                btn.on_click(lambda e, k=k: self._source_action(k, e.target.value))
                self._source_handles.append(f)
                self._source_widgets.append({"md": md, "range": rng, "enabled": en, "lead": lead, "alt": alt,
                                             "ready": False})
        self.g_sources_md.content = "" if self.sources else "Add a video above. Each source is a video and a time " \
            "range; add the same video again (\"Add piece\") to take several pieces."
        self.g_fc.visible = self.kind == "dji" and len({s.video for s in self.sources if s.enabled}) >= 2
        self.show_flight_check()
        self.g_pv_source.options = tuple(f"{k + 1}. {_label(s.video)}" for k, s in enumerate(self.sources)) \
            or ("-",)
        if self.g_pv_source.value not in self.g_pv_source.options:  # the previewed source is gone
            self._clear_preview()
            self.g_pv_source.value = self.g_pv_source.options[0]

    def _clear_preview(self):
        """Empty Frame preview (another project or source): the next preview starts at the source's start."""
        self._pv_gen += 1
        self._pv_fresh = True
        self.g_pv_image.image = np.zeros((36, 64, 3), np.uint8)
        self.g_pv_md.content = ""
        self.server.scene.remove_by_name("/ingest/cursor")

    def _source_action(self, k, action):
        if action == "Remove":
            del self.sources[k]
            self._rebuild_sources()
        elif action == "Add piece":
            s = self.sources[k]
            self.sources.insert(k + 1, ingest.Source(s.video, s.start, s.end, s.enabled, s.lead, s.srt, s.alt_offset))
            self._rebuild_sources()
        elif action == "Preview":
            self.g_pv_source.value = self.g_pv_source.options[k]
            self.g_pv_time.value = float(np.clip(self.sources[k].start, self.g_pv_time.min, self.g_pv_time.max))
            self.request_preview()
            return
        self.request_plan()

    def _range_changed(self, k):
        w = self._source_widgets[k]
        if not w["ready"]:
            return
        a, b = w["range"].value
        s = self.sources[k]
        s.start = float(a)
        s.end = None if b >= w["range"].max - 1e-6 else float(b)
        self.request_plan()

    def _enabled_changed(self, k):
        self.sources[k].enabled = bool(self._source_widgets[k]["enabled"].value)
        self.request_plan()

    def _lead_changed(self, k):
        v = float(self._source_widgets[k]["lead"].value)
        self.sources[k].lead = None if v < 0 else v
        self.request_plan()

    # ------------------------------------------------------------------ flight alignment (dji/flight_check.py)
    def flight_check_wanted(self):
        """Run full pipeline checks the flights first: DJI, two or more videos, an offset left on auto."""
        en = [s for s in self.sources if s.enabled]
        return self.kind == "dji" and len({s.video for s in en}) >= 2 and any(s.alt_offset is None for s in en)

    def check_flights(self):
        if self.running or self.app.running:
            return
        ingest.save_plan(self.path, self.sources, self.settings)  # the check reads ingest.json
        self.app.run_stage("flightcheck")

    def show_flight_check(self):
        from modelforge.dji import flight_check
        res = flight_check.load(self.path) if self.path else None
        if not res:
            self.g_fc_md.content, self.g_fc_use.disabled = "", True
            return
        lines = [f"**Last check** ({res['registered']} of {res['photos']} photos registered, {res['seconds']:.0f} s):"]
        for f in res["flights"]:
            lines.append(f"- {f['name']}: altitude offset **{f['alt_offset_m']:+.1f} m** (was {f['offset_used_m']:+.1f})"
                         f" · {f['merged_points']} points shared"
                         + ("" if f["trusted"] else " · ⚠ not trusted (too few shared points, or still moving)"))
        for k, e in (res.get("apart") or {}).items():
            x, y, z = e["apart_m"]
            lines.append(f"- {k}: still {(x * x + y * y) ** 0.5:.1f} m apart horizontally (GPS; the refine "
                         f"corrects it), {abs(z):.1f} m vertically")
        self.g_fc_md.content = "  \n".join(lines)
        self.g_fc_use.disabled = not any(f["trusted"] for f in res["flights"])

    def apply_flight_check(self, only_auto=False):
        """The measured offsets into the sources' Altitude offset (trusted flights; every piece of that video).
        only_auto: leave offsets set by hand. Returns {video name: offset} applied."""
        from modelforge.dji import flight_check
        res = flight_check.load(self.path)
        if not res:
            return {}
        done = {}
        for f in res["flights"]:
            if not f["trusted"]:
                continue
            for k, s in enumerate(self.sources):
                if os.path.abspath(s.video) == os.path.abspath(f["video"]) and not (only_auto and s.alt_offset is not None):
                    s.alt_offset = round(f["alt_offset_m"], 2)
                    if k < len(self._source_widgets) and "alt" in self._source_widgets[k]:
                        self._source_widgets[k]["alt"].value = f"{s.alt_offset:g}"
                    done[f["name"]] = s.alt_offset
        if done:
            ingest.save_plan(self.path, self.sources, self.settings)
            self.request_plan()
        return done

    def _alt_changed(self, k):
        """Altitude offset: "auto" (or empty) = from the takeoff altitudes, else metres; anything else is left."""
        w = self._source_widgets[k]["alt"]
        text = w.value.strip().lower().replace(",", ".").replace("m", "").strip()
        if text in ("", "auto"):
            v = None
        else:
            try:
                v = float(text)
            except ValueError:
                return  # half typed ("-", "1e"): wait for more
        if v != self.sources[k].alt_offset:
            self.sources[k].alt_offset = v
            self.request_plan()

    def _settings_changed(self):
        if self._loading:
            return
        self.settings = ingest.Settings(
            sampling=self.g_sampling.value, num_images=max(1, int(self.g_num.value)),
            spacing=float(self.g_spacing.value) or None,
            rot_weight=None if self.g_rotw.value < 0 else float(self.g_rotw.value),
            window=float(self.g_window.value), yaw_source=self.g_yaw.value, quality=self.settings.quality,
            ros=self.settings.ros)
        self.request_plan()

    # ------------------------------------------------------------------ ROS bags
    def _load_clip(self, s):
        if self.kind == "rosbag":
            return self.be.load_clip(s.video, ros=self._ros_full(s.video))
        return self.be.load_clip(s.video, s.lead, s.srt)

    def _ros_full(self, path):
        r = dict(ros_ingest.ros_defaults(path))
        r.update({k: v for k, v in (self.settings.ros or {}).items() if v is not None})
        return r

    def _ros_refresh(self, path):
        """Fill the ROS 2 widgets from a bag's topics and TF frames (and the saved Settings.ros)."""
        inv = rbag.inventory(path)
        r = self._ros_full(path)
        frames = rbag.tf_buffer(path).frames
        clouds = rbag.topics_of(inv, rbag.CLOUD_TYPE)
        self._loading = True
        try:
            self.g_ros_image.options = tuple(rbag.topics_of(inv, rbag.IMAGE_TYPES)) or ("-",)
            self.g_ros_info.options = (NO_INFO,) + tuple(rbag.topics_of(inv, rbag.INFO_TYPE))
            self.g_ros_world.options = tuple(frames) or ("-",)
            self.g_ros_camera.options = (IMAGE_FRAME,) + tuple(frames)
            self.g_ros_odom.options = tuple(rbag.topics_of(inv, rbag.ODOM_TYPE)) or ("-",)
            for h, v in ((self.g_ros_image, r["image_topic"]), (self.g_ros_info, r["info_topic"] or NO_INFO),
                         (self.g_ros_pose, r["pose_source"]), (self.g_ros_world, r["world_frame"]),
                         (self.g_ros_camera, r["camera_frame"] or IMAGE_FRAME), (self.g_ros_odom, r["odom_topic"])):
                if v in h.options:
                    h.value = v
            self.g_ros_voxel.value, self.g_ros_range.value = r["voxel"], r["max_range"]
            self._ros_pose_widgets()
            rpy = r.get("camera_rpy")
            self.g_ros_use_rpy.value = rpy is not None
            self.g_ros_rpy.value = tuple(float(v) for v in (rpy if rpy is not None else (0.0, 0.0, 0.0)))
            for h in self._cloud_boxes.values():
                h.remove()
            self._cloud_boxes = {}
            with self.g_ros_clouds:
                for t in clouds:
                    cb = self.gui.add_checkbox(t, initial_value=t in (r["cloud_topics"] or []),
                                               hint=f"Merge {t} into the LiDAR cloud (world frame, voxel-"
                                                    "downsampled).")
                    cb.on_update(lambda _: self._ros_changed())
                    self._cloud_boxes[t] = cb
            self.g_ros_clouds_md.content = "" if clouds else "_no PointCloud2 topics in this bag_"
        finally:
            self._loading = False
        dur = inv["end"] - inv["start"]
        name = os.path.basename(path.rstrip("/"))
        self.g_ros_md.content = f"`{name}` · {dur:.0f} s · {len(inv['topics'])} topics · {len(frames)} TF frames"
        img_frame = inv["frame_of"].get(r["image_topic"], "")
        if img_frame and img_frame not in frames:
            self.g_ros_md.content += (f"  \n⚠️ the images' frame `{img_frame}` is not in TF: set Camera frame to the "
                                      "frame the camera is mounted on and use Extrinsic RPY")
        self._ros_bag = path

    def _ros_pose_widgets(self):
        """Pose source none (images only): the frame / odometry / mount fields do not apply."""
        posed = self.g_ros_pose.value != "none"
        for h in (self.g_ros_world, self.g_ros_camera, self.g_ros_odom, self.g_ros_use_rpy, self.g_ros_rpy):
            h.visible = posed

    def _ros_changed(self):
        self._ros_pose_widgets()
        if self._loading:
            return
        ros = {"image_topic": self.g_ros_image.value, "pose_source": self.g_ros_pose.value,
               "info_topic": "" if self.g_ros_info.value == NO_INFO else self.g_ros_info.value,
               "world_frame": self.g_ros_world.value,
               "camera_frame": "" if self.g_ros_camera.value == IMAGE_FRAME else self.g_ros_camera.value,
               "odom_topic": "" if self.g_ros_odom.value == "-" else self.g_ros_odom.value,
               "cloud_topics": [t for t, cb in self._cloud_boxes.items() if cb.value],
               "voxel": float(self.g_ros_voxel.value), "max_range": float(self.g_ros_range.value),
               "camera_rpy": [float(v) for v in self.g_ros_rpy.value] if self.g_ros_use_rpy.value else None}
        self.settings = ingest.Settings(**{**asdict(self.settings), "ros": ros})
        for w in self._source_widgets:  # the clip (poses) depends on the topics and frames: re-analyse
            w["ready"] = False
        self.request_plan()

    # ------------------------------------------------------------------ planning
    def request_plan(self, edited=True):
        self.edited |= edited
        self._dirty.set()

    def _planner(self):
        """Background replanning: clips decode once (cached), plans take milliseconds."""
        while True:
            self._dirty.wait()
            time.sleep(0.15)  # coalesce slider drags
            self._dirty.clear()
            try:
                with self._lock:
                    self._replan()
            except Exception as e:
                traceback.print_exc()
                self.g_plan_md.content = f"planning failed: `{e}`"

    def _replan(self):
        for k, s in enumerate(list(self.sources)):
            if k >= len(self._source_widgets):
                break
            w = self._source_widgets[k]
            if not w["ready"]:
                w["md"].content = {"video": "⏳ analysing: measuring image motion (about a minute once per clip) …",
                                   "rosbag": "⏳ reading the bag: image stamps, TF / odometry poses …"}.get(
                    self.kind, "⏳ analysing: decoding telemetry, estimating the lead …")
                if self.kind == "rosbag" and k == 0 and self._ros_bag != s.video:
                    self._ros_refresh(s.video)
                c = self._load_clip(s)
                w["range"].max = round(c.duration, 1)
                end = c.duration if s.end is None else min(s.end, c.duration)
                w["range"].value = (min(s.start, end - 0.5), end)
                w["range"].disabled = False
                w["ready"] = True
        if not any(s.enabled for s in self.sources):
            self.plan, self.prev = None, None
            self.g_plan_md.content = "no enabled sources"
            self._draw_scene()
            self._update_status()
            return
        if self.kind == "video":
            for s in self.sources:  # the motion curve decodes the whole clip once (cached on disk)
                if s.enabled and self.settings.sampling == "motion":
                    self.be.load_clip(s.video).motion
        plan = self.be.make_plan(self.sources, self.settings)
        origin = self.app.project.meta.get("enu_origin_lla") if self.app.project else None
        self.plan, self.prev = plan, self.be.preview(plan, self.settings.yaw_source, origin=origin)
        if self.edited:
            ingest.save_plan(self.path, self.sources, self.settings)
        by_source = {id(sp.source): sp for sp in plan.sources}
        for k, s in enumerate(self.sources):
            if k >= len(self._source_widgets):
                break
            c = self._load_clip(s)
            sp = by_source.get(id(s))
            extra = {"dji": f" · lead {c.lead:g} frames ({c.align['method']})"
                            + (f" · heading from {c.yaw_source(self.settings.yaw_source)} yaw"
                               if self.kind == "dji" else ""),
                     "rosbag": f" · poses from {c.align['method']}: {int(c.valid_pose.sum())}/{c.n} images"
                     if self.kind == "rosbag" else ""}.get(self.kind, "")
            lines = [f"📁 `{os.path.dirname(s.video) or '.'}`",
                     f"{c.duration:.0f} s, {c.width}×{c.height} @ {c.fps:.2f} fps" + extra]
            if sp:
                r = sp.report
                if "path_m" in r:
                    lines.append(f"selected: {r['path_m']:.0f} m path, {r['rotation_deg']:.0f}° rotation → "
                                 f"**{r['num_images']} images** · alt offset {sp.alt_offset:+.2f} m")
                else:
                    lines.append(f"selected → **{r['num_images']} images**")
            elif not s.enabled:
                lines.append("disabled")
            self._source_widgets[k]["md"].content = "  \n".join(lines)
        r = plan.report
        md = [f"**plan: {plan.num_images} images** from {len(plan.sources)} source(s) · spacing "
              f"{r['spacing']:.3g} {r['unit']}"]
        if r.get("mode") == "motion" and r["spacing"] > 0.25:
            md.append("⚠️ consecutive images move more than a quarter image width apart: SfM without metadata may "
                      "not connect them (more images or a shorter range)")
        if r.get("rot_weight_m_per_deg") is not None and r["mode"] == "distance":
            md.append(f"path {r['path_m']:.0f} m + rotation {r['rotation_deg']:.0f}° × "
                      f"{r['rot_weight_m_per_deg']:.2f} m/°")
        md += [f"⚠️ {w}" for w in plan.warnings]
        self.g_plan_md.content = "  \n".join(md)
        self._draw_scene()
        self._update_status()
        self._preview_source_changed(keep_time=True)

    def _update_status(self):
        n_img = len(self.app.project.names) if self.app.project else 0
        lines = []
        if n_img:
            extracted = self.app.project.meta.get("plan")
            try:  # through the dataclasses: fields added later (alt_offset) take their defaults
                extracted = {"sources": [asdict(ingest.Source(**x)) for x in extracted["sources"]],
                             "settings": asdict(ingest.Settings(**extracted["settings"]))} if extracted else None
            except (TypeError, KeyError):
                pass
            same = extracted == {"sources": [asdict(s) for s in self.sources], "settings": asdict(self.settings)}
            lines.append(f"{n_img} images extracted" + ("" if same or extracted is None else
                         " · **the plan differs from the extracted images** (re-extract with overwrite)"))
        elif self.path and ingest.has_outputs(self.path):  # images without poses.json: an extraction that stopped
            n_left = len(os.listdir(os.path.join(self.path, "images")))
            lines.append(f"{n_left} images from an interrupted extraction (no poses.json): tick **Overwrite** and "
                         "extract again (they move to `previous/`)")
        else:
            lines.append("no images yet: add sources, check the plan, then extract")
        self.g_md.content = "  \n".join(lines)
        self.g_overwrite.visible = bool(n_img) or bool(self.path and ingest.has_outputs(self.path))

    # ------------------------------------------------------------------ display
    def _draw_scene(self):
        scene = self.server.scene
        sync = getattr(self.app, "_sync_quickbar", None)
        if sync is not None and getattr(self.app, "_quick_labels", None) is not None:
            sync()  # the quick bar's Plan becomes available once there is a plan to show
        if not (self.g_show.value and self.prev):
            scene.remove_by_name("/ingest")
            return
        segs_in, cols_in, segs_out = [], [], []
        pts, pcols = [], []
        # where the refine put each flight (its per-flight offset; poses.json source k = plan source k)
        offsets = getattr(self.app.project, "source_offsets", None) or {} if self.app.project is not None else {}
        for sp, pv in zip(self.plan.sources, self.prev["sources"]):
            o = np.asarray(offsets.get(sp.index, np.zeros(3)), np.float32)
            t, v = pv["track"].astype(np.float32) + o, pv["track_valid"]
            seg = np.stack([t[:-1], t[1:]], 1)
            inside = v[:-1] & v[1:]
            segs_in.append(seg[inside])
            cols_in.append(np.tile(np.array(source_color(sp.index), np.uint8), (int(inside.sum()), 2, 1)))
            segs_out.append(seg[~inside])
            pts.append(np.asarray(pv["samples"], np.float32) + o)
            pcols.append(np.tile(np.array(source_color(sp.index), np.uint8), (len(pv["samples"]), 1)))
        segs_in = np.concatenate(segs_in) if segs_in else np.zeros((0, 2, 3), np.float32)
        cols_in = np.concatenate(cols_in) if cols_in else np.zeros((0, 2, 3), np.uint8)
        segs_out = np.concatenate(segs_out) if segs_out else np.zeros((0, 2, 3), np.float32)
        scene.add_line_segments("/ingest/selected", segs_in, colors=cols_in, thickness=2.5, thickness_units="screen")
        scene.add_line_segments("/ingest/outside", segs_out, colors=OUTSIDE_COLOR, thickness=1.0,
                                thickness_units="screen")
        p = np.concatenate(pts).astype(np.float32) if pts else np.zeros((1, 3), np.float32)
        c = np.concatenate(pcols) if pcols else np.zeros((1, 3), np.uint8)
        self._dot = plan_dot_size(pts)
        scene.add_point_cloud("/ingest/samples", points=p, colors=c, point_size=self._dot, precision="float32",
                              point_shape="circle")

    # ------------------------------------------------------------------ frame preview
    def _preview_index(self):
        opts = self.g_pv_source.options
        if self.g_pv_source.value not in opts or opts == ("-",):
            return None
        return opts.index(self.g_pv_source.value)

    def _preview_source_changed(self, keep_time=False):
        k = self._preview_index()
        if k is None or k >= len(self._source_widgets) or not self._source_widgets[k]["ready"]:
            return
        s = self.sources[k]
        c = self._load_clip(s)
        t = self.g_pv_time.value
        self.g_pv_time.max = round(c.duration, 1)
        if not keep_time or t > c.duration or self._pv_fresh:
            self.g_pv_time.value = float(min(max(s.start, 0.0), c.duration))
        self._pv_fresh = False
        self.request_preview()

    def request_preview(self):
        k = self._preview_index()
        if k is None or k >= len(self.sources):
            return
        with self._preview_lock:
            self._preview_req = (self._pv_gen, k, float(self.g_pv_time.value))

    def _previewer(self):
        last = None
        while True:
            time.sleep(0.1)
            with self._preview_lock:
                req, self._preview_req = self._preview_req, None
            if req is None or req == last:
                continue
            last = req
            try:
                self._show_preview(*req[1:])
            except Exception as e:
                traceback.print_exc()
                self.g_pv_md.content = f"preview failed: `{e}`"

    def _show_preview(self, k, t):
        s = self.sources[k]
        c = self._load_clip(s)
        i = int(np.clip(np.searchsorted(c.pts, t), 0, c.n - 1))
        self.g_pv_image.image = c.frame_image(c.pts[i])
        if self.kind == "rosbag" and not c.has_poses:  # images only: frame and time, as for videos
            inside = s.start <= c.pts[i] <= (s.end if s.end is not None else np.inf)
            self.g_pv_md.content = (f"image {i} · t = {c.pts[i]:.2f} s · {'inside' if inside else 'outside'} the range "
                                    "· no camera poses (images only)")
            return
        if self.kind == "rosbag":  # the camera pose from TF / odometry, and its frustum in 3D
            inside = s.start <= c.pts[i] <= (s.end if s.end is not None else np.inf)
            if not c.valid_pose[i]:
                self.g_pv_md.content = f"image {i} · t = {c.pts[i]:.2f} s · **no pose at this stamp**"
                return
            p = c.centers[i]
            where = "inside" if inside else "outside"
            self.g_pv_md.content = (f"image {i} · t = {c.pts[i]:.2f} s · {where} the range  \n"
                                    f"camera at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m in `{c.ros['world_frame']}`")
            if self.g_show.value:
                self.server.scene.add_camera_frustum(
                    "/ingest/cursor", fov=np.radians(60), aspect=c.width / c.height, scale=self._cursor_size(1.0), color=(255, 255, 255),
                    wxyz=ex.rot_to_quat(c.T[i][:3, :3]), position=p, image=self.g_pv_image.image[::4, ::4].copy())
            return
        if self.kind == "video":  # no telemetry: frame and time only
            inside = s.start <= c.pts[i] <= (s.end if s.end is not None else np.inf)
            self.g_pv_md.content = f"frame {i} · t = {c.pts[i]:.2f} s · {'inside' if inside else 'outside'} the range"
            return
        tel = c.telemetry(self.settings.yaw_source)
        a = tel["a"][i]
        inside = s.start <= c.pts[i] <= (s.end if s.end is not None else np.inf)
        self.g_pv_md.content = (f"frame {i} · t = {c.pts[i]:.2f} s · {'inside' if inside else 'outside'} the range  \n"
                                f"alt {a[2]:.1f} m · speed {np.linalg.norm(a[4:7]):.1f} m/s · heading "
                                f"{tel['yaw'][i]:.0f}° · gimbal pitch {tel['pitch'][i]:.1f}°")
        if self.prev and self.g_show.value:
            g = tel["g"][i]
            R = ex.R_ENU_NED @ ex.r_ned_body(tel["yaw"][i], g[0], g[1]) @ ex.R_BODY_CAM
            sp = next((x for x in self.plan.sources if x.source is s), None)
            alt = a[2] + (sp.alt_offset if sp else 0.0)
            p = ingest.enu(a[0], a[1], alt, self.prev["origin"])[0]
            self.server.scene.add_camera_frustum("/ingest/cursor", fov=np.radians(60), aspect=c.width / c.height,
                                                 scale=self._cursor_size(3.0), color=(255, 255, 255), wxyz=ex.rot_to_quat(R),
                                                 position=p, image=self.g_pv_image.image[::4, ::4].copy())

    def _cursor_size(self, largest):
        """The preview frustum's size: `largest` over flights, 4 plan dots where the planned images stand close."""
        dot = getattr(self, "_dot", None)
        return largest if dot is None else min(largest, 4 * dot)

    def _set_range_here(self, start):
        k = self._preview_index()
        if k is None or k >= len(self._source_widgets) or not self._source_widgets[k]["ready"]:
            return
        r = self._source_widgets[k]["range"]
        a, b = r.value
        t = float(self.g_pv_time.value)
        r.value = (min(t, b - 0.5), b) if start else (a, max(t, a + 0.5))  # triggers _range_changed

    # ------------------------------------------------------------------ extraction
    def extract(self):
        if self.running or self.app.running:
            return
        if not self.plan or not self.plan.num_images:
            self.g_status.content = "⚪ nothing to extract: the plan selects no images"
            return
        if ingest.has_outputs(self.path) and not self.g_overwrite.value:  # also images of an interrupted extraction
            self._update_status()  # shows the Overwrite box
            self.g_status.content = "⚪ this project already has images: tick **Overwrite** to re-extract " \
                                    "(old outputs move to `previous/`)"
            return
        self.running = True
        self.token = jobs.Token()
        self.app.g_stop.disabled = False
        self.g_extract.disabled = True
        self.g_progress.visible, self.g_progress.value = True, 0.0
        threading.Thread(target=self._extract, daemon=True).start()

    def plan_now(self):
        """Replan at once (after the flight check changed offsets; the planner thread would take a moment)."""
        with self._lock:
            self._replan()

    def extract_now(self):
        """Extract synchronously (full pipeline on a project without images); raises on failure."""
        if not self.plan or not self.plan.num_images:
            raise RuntimeError("the Ingest plan selects no images: add a video and its range in the Ingest tab")
        self.running = True
        self.token = self.app._token  # the full pipeline's Stop applies
        self.g_extract.disabled = True
        self.g_progress.visible, self.g_progress.value = True, 0.0
        self._extract()  # resets running / buttons itself
        self.token.check()
        if self.app.project is None:
            raise RuntimeError("extraction failed (details in the Ingest tab)")

    def _extract(self):
        t0 = time.time()
        try:
            def progress(done, total, msg):
                self.token.check()  # Stop: abort between frames
                self.g_progress.value = 100.0 * done / max(total, 1)
                self.g_status.content = f"⏳ **extracting** {done}/{total} images · {msg} · {time.time() - t0:.0f} s"
            self.g_status.content = "⏳ **extracting** … decoding up to the first selected frame"
            self.be.run(self.path, self.sources, self.settings, progress=progress, overwrite=self.g_overwrite.value,
                        log=lambda m: print(m, flush=True))
            self.g_overwrite.value = False
            self.app.load_project()
            n = len(self.app.project.names)
            self.g_status.content = (f"✅ **extracted {n} images** in {time.time() - t0:.0f} s "
                                     f"· {time.strftime('%H:%M:%S')}  \nnext: Sparse tab → **Run SfM (all)**")
        except jobs.Cancelled:
            self.g_status.content = ("⏹ **extraction stopped**: the images are incomplete; tick **Overwrite** and "
                                     "extract again (the partial ones move to `previous/`)")
        except Exception as e:
            traceback.print_exc()
            self.g_status.content = f"❌ **extraction failed**: `{e}`"
        finally:
            self.running = False
            self._update_status()  # images left by a stop or a failure: the Overwrite box
            self.app.g_stop.disabled = not self.app.running
            self.g_extract.disabled = False
            self.g_progress.visible = False
            self._update_status()
            self.request_plan(edited=False)

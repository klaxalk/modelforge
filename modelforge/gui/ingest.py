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
import plotly.graph_objects as go
import viser
from plotly.subplots import make_subplots

from modelforge import jobs, videos
from modelforge.dji import extract as ex
from modelforge.dji import ingest
from modelforge.rosbag import bag as rbag
from modelforge.rosbag import ingest as ros_ingest
from modelforge.video import ingest as video_ingest

SAMPLING = {"dji": ("distance", "time"), "video": video_ingest.SAMPLING, "rosbag": ros_ingest.SAMPLING}
NO_INFO, IMAGE_FRAME = "(none: estimate intrinsics)", "(image header frame)"

PALETTE = [(70, 134, 251), (251, 185, 56), (27, 229, 181), (228, 70, 11), (164, 252, 60), (180, 90, 250),
           (250, 110, 180), (120, 200, 255)]
OUTSIDE_COLOR = (110, 110, 110)


def source_color(k):
    return PALETTE[k % len(PALETTE)]


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
        self.g_upload.visible = self.g_upload_md.visible = self.kind != "rosbag"  # bags are folders
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
        self.g_status.content = ""
        self.g_plan_md.content = ""
        self.g_plot.figure = go.Figure()
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
            self.g_video = gui.add_dropdown("Video", options=("(none found)",),
                                            hint="Videos of the library (data root, incl. data/videos/; project "
                                                 "folders are skipped; Videos tab to manage them). An .SRT sidecar "
                                                 "is optional: used if present for a quicker telemetry alignment.")
            self.g_video_path = gui.add_text("or path", initial_value="",
                                             hint="Any video path inside the container (repo = /workspace); "
                                                  "overrides the dropdown.")
            row = gui.add_button_group("", options=("Add as source", "Rescan"),
                                       hint="Add as source: the whole clip becomes a new source (narrow it with its "
                                            "range). Rescan: look for new videos.")
            row.on_click(lambda e: self.add_source() if e.target.value == "Add as source" else self.rescan())
            self.g_upload = gui.add_upload_button(
                "Upload video / SRT", icon=viser.Icon.UPLOAD,
                mime_type="video/mp4,video/quicktime,.mp4,.MP4,.mov,.MOV,.srt,.SRT",
                hint="Upload a DJI video (.MP4 / .MOV), and optionally its .SRT sidecar, from this computer into "
                     "the shared video library (data/videos/, managed in the Videos tab). The video carries all the "
                     "metadata; the .SRT only gives a quicker, firmer telemetry alignment. Large clips take a while: "
                     "progress shows below. An existing file is not overwritten.")
            self.g_upload.on_upload(lambda e: self._on_upload(e.target.value))
            self.g_upload_md = gui.add_markdown("")
        self.g_ros = gui.add_folder("ROS 2 bag")
        with self.g_ros:
            self.g_ros_md = gui.add_markdown("_add a bag as a source above_")
            self.g_ros_image = gui.add_dropdown("Image topic", options=("-",),
                                                hint="sensor_msgs/CompressedImage (preferred) or Image.")
            self.g_ros_info = gui.add_dropdown("Camera info", options=(NO_INFO,),
                                               hint="sensor_msgs/CameraInfo: known intrinsics, kept fixed in BA "
                                                    "(plumb_bob -> OPENCV, equidistant -> fisheye). None: estimated.")
            self.g_ros_pose = gui.add_dropdown("Pose source", options=("tf", "odometry"),
                                               hint="tf: the camera pose from the TF tree (world -> camera frame) at "
                                                    "each image stamp. odometry: an Odometry topic (its frame -> its "
                                                    "child), chained with TF to the world and to the camera.")
            self.g_ros_world = gui.add_dropdown("World frame", options=("-",),
                                                hint="The frame the poses are expressed in: the project's world.")
            self.g_ros_camera = gui.add_dropdown("Camera frame", options=(IMAGE_FRAME,),
                                                 hint="The camera's optical frame (x right, y down, z forward; ROS "
                                                      "*_optical_frame). Default: the image messages' frame_id.")
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
                      self.g_ros_odom, self.g_ros_voxel, self.g_ros_range):
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
            self.g_yaw = gui.add_dropdown("Yaw source", options=("aircraft", "gimbal"), initial_value=st.yaw_source,
                                          hint="Camera heading from the aircraft yaw (NEO2: the gimbal cannot "
                                               "yaw). gimbal only for drones whose gimbal yaws and reports a true "
                                               "heading.")
            for h in (self.g_sampling, self.g_num, self.g_spacing, self.g_rotw, self.g_window, self.g_yaw):
                h.on_update(lambda _: self._settings_changed())
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
        self.g_plot = gui.add_plotly(go.Figure(), aspect=1.0)
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
        found = list(dict.fromkeys(found))
        self.g_video.options = tuple(found) or ("(none found)",)

    def _on_upload(self, up):
        def work():
            try:
                path = videos.save_upload(self.app.root, up.name, up.content)
                rel = os.path.relpath(path)
                self.rescan()
                if rel in self.g_video.options:
                    self.g_video.value = rel
                hint = "" if rel.lower().endswith(".srt") else (
                    "; then **Add as source**" + ("" if ingest.find_srt(path) else
                                                  " (no .SRT: fine, the alignment is estimated from the video, "
                                                  "about a minute once per clip)"))
                self.g_upload_md.content = f"✅ uploaded `{rel}` ({len(up.content) / 1e6:.0f} MB){hint}"
                self.app.videos.refresh()
            except (OSError, ValueError) as e:
                self.g_upload_md.content = f"❌ upload failed: `{e}`"
        threading.Thread(target=work, daemon=True).start()

    def add_source(self, video=None):
        v = video or self.g_video_path.value.strip() or self.g_video.value
        if not v or v == "(none found)" or not os.path.exists(v):
            self.g_status.content = f"❌ video not found: `{v}`"
            return
        if self.kind == "rosbag" and not rbag.is_bag(v):
            self.g_status.content = f"❌ not a ROS 2 bag (a folder with metadata.yaml, or an .mcap file): `{v}`"
            return
        if self.kind != "rosbag" and rbag.is_bag(v):
            self.g_status.content = f"❌ `{v}` is a ROS 2 bag: it needs a project of kind ROS 2 bag"
            return
        self.sources.append(ingest.Source(v))
        self.g_video_path.value = ""
        self._rebuild_sources()
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
                f = gui.add_folder(f"{k + 1}. {os.path.basename(s.video)}")
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
                    btn = gui.add_button_group("", options=("Preview", "Add piece", "Remove"),
                                               hint="Preview: show this source in Frame preview. Add piece: another "
                                                    "range of the same video. Remove: drop the source.")
                rng.on_update(lambda _, k=k: self._range_changed(k))
                en.on_update(lambda _, k=k: self._enabled_changed(k))
                lead.on_update(lambda _, k=k: self._lead_changed(k))
                lead.visible = self.kind == "dji"
                btn.on_click(lambda e, k=k: self._source_action(k, e.target.value))
                self._source_handles.append(f)
                self._source_widgets.append({"md": md, "range": rng, "enabled": en, "lead": lead, "ready": False})
        self.g_sources_md.content = "" if self.sources else "Add a video above. Each source is a video and a time " \
            "range; add the same video again (\"Add piece\") to take several pieces."
        self.g_pv_source.options = tuple(f"{k + 1}. {os.path.basename(s.video)}" for k, s in enumerate(self.sources)) \
            or ("-",)

    def _source_action(self, k, action):
        if action == "Remove":
            del self.sources[k]
            self._rebuild_sources()
        elif action == "Add piece":
            s = self.sources[k]
            self.sources.insert(k + 1, ingest.Source(s.video, s.start, s.end, s.enabled, s.lead, s.srt))
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

    # ------------------------------------------------------------------ ROS 2 bags
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
        self._ros_bag = path

    def _ros_changed(self):
        if self._loading:
            return
        ros = {"image_topic": self.g_ros_image.value, "pose_source": self.g_ros_pose.value,
               "info_topic": "" if self.g_ros_info.value == NO_INFO else self.g_ros_info.value,
               "world_frame": self.g_ros_world.value,
               "camera_frame": "" if self.g_ros_camera.value == IMAGE_FRAME else self.g_ros_camera.value,
               "odom_topic": "" if self.g_ros_odom.value == "-" else self.g_ros_odom.value,
               "cloud_topics": [t for t, cb in self._cloud_boxes.items() if cb.value],
               "voxel": float(self.g_ros_voxel.value), "max_range": float(self.g_ros_range.value)}
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
            self._update_plot()
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
            extra = {"dji": f" · lead {c.lead:g} frames ({c.align['method']})",
                     "rosbag": f" · poses from {c.align['method']}: {int(c.valid_pose.sum())}/{c.n} images"
                     if self.kind == "rosbag" else ""}.get(self.kind, "")
            lines = [f"{c.duration:.0f} s, {c.width}×{c.height} @ {c.fps:.2f} fps" + extra]
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
        if self.kind == "video" and r.get("mode") == "motion" and r["spacing"] > 0.25:
            md.append("⚠️ consecutive images move more than a quarter image width apart: SfM without metadata may "
                      "not connect them (more images or a shorter range)")
        if r.get("rot_weight_m_per_deg") is not None and r["mode"] == "distance":
            md.append(f"path {r['path_m']:.0f} m + rotation {r['rotation_deg']:.0f}° × "
                      f"{r['rot_weight_m_per_deg']:.2f} m/°")
        md += [f"⚠️ {w}" for w in plan.warnings]
        self.g_plan_md.content = "  \n".join(md)
        self._draw_scene()
        self._update_plot()
        self._update_status()
        self._preview_source_changed(keep_time=True)

    def _update_status(self):
        n_img = len(self.app.project.names) if self.app.project else 0
        lines = []
        if n_img:
            extracted = self.app.project.meta.get("plan")
            same = extracted == {"sources": [asdict(s) for s in self.sources], "settings": asdict(self.settings)}
            lines.append(f"{n_img} images extracted" + ("" if same or extracted is None else
                         " · **the plan differs from the extracted images** (re-extract with overwrite)"))
        else:
            lines.append("no images yet: add sources, check the plan, then extract")
        self.g_md.content = "  \n".join(lines)
        self.g_overwrite.visible = bool(n_img)

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
        for sp, pv in zip(self.plan.sources, self.prev["sources"]):
            t, v = pv["track"].astype(np.float32), pv["track_valid"]
            seg = np.stack([t[:-1], t[1:]], 1)
            inside = v[:-1] & v[1:]
            segs_in.append(seg[inside])
            cols_in.append(np.tile(np.array(source_color(sp.index), np.uint8), (int(inside.sum()), 2, 1)))
            segs_out.append(seg[~inside])
            pts.append(pv["samples"])
            pcols.append(np.tile(np.array(source_color(sp.index), np.uint8), (len(pv["samples"]), 1)))
        segs_in = np.concatenate(segs_in) if segs_in else np.zeros((0, 2, 3), np.float32)
        cols_in = np.concatenate(cols_in) if cols_in else np.zeros((0, 2, 3), np.uint8)
        segs_out = np.concatenate(segs_out) if segs_out else np.zeros((0, 2, 3), np.float32)
        scene.add_line_segments("/ingest/selected", segs_in, colors=cols_in, thickness=2.5, thickness_units="screen")
        scene.add_line_segments("/ingest/outside", segs_out, colors=OUTSIDE_COLOR, thickness=1.0,
                                thickness_units="screen")
        p = np.concatenate(pts).astype(np.float32) if pts else np.zeros((1, 3), np.float32)
        c = np.concatenate(pcols) if pcols else np.zeros((1, 3), np.uint8)
        scene.add_point_cloud("/ingest/samples", points=p, colors=c, point_size=1.2, precision="float32",
                              point_shape="circle")

    def _update_plot(self):
        if not self.plan or not self.plan.sources:
            self.g_plot.figure = go.Figure()
            return
        if self.kind == "video":
            self._update_plot_video()
            return
        if self.kind == "rosbag":
            self._update_plot_rosbag()
            return
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=("altitude [m] · ● planned images", "speed [m/s]", "gimbal pitch [deg]"))
        for sp in self.plan.sources:
            c, tel = sp.clip, sp.clip.telemetry(self.settings.yaw_source)
            idx = np.arange(0, c.n, 6)
            t = sp.time_base + c.pts[idx]
            a = tel["a"][idx]
            col = "rgb(%d,%d,%d)" % source_color(sp.index)
            sel = np.where(sp.valid[idx], 1.0, np.nan)
            name = f"{sp.index + 1}. {sp.prefix}"
            for row, y in ((1, a[:, 2] + sp.alt_offset), (2, np.linalg.norm(a[:, 4:7], axis=1)),
                           (3, tel["pitch"][idx])):
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="rgb(150,150,150)", width=1),
                                         showlegend=False, hoverinfo="skip"), row, 1)
                fig.add_trace(go.Scatter(x=t, y=y * sel, mode="lines", line=dict(color=col, width=2), name=name,
                                         showlegend=row == 1), row, 1)
            if sp.windows:
                k = np.array([(w0 + w1) // 2 for w0, w1 in sp.windows])
                fig.add_trace(go.Scatter(x=sp.time_base + c.pts[k], y=tel["a"][k, 2] + sp.alt_offset, mode="markers",
                                         marker=dict(size=4, color=col), showlegend=False), 1, 1)
        fig.update_xaxes(title_text="time [s] (project timeline)", row=3, col=1)
        fig.update_layout(margin=dict(l=40, r=10, t=30, b=30), height=560,
                          legend=dict(orientation="h", y=-0.12), template=self.app.plot_template())
        self.g_plot.figure = fig

    def _update_plot_video(self):
        """No telemetry: the sampling curve (cumulative image motion, or time) with the planned images."""
        fig = go.Figure()
        for sp in self.plan.sources:
            c = sp.clip
            idx = np.arange(0, c.n, 6)
            t = sp.time_base + c.pts[idx]
            col = "rgb(%d,%d,%d)" % source_color(sp.index)
            sel = np.where(sp.valid[idx], 1.0, np.nan)
            fig.add_trace(go.Scatter(x=t, y=sp.s[idx], mode="lines", line=dict(color="rgb(150,150,150)", width=1),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=t, y=sp.s[idx] * sel, mode="lines", line=dict(color=col, width=2),
                                     name=f"{sp.index + 1}. {sp.prefix}"))
            if sp.windows:
                k = np.array([(w0 + w1) // 2 for w0, w1 in sp.windows])
                fig.add_trace(go.Scatter(x=sp.time_base + c.pts[k], y=sp.s[k], mode="markers",
                                         marker=dict(size=5, color=col), showlegend=False))
        unit = "cumulative image motion [image widths]" if self.settings.sampling == "motion" else "time [s]"
        fig.update_layout(title=f"{unit} · ● planned images", margin=dict(l=40, r=10, t=40, b=30), height=360,
                          legend=dict(orientation="h", y=-0.2), template=self.app.plot_template(),
                          xaxis_title="time [s] (project timeline)")
        self.g_plot.figure = fig

    def _update_plot_rosbag(self):
        """Camera height and speed (from the TF / odometry poses) with the planned images."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("camera height (world z) [m] · ● planned images", "speed [m/s]"))
        for sp in self.plan.sources:
            c = sp.clip
            t = sp.time_base + c.pts
            z = c.centers[:, 2]
            v = np.r_[0.0, np.linalg.norm(np.diff(c.centers, axis=0), axis=1) / np.maximum(np.diff(c.stamps), 1e-6)]
            col = "rgb(%d,%d,%d)" % source_color(sp.index)
            sel = np.where(sp.valid, 1.0, np.nan)
            for row, y in ((1, z), (2, v)):
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="rgb(150,150,150)", width=1),
                                         showlegend=False, hoverinfo="skip"), row, 1)
                fig.add_trace(go.Scatter(x=t, y=y * sel, mode="lines", line=dict(color=col, width=2),
                                         name=f"{sp.index + 1}. {sp.prefix}", showlegend=row == 1), row, 1)
            if sp.windows:
                k = np.array([(w0 + w1) // 2 for w0, w1 in sp.windows])
                fig.add_trace(go.Scatter(x=sp.time_base + c.pts[k], y=z[k], mode="markers",
                                         marker=dict(size=5, color=col), showlegend=False), 1, 1)
        fig.update_xaxes(title_text="time [s] (project timeline)", row=2, col=1)
        fig.update_layout(margin=dict(l=40, r=10, t=30, b=30), height=440, legend=dict(orientation="h", y=-0.15),
                          template=self.app.plot_template())
        self.g_plot.figure = fig

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
        if not keep_time or t > c.duration:
            self.g_pv_time.value = float(min(max(s.start, 0.0), c.duration))
        self.request_preview()

    def request_preview(self):
        k = self._preview_index()
        if k is None or k >= len(self.sources):
            return
        with self._preview_lock:
            self._preview_req = (k, float(self.g_pv_time.value))

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
                self._show_preview(*req)
            except Exception as e:
                traceback.print_exc()
                self.g_pv_md.content = f"preview failed: `{e}`"

    def _show_preview(self, k, t):
        s = self.sources[k]
        c = self._load_clip(s)
        i = int(np.clip(np.searchsorted(c.pts, t), 0, c.n - 1))
        self.g_pv_image.image = c.frame_image(c.pts[i])
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
                    "/ingest/cursor", fov=np.radians(60), aspect=c.width / c.height, scale=1.0, color=(255, 255, 255),
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
                                                 scale=3.0, color=(255, 255, 255), wxyz=ex.rot_to_quat(R),
                                                 position=p, image=self.g_pv_image.image[::4, ::4].copy())

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
        if self.app.project and not self.g_overwrite.value:
            self.g_status.content = "⚪ this project already has images: tick **Overwrite** to re-extract " \
                                    "(old outputs move to `previous/`)"
            return
        self.running = True
        self.token = jobs.Token()
        self.app.g_stop.disabled = False
        self.g_extract.disabled = True
        self.g_progress.visible, self.g_progress.value = True, 0.0
        threading.Thread(target=self._extract, daemon=True).start()

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
            self.app.g_stop.disabled = not self.app.running
            self.g_extract.disabled = False
            self.g_progress.visible = False
            self._update_status()
            self.request_plan(edited=False)

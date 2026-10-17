"""GUI Ingest tab: pick videos, cut them into sources (time ranges), plan sampling, extract.

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

from modelforge.dji import extract as ex
from modelforge.dji import ingest

PALETTE = [(70, 134, 251), (251, 185, 56), (27, 229, 181), (228, 70, 11), (164, 252, 60), (180, 90, 250),
           (250, 110, 180), (120, 200, 255)]
OUTSIDE_COLOR = (110, 110, 110)


def source_color(k):
    return PALETTE[k % len(PALETTE)]


class IngestPanel:
    def __init__(self, app, gui):
        self.app, self.gui, self.server = app, gui, app.server
        self.path = app.path
        self.sources, self.settings = ingest.load_plan(self.path)
        if not self.sources:
            self.sources, self.settings = self._plan_from_poses()
        self.plan, self.prev = None, None
        self.running = False
        self.edited = False  # ingest.json is written only once the user changes the plan
        self._lock = threading.RLock()  # source widgets: GUI callbacks vs. the planner thread
        self._dirty = threading.Event()
        self._preview_req = None
        self._preview_lock = threading.Lock()
        self._source_handles = []
        self._build()
        threading.Thread(target=self._planner, daemon=True).start()
        threading.Thread(target=self._previewer, daemon=True).start()
        self.request_plan(edited=False)

    def _plan_from_poses(self):
        """Older single-video projects have no ingest.json: rebuild the plan from poses.json."""
        meta = self.app.project.meta if self.app.project else None
        if not meta:
            return [], ingest.Settings()
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
            self.g_video = gui.add_dropdown("Video", options=("(none found)",))
            self.g_video_path = gui.add_text("or path", initial_value="")
            row = gui.add_button_group("", options=("Add as source", "Rescan"))
            row.on_click(lambda e: self.add_source() if e.target.value == "Add as source" else self.rescan())
        self.g_sources = gui.add_folder("Sources")
        with self.g_sources:
            self.g_sources_md = gui.add_markdown("")
        with gui.add_folder("Sampling"):
            st = self.settings
            self.g_sampling = gui.add_dropdown("Sampling", options=("distance", "time"), initial_value=st.sampling,
                                               hint="distance: travelled distance + view-rotation credit")
            self.g_num = gui.add_number("Images (all sources)", initial_value=st.num_images, min=1, step=1)
            self.g_spacing = gui.add_number("Spacing (0 = from images)", initial_value=st.spacing or 0.0, min=0.0,
                                            step=0.1, hint="m (distance) or s (time)")
            self.g_rotw = gui.add_number("Rotation credit m/° (−1 = auto)",
                                         initial_value=-1.0 if st.rot_weight is None else st.rot_weight,
                                         min=-1.0, step=0.01)
            self.g_window = gui.add_slider("Sharpest-frame window", min=0.1, max=1.0, step=0.05,
                                           initial_value=st.window)
            self.g_yaw = gui.add_dropdown("Yaw source", options=("aircraft", "gimbal"), initial_value=st.yaw_source,
                                          hint="aircraft unless the gimbal can yaw (not the NEO2)")
            for h in (self.g_sampling, self.g_num, self.g_spacing, self.g_rotw, self.g_window, self.g_yaw):
                h.on_update(lambda _: self._settings_changed())
        self.g_plan_md = gui.add_markdown("")
        with gui.add_folder("Frame preview"):
            self.g_pv_source = gui.add_dropdown("Source", options=("-",))
            self.g_pv_time = gui.add_slider("Time [s]", min=0.0, max=1.0, step=0.1, initial_value=0.0)
            self.g_pv_image = gui.add_image(np.zeros((36, 64, 3), np.uint8))
            self.g_pv_md = gui.add_markdown("")
            row = gui.add_button_group("Set range", options=("Start here", "End here"))
            row.on_click(lambda e: self._set_range_here(e.target.value == "Start here"))
            self.g_pv_source.on_update(lambda _: self._preview_source_changed())
            self.g_pv_time.on_update(lambda _: self.request_preview())
        self.g_show = gui.add_checkbox("Show flight plan in 3D", initial_value=True)
        self.g_show.on_update(lambda _: self._draw_scene())
        self.g_plot = gui.add_plotly(go.Figure(), aspect=1.0)
        self.g_overwrite = gui.add_checkbox("Overwrite existing images", initial_value=False,
                                            hint="old images and everything derived from them move to previous/")
        self.g_extract = gui.add_button("Extract images + priors", icon=viser.Icon.PHOTO)
        self.g_extract.on_click(lambda _: self.extract())
        self.g_progress = gui.add_progress_bar(value=0.0, visible=False)
        self.g_status = gui.add_markdown("")
        self.rescan()
        self._rebuild_sources()

    def rescan(self):
        roots = [os.path.dirname(os.path.abspath(self.path)), os.path.join(os.getcwd(), "data")]
        found = []
        for r in dict.fromkeys(roots):
            if os.path.isdir(r):
                found += [os.path.relpath(v) for v in ingest.find_videos(r)]
        found = list(dict.fromkeys(found))
        self.g_video.options = tuple(found) or ("(none found)",)

    def add_source(self):
        v = self.g_video_path.value.strip() or self.g_video.value
        if not v or v == "(none found)" or not os.path.exists(v):
            self.g_status.content = f"video not found: `{v}`"
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
                                               initial_value=(s.start, end), min_range=0.5, disabled=True)
                    en = gui.add_checkbox("Enabled", initial_value=s.enabled)
                    lead = gui.add_number("Lead [frames] (−1 = auto)", initial_value=-1.0 if s.lead is None else s.lead,
                                          min=-1.0, step=0.1)
                    btn = gui.add_button_group("", options=("Preview", "Add piece", "Remove"))
                rng.on_update(lambda _, k=k: self._range_changed(k))
                en.on_update(lambda _, k=k: self._enabled_changed(k))
                lead.on_update(lambda _, k=k: self._lead_changed(k))
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
        self.settings = ingest.Settings(
            sampling=self.g_sampling.value, num_images=max(1, int(self.g_num.value)),
            spacing=float(self.g_spacing.value) or None,
            rot_weight=None if self.g_rotw.value < 0 else float(self.g_rotw.value),
            window=float(self.g_window.value), yaw_source=self.g_yaw.value, quality=self.settings.quality)
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
                w["md"].content = "analysing (decoding telemetry, estimating lead) …"
                c = ingest.load_clip(s.video, s.lead, s.srt)
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
        plan = ingest.make_plan(self.sources, self.settings)
        origin = self.app.project.meta.get("enu_origin_lla") if self.app.project else None
        self.plan, self.prev = plan, ingest.preview(plan, self.settings.yaw_source, origin=origin)
        if self.edited:
            ingest.save_plan(self.path, self.sources, self.settings)
        by_source = {id(sp.source): sp for sp in plan.sources}
        for k, s in enumerate(self.sources):
            if k >= len(self._source_widgets):
                break
            c = ingest.load_clip(s.video, s.lead, s.srt)
            sp = by_source.get(id(s))
            lines = [f"{c.duration:.0f} s, {c.width}×{c.height} @ {c.fps:.2f} fps · lead {c.lead:g} frames "
                     f"({c.align['method']})"]
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
              f"{r['spacing']:.2f} {r['unit']}"]
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
                          legend=dict(orientation="h", y=-0.12))
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
        c = ingest.load_clip(s.video, s.lead, s.srt)
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
        c = ingest.load_clip(s.video, s.lead, s.srt)
        i = int(np.clip(np.searchsorted(c.pts, t), 0, c.n - 1))
        self.g_pv_image.image = c.frame_image(c.pts[i])
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
            self.g_status.content = "nothing to extract"
            return
        if self.app.project and not self.g_overwrite.value:
            self.g_status.content = "this project already has images: tick **Overwrite** to re-extract " \
                                    "(old outputs move to `previous/`)"
            return
        self.running = True
        self.g_extract.disabled = True
        self.g_progress.visible, self.g_progress.value = True, 0.0
        threading.Thread(target=self._extract, daemon=True).start()

    def _extract(self):
        t0 = time.time()
        try:
            def progress(done, total, msg):
                self.g_progress.value = 100.0 * done / max(total, 1)
                self.g_status.content = f"extracting {done}/{total} · {msg} · {time.time() - t0:.0f} s"
            self.g_status.content = "extracting … (decoding up to the first selected frame)"
            ingest.run(self.path, self.sources, self.settings, progress=progress, overwrite=self.g_overwrite.value,
                       log=lambda m: print(m, flush=True))
            self.g_overwrite.value = False
            self.app.load_project()
            self.g_status.content = f"extracted in {time.time() - t0:.0f} s · next: Run tab → **Run SfM (all)**"
        except Exception as e:
            traceback.print_exc()
            self.g_status.content = f"extraction failed: `{e}`"
        finally:
            self.running = False
            self.g_extract.disabled = False
            self.g_progress.visible = False
            self._update_status()
            self.request_plan(edited=False)

"""modelforge web GUI (viser): ingest videos, run stages live, inspect snapshots and products.

Usage (in the dev container; port 8080 is published by compose.yaml):
    python -m modelforge.gui.app data/site [--port 8080]
The project folder may be new or empty: the Ingest tab builds it from one or more videos.
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

from modelforge import dense, pipeline
from modelforge.gui import data as gdata
from modelforge.gui.data import Project
from modelforge.gui.ingest import IngestPanel, source_color

COLOR_BY = ("position deviation", "rotation deviation", "reprojection error", "time", "source", "uniform")
SOURCE_SWATCH = ("🟦", "🟨", "🟩", "🟥", "🟢", "🟪", "🩷", "🔵")  # approximates gui.ingest.PALETTE
UNITS = {"position deviation": "m", "rotation deviation": "deg", "reprojection error": "px", "time": "s"}
PRIOR_COLOR = (150, 150, 150)
SELECTED_COLOR = (255, 255, 255)
_ANCHORS = np.array([[48, 18, 59], [70, 134, 251], [27, 229, 181], [164, 252, 60],
                     [251, 185, 56], [228, 70, 11], [122, 4, 3]], float)  # turbo-like


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
    def __init__(self, project_path, port=8080):
        self.path = project_path
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
        self._build_gui()
        if self.project:
            names = self.project.snapshot_names()
            self.load_snapshot("refined" if "refined" in names else names[-1])
        self._refresh_dense()
        self.server.on_client_connect(self._on_connect)

    # ---------------------------------------------------------------- GUI
    def _build_gui(self):
        gui = self.server.gui
        self.g_header = gui.add_markdown(self._header())
        tabs = gui.add_tab_group()
        if self.project is None:  # new project: start where the work starts
            with tabs.add_tab("Ingest", viser.Icon.VIDEO):
                self.ingest = IngestPanel(self, gui)

        with tabs.add_tab("View", viser.Icon.EYE):
            names = tuple(self.project.snapshot_names()) if self.project else ("priors",)
            self.g_snapshot = gui.add_dropdown("Snapshot", options=names)
            self.g_snapshot.on_update(lambda _: self.load_snapshot(self.g_snapshot.value))
            self.g_stats = gui.add_markdown("")
            with gui.add_folder("Layers"):
                self.g_points = gui.add_checkbox("Points", initial_value=True)
                self.g_point_size = gui.add_slider("Point size", min=0.01, max=0.5, step=0.01, initial_value=0.08)
                self.g_cams = gui.add_checkbox("Cameras", initial_value=True)
                self.g_thumbs = gui.add_checkbox("Thumbnails", initial_value=True)
                self.g_priors = gui.add_checkbox("Prior cameras", initial_value=False)
                self.g_corr = gui.add_checkbox("Prior → refined", initial_value=True)
                self.g_traj = gui.add_checkbox("Trajectories", initial_value=True)
                self.g_scale = gui.add_slider("Camera size", min=0.2, max=5.0, step=0.1, initial_value=1.0)
            with gui.add_folder("Dense / splat"):
                self.g_dense_pts = gui.add_checkbox("Dense points", initial_value=False, disabled=True)
                self.g_mesh = gui.add_dropdown("Mesh", options=("off",), initial_value="off", disabled=True)
                self.g_splat = gui.add_checkbox("Splat (ENU)", initial_value=False, disabled=True)
                self.g_dense_md = gui.add_markdown("")
                self.g_dense_pts.on_update(lambda _: self._draw_dense_points())
                self.g_mesh.on_update(lambda _: self._draw_mesh())
                self.g_splat.on_update(lambda _: self._draw_splat())
            self.g_color = gui.add_dropdown("Color cameras by", options=COLOR_BY, initial_value="position deviation")
            self.g_legend = gui.add_markdown("")
            for h in (self.g_points, self.g_point_size):
                h.on_update(lambda _: self._draw_points())
            for h in (self.g_cams, self.g_thumbs, self.g_priors, self.g_scale):
                h.on_update(lambda _: self._draw_cameras())
            self.g_color.on_update(lambda _: self._apply_colors())
            for h in (self.g_corr, self.g_traj):
                h.on_update(lambda _: self._draw_lines())

        with tabs.add_tab("Inspect", viser.Icon.PHOTO):
            self.g_inspect_md = gui.add_markdown("Click a camera to inspect it.")
            self.g_gain = gui.add_slider("Residual ×", min=1.0, max=50.0, step=1.0, initial_value=10.0)
            self.g_gain.on_update(lambda _: self.select(self.selected) if self.selected is not None else None)
            self.g_image = gui.add_image(np.zeros((54, 96, 3), np.uint8), label="observations")

        with tabs.add_tab("Plots", viser.Icon.CHART_LINE):
            self.g_plot = gui.add_plotly(go.Figure(), aspect=0.9)

        with tabs.add_tab("Run", viser.Icon.PLAYER_PLAY):
            gui.add_markdown("Stages run in the background; poses update live during BA.")
            self.g_buttons = {}
            b = gui.add_button("Run SfM (all)", hint="features → pairs → match → refine → export")
            b.on_click(lambda _: self.run_stage("sfm"))
            self.g_buttons["sfm"] = b
            for stage in pipeline.SFM_STAGES:
                b = gui.add_button(f"Run {stage}" + (" (live)" if stage == "refine" else ""))
                b.on_click(lambda _, s=stage: self.run_stage(s))
                self.g_buttons[stage] = b
            with gui.add_folder("Dense (COLMAP MVS)"):
                gui.add_markdown("PatchMatch stereo: " + ("CUDA available" if dense.has_cuda() else
                                 "**needs CUDA** - run the GUI in the CUDA container (`MODELFORGE_GPU=1 scripts/dev`)"))
                self.g_mesh_method = gui.add_dropdown("Mesh method", options=dense.MESH_METHODS + ("both",),
                                                      initial_value="delaunay")
                b = gui.add_button("Run dense (all)", disabled=not dense.has_cuda())
                b.on_click(lambda _: self.run_stage("dense"))
                self.g_buttons["dense"] = b
                for stage in pipeline.DENSE_STAGES:
                    b = gui.add_button(f"Run {stage}", disabled=(stage == "stereo" and not dense.has_cuda()))
                    b.on_click(lambda _, s=stage: self.run_stage(s))
                    self.g_buttons[stage] = b
            with gui.add_folder("Gaussian splat (nerfstudio)", expand_by_default=False):
                gui.add_markdown("Train on the host: `scripts/splat <project>` (nerfstudio container, GPU). "
                                 "It writes `splat/splat_enu.ply`, aligned with the mesh.")
                gui.add_button("Reload dense / splat").on_click(lambda _: self._refresh_dense())
            self.g_progress = gui.add_progress_bar(value=0.0, visible=False)
            self.g_status = gui.add_markdown("idle")

        if self.project is not None:
            with tabs.add_tab("Ingest", viser.Icon.VIDEO):
                self.ingest = IngestPanel(self, gui)

    def _header(self):
        if self.project is None:
            return f"### modelforge\n`{self.path}` · no images yet"
        n_src = len(set(self.project.source.tolist()))
        return (f"### modelforge\n`{self.path}` · {len(self.project.names)} images"
                + (f" from {n_src} sources" if n_src > 1 else ""))

    def load_project(self):
        """(Re)load PROJECT after extraction: new images, no snapshots yet."""
        with self.lock:
            for h in list(self.cam_handles.values()) + list(self.prior_handles.values()):
                h.remove()
            self.cam_handles, self.prior_handles, self.selected, self.snap = {}, {}, None, None
            self.project = Project(self.path)
            self.g_header.content = self._header()
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
            return
        if self.project is None:
            self.g_status.content = "no images yet: extract them in the Ingest tab first"
            return
        self.running = True
        for b in self.g_buttons.values():
            b.disabled = True
        self.g_progress.visible = True
        self.g_progress.value = 0.0
        threading.Thread(target=self._run, args=(stage,), daemon=True).start()

    def _run(self, stage):
        project, t0 = self.path, time.time()
        self.g_status.content = f"running **{stage}** …"
        try:
            report = pipeline.Report(project)
            pri = self.project.priors
            if stage == "sfm" or stage in pipeline.SFM_STAGES:
                seq = pipeline.SFM_STAGES if stage == "sfm" else (stage,)
                if stage != "refine":
                    self._ticker(stage, t0)
                for k, st in enumerate(seq):
                    self._stage_label = f"{st} ({k + 1}/{len(seq)})" if len(seq) > 1 else st
                    if len(seq) > 1:
                        self.g_progress.value = 100.0 * k / len(seq)
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
            elif stage == "dense" or stage in pipeline.DENSE_STAGES:
                seq = pipeline.DENSE_STAGES if stage == "dense" else (stage,)
                m = self.g_mesh_method.value
                methods = dense.MESH_METHODS if m == "both" else (m,)
                self._ticker(stage, t0)
                for k, st in enumerate(seq):
                    self.g_progress.value = 100.0 * k / len(seq)
                    self._stage_label = st
                    pipeline.stage_dense(project, report, st, methods)
                self._refresh_dense()
            self.g_status.content = f"**{stage}** done in {time.time() - t0:.0f} s"
            self.refresh_snapshot_list(select="refined" if stage in ("refine", "sfm") else None)
        except Exception as e:  # surface failures in the GUI instead of killing the thread silently
            traceback.print_exc()
            self.g_status.content = f"**{stage}** failed: `{e}`"
        finally:
            self.running = False
            for b in self.g_buttons.values():
                b.disabled = False
            self.g_progress.visible = False

    def _ticker(self, stage, t0):
        self._stage_label = stage

        def tick():
            while self.running:
                if not self._stage_label.startswith("refine"):  # refine reports its own progress
                    self.g_status.content = f"running **{self._stage_label}** … {time.time() - t0:.0f} s"
                time.sleep(1.0)
        threading.Thread(target=tick, daemon=True).start()

    def _on_refine_event(self, ev):
        phase, rnd, rounds = ev["phase"], ev.get("round", 0), ev.get("rounds", 1)
        if phase == "triangulate":
            self.g_status.content = f"round {rnd + 1}/{rounds}: triangulating …"
            self.g_progress.value = 100.0 * rnd / rounds
        elif phase == "ba":
            now = time.time()
            if now - self._last_live < 0.25:  # throttle scene updates
                return
            self._last_live = now
            self.g_status.content = (f"round {rnd + 1}/{rounds}: BA iteration {ev['iteration']}, "
                                     f"cost {ev['cost']:.4g}")
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
        self.g_dense_md.content = "available: " + (", ".join(have) if have else "nothing yet (Run tab)")
        self._draw_dense_points()
        self._draw_mesh()
        self._draw_splat()

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
    ap.add_argument("project")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    app = App(os.path.abspath(args.project) if not os.path.isabs(args.project) else args.project, args.port)
    print(f"modelforge GUI on http://localhost:{args.port}", flush=True)
    app.serve_forever()


if __name__ == "__main__":
    main()

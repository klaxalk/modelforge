"""Tie points tab and window (the user asked, 2026-10-25: human in the loop SfM; modelforge/sfm/tie_points.py).

Workflow: after a first SfM, find a spot where flights did not merge; Ctrl + click it in the 3D view: the photos that
see it are listed, nearest first, by flight; open one: the Photo window's Tie point tab shows it whole, a click
zooms in around that spot (ZOOM x), a second click records the spot's pixel. Two or more clicks per flight (or pass) on the same
physical spot make the tie point; Refine again (refine + export) uses them. Markers: yellow = the Ctrl + clicked
spot, cyan = where the tie point's clicks so far put it, green = this photo's own click.

Clicks on the photo reach the server through a hidden text input (docker/patch_viser_page.py: img[data-mf-click]
reports the click's position as fractions of the image).
"""
import base64
import html
import os
import threading

import cv2
import numpy as np
import viser

from modelforge.sfm import tie_points as tp

WINDOW_WIDTH = 960  # px of the Photo window while it shows the Tie point tab
TAB = "Tie point"  # its tab in the Photo window
ZOOM = 5  # the zoomed view shows 1 / ZOOM of the photo's width
PER_FLIGHT = 8  # candidate photos listed per flight


class TiePointsPanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self._rec, self._rec_key, self._lock = None, None, threading.Lock()
        self.spot = None  # the last Ctrl + clicked 3D spot
        self.candidates = []  # (label, image name)
        self.view = None  # the open photo: {"name", "box": (x0, y0, x1, y1) full-res, "zoomed"}
        gui.add_markdown(
            "Tie the same spot together across flights where the SfM left them apart (a doubled roof edge, "
            "copies of the ground). **1.** Ctrl + click the spot in the 3D view. **2.** Open a photo below: in the "
            "**Tie point** window click the spot (the first click zooms in, the second records it). **3.** Do that "
            "in **2+ photos of each flight** (or pass). **4.** Refine again. The tie points move whole flights and "
            "passes (the per-flight offset and drift), so a few good ones are enough.")
        self.g_point = gui.add_dropdown("Tie point", options=("(none)",), initial_value="(none)",
                                        hint="The tie point the clicks go to.")
        row = gui.add_button_group("", options=("＋ New", "🗑 Delete"),
                                   hint="New: a new tie point (T1, T2, …). Delete: the selected one with all its clicks.")
        row.on_click(lambda e: self.new() if e.target.value == "＋ New" else self.delete(self.g_point.value))
        self.g_point.on_update(lambda _: self.refresh_status())
        self.g_status = gui.add_markdown("")
        self.g_photos = gui.add_dropdown("Photos seeing the spot", options=("(Ctrl + click a spot first)",),
                                         initial_value="(Ctrl + click a spot first)",
                                         hint="The registered photos the Ctrl + clicked spot (or the tie point) "
                                              "projects into, nearest first, per flight.")
        row = gui.add_button_group("", options=("◀", "Open photo", "▶"),
                                   hint="Open the photo in the Photo window (its Tie point tab); ◀ ▶ the previous / next one.")
        row.on_click(lambda e: self._photo_button(e.target.value))
        b = gui.add_button("Refine again (refine + export)", icon=viser.Icon.REFRESH,
                           hint="Run the refine with the tie points (features and matches are kept), then the export. "
                                "The History line lists how well each tie point agrees afterwards (px).")
        b.on_click(lambda _: self.app.run_stage(("refine", "export")))
        self.g_errors = gui.add_markdown("")
        self.panel = None
        self.refresh()

    def make_window(self, panel):
        """The Tie point view: a tab of the Photo window (App.photo_panel, the user asked, 2026-10-25; it was a
        window of its own). App calls it outside the left panel's tabs."""
        app = self.app
        self.panel = panel
        with panel.add_tab(TAB, viser.Icon.TARGET):
            self.g_img = app.server.gui.add_html("")
            self.g_info = app.server.gui.add_markdown("")
            row = app.server.gui.add_button_group("", options=("Zoom out", "Remove this click", "Close"))
            row.on_click(lambda e: {"Zoom out": self._zoom_out, "Remove this click": self._remove_click,
                                    "Close": self.app.photo_tab}[e.target.value]())
            self.g_click = app.server.gui.add_text("tie click", initial_value="", visible=False)
            self.g_click.on_update(lambda e: self._clicked(e.target.value))

    # --- state ------------------------------------------------------------------------------------------------
    def _project(self):
        return self.app.path if self.app.path and os.path.exists(os.path.join(self.app.path, "poses.json")) else None

    def rec(self):
        """The refined model (cached while its files stay), None before an SfM."""
        import pycolmap
        d = os.path.join(self._project() or "", "sparse", "refined")
        f = os.path.join(d, "images.bin")
        if not os.path.exists(f):
            return None
        key = (d, os.path.getmtime(f))
        with self._lock:
            if key != self._rec_key:
                self._rec, self._rec_key = pycolmap.Reconstruction(d), key
            return self._rec

    def refresh(self):
        """On a project switch and after a refine."""
        project = self._project()
        pts = tp.load(project) if project else []
        names = [p["name"] for p in pts]
        cur = self.g_point.value
        self.g_point.options = tuple(names) or ("(none)",)
        self.g_point.value = cur if cur in names else (names[-1] if names else "(none)")
        rep = self.app.project.report().get("refine", {}).get("tie_points") if self.app.project else None
        self.g_errors.content = ("Last refine: " + ", ".join(
            f"**{k}** {v['error_px']} px ({v['clicks']} clicks)" for k, v in rep.items()) if rep else "")
        self.refresh_status()

    def refresh_status(self):
        project = self._project()
        p = next((p for p in (tp.load(project) if project else []) if p["name"] == self.g_point.value), None)
        if p is None:
            self.g_status.content = "_no tie point yet: **＋ New**_" if project else ""
            return
        pri = self.app.project.priors if self.app.project else {}
        by_src = {}
        for o in p["obs"]:
            s = getattr(pri.get(o["image"]), "source", 0)
            by_src.setdefault(s, []).append(o["image"])
        names = getattr(self.app.project, "source_names", [])
        lines = [f"**{p['name']}**: {len(p['obs'])} clicks · " + " · ".join(
            f"{names[s] if s < len(names) else f'flight {s + 1}'}: {len(v)}" for s, v in sorted(by_src.items()))
            if p["obs"] else f"**{p['name']}**: no clicks yet"]
        rec = self.rec()
        if rec is not None and len(p["obs"]) >= 2:  # per flight: its clicks alone (how good, how far apart)
            pf = tp.per_flight(rec, pri, p)
            for s, f in pf["flights"].items():
                flight = names[s] if s < len(names) else f"flight {s + 1}"
                note = ("" if f["clicks"] >= 2 and f["angle_deg"] >= tp.MIN_ANGLE_DEG else
                        f" ⚠ add a photo from another direction (≥ {tp.MIN_ANGLE_DEG:.0f}° between the clicks)"
                        if f["clicks"] >= 2 else " ⚠ needs 2+ clicks")
                lines.append(f"{flight}: {f['clicks']} clicks, {f['angle_deg']}° apart"
                             + (f", agree to {f['error_px']} px" if f["error_px"] is not None else "") + note)
            if pf["apart_m"] is not None:
                lines.append(f"the flights' points are **{pf['apart_m'] * 100:.0f} cm** apart here (current model)")
        self.g_status.content = "  \n".join(lines)

    def new(self):
        project = self._project()
        if project is None:
            return
        pts = tp.load(project)
        name = tp.new_name(pts)
        pts.append({"name": name, "obs": []})
        tp.save(project, pts)
        self.refresh()
        self.g_point.value = name

    def delete(self, name):
        project = self._project()
        if project and name in [p["name"] for p in tp.load(project)]:
            tp.remove(project, name)
            self.refresh()

    # --- finding photos -----------------------------------------------------------------------------------------
    def on_pick(self, X):
        """Ctrl + click in the 3D view: list the photos that see the spot."""
        self.spot = np.asarray(X, float)
        self._list_photos(self.spot)

    def _tie_xyz(self, name=None):
        rec = self.rec()
        project = self._project()
        if rec is None or project is None:
            return None
        p = [q for q in tp.load(project) if q["name"] == (name or self.g_point.value)]
        res = tp.resolve(rec, p)
        return tp.triangulate(rec, res[0][1]) if res else None

    def _list_photos(self, X):
        rec = self.rec()
        if rec is None:
            self.g_photos.options = ("(run SfM first)",)
            return
        pri = self.app.project.priors if self.app.project else {}
        names = getattr(self.app.project, "source_names", [])
        found = {}
        for i in rec.reg_image_ids():
            img = rec.image(i)
            q = tp.project(rec, i, X)
            cam = rec.cameras[img.camera_id]
            if q is None or not (0.03 * cam.width < q[0] < 0.97 * cam.width and 0.03 * cam.height < q[1] < 0.97 *
                                 cam.height):
                continue
            c = img.projection_center()
            s = getattr(pri.get(img.name), "source", 0)
            found.setdefault(s, []).append((float(np.linalg.norm(c - X)), img.name))
        self.candidates = []
        for s in sorted(found):
            for d, n in sorted(found[s])[:PER_FLIGHT]:
                flight = names[s] if s < len(names) else f"flight {s + 1}"
                self.candidates.append((f"{flight} · {n} · {d:.0f} m", n))
        opts = tuple(lab for lab, _ in self.candidates) or ("(no photo sees it)",)
        self.g_photos.options = opts
        self.g_photos.value = opts[0]

    def _photo_button(self, which):
        labels = [lab for lab, _ in self.candidates]
        if not labels:
            if self.g_point.value != "(none)":  # no Ctrl + click yet: the photos that see the tie point itself
                X = self._tie_xyz()
                if X is not None:
                    self._list_photos(X)
                    labels = [lab for lab, _ in self.candidates]
            if not labels:
                return
        k = labels.index(self.g_photos.value) if self.g_photos.value in labels else 0
        if which != "Open photo":
            k = (k + (1 if which == "▶" else -1)) % len(labels)
            self.g_photos.value = labels[k]
        self.open(self.candidates[k][1])

    # --- the window -----------------------------------------------------------------------------------------------
    def open(self, name, box=None, zoomed=False):
        project = self._project()
        if project is None or self.panel is None:
            return
        if self.g_point.value == "(none)":
            self.new()
        img = cv2.imread(os.path.join(project, "images", name))
        if img is None:
            return
        H, W = img.shape[:2]
        box = box or (0.0, 0.0, float(W), float(H))
        self.view = {"name": name, "box": box, "zoomed": zoomed, "size": (W, H)}
        x0, y0, x1, y1 = box
        crop = img[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        s = WINDOW_WIDTH / crop.shape[1]
        show = cv2.resize(crop, (WINDOW_WIDTH, round(crop.shape[0] * s)),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

        def mark(x, y, color, r):
            u, v = (x - x0) * s, (y - y0) * s
            if 0 <= u < show.shape[1] and 0 <= v < show.shape[0]:
                cv2.circle(show, (round(u), round(v)), r, color, 2, cv2.LINE_AA)
                cv2.drawMarker(show, (round(u), round(v)), color, cv2.MARKER_CROSS, 2 * r + 8, 1, cv2.LINE_AA)
        rec = self.rec()
        iid = next((i for i in rec.reg_image_ids() if rec.image(i).name == name), None) if rec is not None else None
        if iid is not None:
            for X, color in ((self.spot, (0, 220, 255)), (self._tie_xyz(), (255, 220, 0))):  # BGR: yellow, cyan
                if X is not None:
                    q = tp.project(rec, iid, X)
                    if q is not None:
                        mark(q[0], q[1], color, 10)
        own = next((o for p in tp.load(project) if p["name"] == self.g_point.value for o in p["obs"]
                    if o["image"] == name), None)
        if own:
            mark(own["x"], own["y"], (60, 220, 60), 6)
        ok, jpg = cv2.imencode(".jpg", show, [cv2.IMWRITE_JPEG_QUALITY, 88])
        src = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        uuid = self.g_click._impl.uuid
        self.g_img.content = (f'<img data-mf-click="{html.escape(uuid)}" src="{src}" '
                              f'style="width:100%;cursor:crosshair;display:block">')
        self.g_info.content = (f"**{self.g_point.value}** in `{name}` · "
                               + ("zoomed in: click the spot to record it" if zoomed else
                                  "click the spot to zoom in there")
                               + (f" · already clicked here ({own['x']:.0f}, {own['y']:.0f})" if own else ""))
        if self.app.photo_panel_tab != TAB:  # the Photo window at the tie point size (kept while it stays here)
            self.panel.set_width(WINDOW_WIDTH + 24)
            self.panel.set_height(round(WINDOW_WIDTH * 9 / 16) + 150)
        self.app.show_tab(TAB)  # also when the user went to the Photo tab meanwhile
        self.panel.expand()

    def _clicked(self, value):
        try:
            fx, fy = (float(v) for v in value.split(",")[:2])
        except (ValueError, AttributeError):
            return
        v = self.view
        if v is None or not (0 <= fx <= 1 and 0 <= fy <= 1):
            return
        x0, y0, x1, y1 = v["box"]
        x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        if not v["zoomed"]:
            W, H = v["size"]
            w = W / ZOOM
            h = w * 9 / 16
            bx = min(max(x - w / 2, 0.0), W - w)
            by = min(max(y - h / 2, 0.0), H - h)
            self.open(v["name"], (bx, by, bx + w, by + h), zoomed=True)
            return
        tp.set_obs(self._project(), self.g_point.value, v["name"], x, y)
        self.refresh_status()
        self.open(v["name"])  # the whole photo again, with its new mark
        self.g_info.content += f" · ✅ recorded ({x:.1f}, {y:.1f})"

    def _zoom_out(self):
        if self.view:
            self.open(self.view["name"])

    def _remove_click(self):
        if self.view:
            tp.remove(self._project(), self.g_point.value, self.view["name"])
            self.refresh_status()
            self.open(self.view["name"])

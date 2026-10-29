"""The Map window (the user asked, 2026-10-27): a classical north-up 2D map floating over the 3D view (quick bar
**Map**), drawn by the server into an image whose clicks come back through the page's img[data-mf-click] hook
(docker/patch_viser_page.py, as the Tie point view's).

Shown: map tiles (Esri aerial imagery / OpenStreetMap streets, modelforge/map2d.py) or nothing, the splat's
orthophoto over them (modelforge/ortho.py), the flight path per source, the cameras, the 3D view's camera (position
and field-of-view wedge, following Fly / Orbit / the mouse), and as layers: a heat map (photos per area, overlap
between neighbouring photos, held-out quality, Difix's thin spots), the photos the export left out as isolated, the
splat tiles, the Ingest plan, the search results and the saved / Difix views as pins.

A click does what the Click picker says: glide the 3D view there (a pin under the click: go to that object / view),
centre the map, leave a tile out of the splat (or take it back), draw a rectangle whose looked-at parts of the
flights become the Ingest sources, set where a source's range starts / ends, or draw an area that limits Images
used / Register more photos."""

import base64
import html
import json
import os
import threading
import time
import traceback

import cv2
import numpy as np
import viser

from modelforge import difix, map2d, ortho, splat_tiles
from modelforge.dji import ingest as dji_ingest
from modelforge.gui import views as views_mod
from modelforge.gui.ingest import OUTSIDE_COLOR, source_color

SIZE = 640  # the map image (px, square); the window shows it at its width
WINDOW_WIDTH = 460
CLICK_MODES = {
    "Go there (3D view)": "go",
    "Centre the map here": "centre",
    "Tile: leave out / take back": "tile",
    "Ingest: draw a rectangle": "rect",
    "Ingest: start of a range": "start",
    "Ingest: end of a range": "end",
    "Draw an area (photos)": "area",
}
BACKGROUNDS = {"aerial (Esri)": "aerial", "streets (OpenStreetMap)": "streets", "none": None}
HEATS = ("none", "photos per area", "overlap of neighbouring photos", "held-out quality (PSNR)", "Difix thin spots")
LAYERS = ("Flight path", "Cameras", "Isolated photos left out", "Splat tiles", "Ingest plan", "Search results",
          "Saved views", "Difix views")
LAYER_ON = {"Flight path", "Cameras", "Search results", "Saved views", "Difix views"}
PIN_PX = 14  # a click this close to a pin goes to it
TRACK_PX = 20  # Ingest start / end: the click must be this close to a source's track
OVERLAP_STEPS = ((0.06, (220, 40, 40)), (0.12, (245, 140, 30)), (0.2, (240, 220, 40)), (9.9, (60, 200, 80)))


def _mtime(*parts):
    try:
        return os.path.getmtime(os.path.join(*parts))
    except (OSError, TypeError):
        return None


def _nice(m):
    """A round length near m (1, 2, 5 x 10^k)."""
    k = 10 ** np.floor(np.log10(max(m, 1e-6)))
    return float(next(f * k for f in (5, 2, 1) if f * k <= m) if m >= k else k)


def _ramp(t):
    """Values 0..1 -> RGB uint8, red (0) -> yellow -> green (1)."""
    t = np.clip(np.asarray(t, float), 0, 1)[..., None]
    red, yellow, green = np.array([215, 40, 40]), np.array([245, 215, 50]), np.array([50, 190, 80])
    return np.where(t < 0.5, red + (yellow - red) * (t / 0.5), yellow + (green - yellow) * ((t - 0.5) / 0.5)).astype(
        np.uint8)


def _text(img, s, org, scale=0.45, color=(255, 255, 255), outline=(0, 0, 0)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, outline, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _ipts(p):
    return np.round(np.asarray(p, float)).astype(np.int32)


class MapPanel:
    def __init__(self, app):
        self.app = app
        gui = app.server.gui
        self.view = None  # map2d.View; None = fit on the next draw
        self.follow = False
        self.cam = None  # the 3D view's camera: (position, look_at, vertical fov, aspect)
        self.draft = []  # the rectangle's / area's corners (ENU x / y)
        self.rect_result = None  # (sources, {video: pieces}) of the drawn rectangle
        self._undo_sources = None
        self._base, self._base_key, self._pins = None, None, []
        self._cache = {}  # computed layer data by key
        self._busy = set()  # cache keys being computed
        self._wake, self._lock = threading.Event(), threading.RLock()
        self._last_cam_draw = 0.0
        self._msg, self._draw_msg = "", ""  # the last action's result: the status lines and the Drawing folder
        self.tiles = map2d.TileCache(app.root, on_tile=self.redraw)
        self.panel = gui.add_panel()
        with self.panel.add_tab("Map", viser.Icon.MAP):
            self.g_img = gui.add_html("")
            self.g_click = gui.add_text("map click", initial_value="", visible=False)
            self.g_click.on_update(lambda e: self._clicked(e.target.value, e.client))
            nav = gui.add_button_group("", options=("＋", "－", "Fit", "Follow"),
                                       hint="＋ / －: zoom about the map's centre. Fit: everything there is. Follow: "
                                            "keep the 3D view's camera on the map (pressing it again stops).")
            nav.on_click(lambda e: self._nav(e.target.value))
            self.g_mode = gui.add_dropdown(
                "Click", options=tuple(CLICK_MODES), initial_value="Go there (3D view)",
                hint="What a click on the map does. Go there: the 3D view glides to look at that spot from where it "
                     "looks now (a pin under the click: go to that search result / view). Tile: leave a splat tile "
                     "out of training and the merge, or take it back. Ingest rectangle: two clicks (corners); the "
                     "parts of the flights looking into it become the Ingest sources. Ingest start / end: the "
                     "nearest point of a source's track sets where its range starts / ends. Area: click the corners "
                     "of a polygon, then use it for Images used / Register more photos.")
            self.g_mode.on_update(lambda _: self._mode_changed())
            self.g_md = gui.add_markdown("")
            with gui.add_folder("Drawing", expand_by_default=True) as self.g_draw:
                self.g_draw_md = gui.add_markdown("")
                self.g_by = gui.add_dropdown(
                    "Inside means", options=("look-at point", "drone position"), initial_value="look-at point",
                    hint="Ingest rectangle: a frame counts when the point its camera looks at (the optical axis on "
                         "the ground at the flight's takeoff height) lies inside, or the drone itself.")
                self.g_min = gui.add_number("Min piece [s]", initial_value=5.0, min=0.0, step=0.5,
                                            hint="Ingest rectangle: shorter stretches inside are left out.")
                self.g_gap = gui.add_number("Join gaps [s]", initial_value=3.0, min=0.0, step=0.5,
                                            hint="Ingest rectangle: stretches less than this apart become one piece.")
                for g in (self.g_by, self.g_min, self.g_gap):
                    g.on_update(lambda _: self._rect_preview())
                self.g_draw_btn = gui.add_button_group(
                    "", options=("Apply", "Undo", "Clear"),
                    hint="Apply: the rectangle's pieces replace the Ingest sources (Undo brings the previous ones "
                         "back), or the area goes to Images used and Register more photos (only photos looking into "
                         "it). Clear: start the drawing again (an area applied before is removed).")
                self.g_draw_btn.on_click(lambda e: self._draw_action(e.target.value))
            with gui.add_folder("Background and layers", expand_by_default=False):
                self.g_bg = gui.add_dropdown("Background", options=tuple(BACKGROUNDS), initial_value="aerial (Esri)",
                                             hint="Map tiles from the internet (cached in the data folder's "
                                                  ".map_tiles); none for an offline map. Projects without GPS have "
                                                  "none.")
                self.g_ortho = gui.add_checkbox(
                    "Orthophoto of the splat", initial_value=True,
                    hint="The splat rendered straight down (Render orthophoto), over the map tiles.")
                ob = gui.add_button_group("", options=("Render orthophoto", "Export GeoTIFF"),
                                          hint="Render: the splat straight down, north up (a job, seconds to a "
                                               "minute; splat/ortho.png). Export: export/ortho.tif in the UTM zone "
                                               "of the project, for GIS tools.")
                ob.on_click(lambda e: self.app.run_stage("ortho") if e.target.value == "Render orthophoto"
                            else self._export_geotiff())
                self.g_heat = gui.add_dropdown(
                    "Heat map", options=HEATS, initial_value="none",
                    hint="photos per area: how many registered photos see each ground cell (few = fly there again). "
                         "overlap: the shared features of consecutive photos in use (Images used → Analyze; red < 0.06 "
                         "breaks SfM, green >= 0.2). held-out quality: PSNR of the evaluation's held-out photos at "
                         "the spots they look at. Difix thin spots: where the splat is thin on the SfM mesh.")
                self.g_layers = {}
                for name in LAYERS:
                    self.g_layers[name] = gui.add_checkbox(name, initial_value=name in LAYER_ON)
                for g in (self.g_bg, self.g_ortho, self.g_heat, *self.g_layers.values()):
                    g.on_update(lambda _: self.redraw())
        self._mode_changed()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    # --- state -----------------------------------------------------------------------------------------------
    def origin(self):
        """WGS84 origin of the ENU frame: the project's, else the Ingest plan's; None without GPS."""
        p = self.app.project
        if p is not None and p.meta.get("enu_origin_lla"):
            return p.meta["enu_origin_lla"]
        prev = self.app.ingest.prev
        if prev and self.app.ingest.kind != "video":
            return prev.get("origin")
        return None

    def reset(self):
        """Another project: fit again, forget drawings and cached layers."""
        with self._lock:
            self.view, self.draft, self.rect_result, self._undo_sources = None, [], None, None
            self._cache, self._base_key = {}, None
        self._mode_changed()

    def redraw(self):
        self._base_key = None
        self._wake.set()

    def on_camera(self, client):
        cam = client.camera
        try:
            self.cam = (np.array(cam.position, float), np.array(cam.look_at, float), float(cam.fov),
                        float(cam.image_width) / max(float(cam.image_height), 1.0))
        except Exception:  # noqa: BLE001 - a camera without its size yet
            return
        if time.time() - self._last_cam_draw > 0.2:
            self._wake.set()

    def _data_key(self):
        """Changes when anything the map shows changes on disk or in the app (checked every 2 s)."""
        a, path = self.app, self.app.path
        sem = getattr(a, "semantic", None)
        return (path, id(a.snap), id(a.ingest.prev), id(getattr(sem, "objs", None)),
                _mtime(path, views_mod.FILE), _mtime(path, difix.VIEWS_FILE), _mtime(path, "tiles", "tiles.json"),
                _mtime(path, "tiles", "skip.json"), _mtime(path, ortho.META), _mtime(path, "report.json"),
                _mtime(path, "analysis", "overlap.json"), _mtime(path, "evaluation", "metrics.json"),
                _mtime(path, "splat", "splat_enu.ply"))

    def _watch(self):
        last = None
        while True:
            time.sleep(2.0)
            try:
                k = self._data_key()
            except Exception:  # noqa: BLE001
                continue
            if k != last:
                last = k
                self.redraw()

    # --- geometry --------------------------------------------------------------------------------------------
    def _cameras(self):
        """(centres (n, 3), R_wc, sources, times) of the shown snapshot's registered cameras, else the priors."""
        a = self.app
        p = a.project
        if p is None:
            return None
        cams = a.snap.cams if a.snap is not None and a.snap.cams.valid.any() else p.prior_cams
        return cams.centers, cams.R_wc, p.source, p.time

    def _extent_points(self):
        pts = []
        c = self._cameras()
        if c is not None:
            pts.append(c[0][np.isfinite(c[0][:, 0]), :2])
        prev = self.app.ingest.prev
        if prev:
            for s in prev["sources"]:
                pts.append(np.asarray(s["track"])[np.asarray(s["track_valid"], bool), :2])
        if self.draft:
            pts.append(np.array(self.draft))
        pts = [q for q in pts if len(q)]
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def _ground(self):
        """Terrain (pairs.Terrain) from the shown snapshot's points, else None (cached per snapshot)."""
        snap = self.app.snap
        key = ("terrain", id(snap))
        if key not in self._cache:
            t = None
            if snap is not None and len(snap.points) >= 1000:
                from modelforge.sfm.pairs import Terrain
                t = Terrain(snap.points)
            self._cache[key] = t
        return self._cache[key]

    def _ground_z(self, xy):
        t = self._ground()
        if t is not None:
            return float(t.z(np.asarray(xy, float)[None])[0])
        snap = self.app.snap
        if snap is not None and len(snap.points):
            return float(np.median(snap.points[:, 2]))
        c = self._cameras()
        return float(np.nanmin(c[0][:, 2])) - 20.0 if c is not None else 0.0

    # --- layer data, computed in the background --------------------------------------------------------------
    def _later(self, key, fn):
        """Cached value of key, else None while fn() runs in a thread (a redraw follows)."""
        if key in self._cache:
            return self._cache[key]
        if key not in self._busy:
            self._busy.add(key)

            def work():
                try:
                    self._cache[key] = fn()
                except Exception as e:  # noqa: BLE001 - shown on the map's status line
                    traceback.print_exc()
                    self._cache[key] = {"error": str(e)}
                self._busy.discard(key)
                self.redraw()
            threading.Thread(target=work, daemon=True).start()
        return None

    def _heat(self, kind):
        a, path = self.app, self.app.path
        if kind == "photos per area":
            c = self._cameras()
            snap = a.snap
            if c is None or snap is None or not snap.intrinsics:
                return {"error": "needs an SfM model (Run SfM)"}
            key = ("cover", id(snap))

            def fn():
                intr = snap.intrinsics
                p = intr["params"]
                K = (p[0], p[1], p[2], p[3]) if intr["model"] in ("OPENCV", "PINHOLE", "OPENCV_FISHEYE") \
                    else (p[0], p[0], p[1], p[2])
                ok = np.isfinite(c[0][:, 0])
                xy = c[0][ok, :2]
                lo, hi = xy.min(0) - 60.0, xy.max(0) + 60.0
                cell = max(float((hi - lo).max()) / 160, 0.5)
                t = self._ground()
                ground = t.z if t is not None else (lambda q: np.full(len(q), self._ground_z(q[0])))
                grid, geo_ = map2d.coverage(c[0], c[1], K, (intr["width"], intr["height"]), ground,
                                            (*lo, *hi), cell)
                return {"grid": grid, "geo": geo_}
            return self._later(key, fn)
        if kind == "overlap of neighbouring photos":
            c = self._cameras()
            if c is None:
                return {"error": "no project"}
            return self._later(("overlap", id(a.snap), _mtime(path, "analysis", "overlap.json")),
                               lambda: {"pairs": map2d.neighbour_overlap(path, a.project.names, c[0])}
                               if map2d.neighbour_overlap(path, a.project.names, c[0]) is not None else
                               {"error": "no overlap analysis (Sparse tab → Images used → Analyze extracted images)"})
        if kind == "held-out quality (PSNR)":
            per = map2d.heldout(path) if path else None
            if not per:
                return {"error": "no held-out evaluation (evaluation/metrics.json; modelforge.evaluate)"}

            def fn():
                names, _, look = self._pool()
                at = {n: look[i] for i, n in enumerate(names)}
                pts = [(at[n][0], at[n][1], v) for n, v in per if n in at]
                return {"points": np.array(pts) if pts else np.zeros((0, 3))}
            return self._later(("heldout", path, _mtime(path, "evaluation", "metrics.json")), fn)
        if kind == "Difix thin spots":
            if not path or difix.can_suggest(path) is not None:
                return {"error": difix.can_suggest(path) if path else "no project"}

            def fn():
                r = map2d.thin_spots(path)
                return {"error": "nothing thin"} if r is None or not len(r[0]) else {"xy": r[0]}
            return self._later(("thin", path, _mtime(path, "splat", "splat_enu.ply")), fn)
        return None

    def _pool(self):
        path = self.app.path
        key = ("pool", path, _mtime(path, "sparse_pc.ply"), _mtime(path, "extracted", "poses.json"),
               _mtime(path, "poses.json"))
        if key not in self._cache:
            self._cache[key] = map2d.pool_lookats(path)
        return self._cache[key]

    def _tiles(self):
        path = self.app.path
        if not path or not os.path.exists(os.path.join(path, "transforms.json")):
            return None
        a = self.app
        key = ("tiles", path, _mtime(path, "tiles", "tiles.json"), _mtime(path, "tiles", "skip.json"),
               int(a.g_splat_tile_photos.value), a.g_splat_tile_layout.value)
        return self._later(key, lambda: dict(zip(("cells", "planned"),
                                                 splat_tiles.overlay(path, key[4], key[5]))))

    def _ortho(self):
        path = self.app.path
        meta = ortho.current(path) if path else None
        if meta is None:
            return None
        return self._later(("ortho", path, tuple(meta["splat"])), lambda: dict(zip(("img", "meta"),
                                                                                   ortho.load(path, meta))))

    # --- drawing ---------------------------------------------------------------------------------------------
    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            time.sleep(0.05)  # coalesce
            try:
                self._draw()
            except Exception:  # noqa: BLE001 - the map never breaks the GUI
                traceback.print_exc()

    def _fit(self):
        self.view = map2d.View.fit(self._extent_points(), SIZE, SIZE)

    def _draw(self):
        with self._lock:
            if self.view is None:
                self._fit()
            if self.follow and self.cam is not None:
                u, v = self.view.to_px(self.cam[0][:2])[0]
                if not (0.2 * SIZE < u < 0.8 * SIZE and 0.2 * SIZE < v < 0.8 * SIZE):
                    self.view = map2d.View(self.cam[0][0], self.cam[0][1], self.view.mpp, SIZE, SIZE)
                    self._base_key = None
            view = self.view
            key = (view.key(), self.g_bg.value, self.g_ortho.value, self.g_heat.value,
                   tuple(g.value for g in self.g_layers.values()), self._data_key(), len(self._cache))
            notes = []
            if key != self._base_key or self._base is None:
                self._base, self._pins, notes = self._draw_base(view)
                self._base_key, self._notes = key, notes
            img = self._base.copy()
            self._draw_dynamic(img, view)
        self._last_cam_draw = time.time()
        ok, jpg = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        src = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        uuid = self.g_click._impl.uuid
        self.g_img.content = (f'<img data-mf-click="{html.escape(uuid)}" src="{src}" '
                              f'style="width:100%;cursor:crosshair;display:block">')
        self._status()

    def _draw_base(self, view):
        notes = []
        origin = self.origin()
        style = BACKGROUNDS[self.g_bg.value]
        img, complete = map2d.basemap(view, origin if style else None, style, self.tiles)
        if style and origin is None:
            notes.append("no GPS reference: no map tiles")
        elif style and not complete:
            notes.append("loading map tiles …")
        if self.g_ortho.value:
            o = self._ortho()
            if o and "img" in o and o["img"] is not None:
                self._blend_raster(img, view, o["img"], o["meta"]["bounds"], o["meta"]["mpp"])
            elif self.app.path and os.path.exists(ortho.splat_path(self.app.path)) and ortho.current(
                    self.app.path) is None:
                notes.append("no orthophoto of this splat yet: Background and layers → Render orthophoto")
        heat = self.g_heat.value
        if heat != "none":
            d = self._heat(heat)
            if d is None:
                notes.append(f"{heat}: computing …")
            elif "error" in d:
                notes.append(f"{heat}: {d['error']}")
            else:
                notes.append(self._draw_heat(img, view, heat, d))
        L = {k: g.value for k, g in self.g_layers.items()}
        pins = []
        if L["Splat tiles"]:
            t = self._tiles()
            if t and "cells" in t:
                self._draw_tiles(img, view, t)
            elif t and "error" in t:
                notes.append(f"tiles: {t['error']}")
        if L["Ingest plan"]:
            self._draw_plan(img, view)
        c = self._cameras()
        if c is not None and (L["Flight path"] or L["Cameras"]):
            self._draw_path(img, view, c, L["Flight path"], L["Cameras"])
        if L["Isolated photos left out"] and c is not None:
            idx = [self.app.project.index[n] for n in map2d.isolated(self.app.path) if n in self.app.project.index]
            at = np.where(np.isfinite(c[0][:, :1]), c[0], self.app.project.prior_cams.centers)  # left out: no pose
            for u, v in view.to_px(at[idx, :2]) if idx else []:
                if np.isfinite(u):
                    p = (int(u), int(v))
                    cv2.drawMarker(img, p, (255, 40, 40), cv2.MARKER_TILTED_CROSS, 9, 2, cv2.LINE_AA)
            if idx:
                notes.append(f"✕ {len(idx)} photos left out as isolated")
        if L["Search results"]:
            pins += self._draw_objects(img, view)
        if L["Saved views"] and self.app.path:
            pins += self._draw_views(img, view, views_mod.load(self.app.path), (80, 170, 255), "view")
        if L["Difix views"] and self.app.path:
            pins += self._draw_views(img, view, views_mod.load(self.app.path, difix.VIEWS_FILE), (255, 120, 220),
                                     "difix")
        self._draw_scale(img, view)
        if style and origin is not None:
            a = map2d.STYLES[style]["attribution"]
            (tw, _), _ = cv2.getTextSize(a, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            _text(img, a, (SIZE - tw - 6, SIZE - 8), 0.38)
        return img, pins, notes

    def _blend_raster(self, img, view, rgba, bounds, mpp):
        """An RGBA raster over ENU bounds (north up) into the view, alpha blended."""
        x0, y0, x1, y1 = bounds
        s = mpp / view.mpp  # raster px -> view px
        u0, v0 = view.to_px([[x0, y1]])[0]
        M = np.array([[s, 0, u0], [0, s, v0]], np.float32)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        if s < 0.5:  # shrink first: a warp alone aliases
            f = max(1, int(1 / s))
            rgba = cv2.resize(rgba, (max(1, rgba.shape[1] // f), max(1, rgba.shape[0] // f)), interpolation=cv2.INTER_AREA)
            M[:, :2] *= f
        w = cv2.warpAffine(rgba, M, (view.w, view.h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = w[:, :, 3:4].astype(np.float32) / 255.0
        img[:] = (img * (1 - a) + w[:, :, :3] * a).astype(np.uint8)

    def _draw_heat(self, img, view, kind, d):
        if kind == "photos per area":
            grid, (x0, y0, cell) = d["grid"], d["geo"]
            g = np.flipud(grid)  # row 0 = north
            seen = grid[grid > 0]
            lo, hi = (float(np.percentile(seen, 2)), float(np.percentile(seen, 98))) if len(seen) else (0.0, 1.0)
            col = _ramp((g - lo) / max(hi - lo, 1.0))
            alpha = np.where(g > 0, 130, 0).astype(np.uint8)[..., None]
            self._blend_raster(img, view, np.concatenate([col, alpha], 2), (x0, y0, x0 + grid.shape[1] * cell,
                                                                             y0 + grid.shape[0] * cell), cell)
            return (f"photos per area: red ≤ {lo:.0f} … green ≥ {hi:.0f} photos per {cell:.1f} m cell (none: "
                    "uncoloured)")
        if kind == "overlap of neighbouring photos":
            c = self._cameras()
            P = view.to_px(c[0][:, :2])
            bad = 0
            for i, j, o in d["pairs"]:
                if not (np.isfinite(P[i]).all() and np.isfinite(P[j]).all()):
                    continue
                col = next(cc for t, cc in OVERLAP_STEPS if o < t)
                bad += o < 0.12
                cv2.line(img, tuple(_ipts(P[i])), tuple(_ipts(P[j])), col, 4, cv2.LINE_AA)
            return (f"overlap of consecutive photos: red < 0.06, orange < 0.12, yellow < 0.2, green ≥ 0.2 "
                    f"({bad} pairs below 0.12)")
        if kind == "held-out quality (PSNR)":
            pts = d["points"]
            if not len(pts):
                return "held-out quality: no held-out photo found among the extracted ones"
            lo, hi = np.percentile(pts[:, 2], 5), np.percentile(pts[:, 2], 95)
            for (x, y, v), (u, w) in zip(pts, view.to_px(pts[:, :2])):
                col = tuple(int(c) for c in _ramp((v - lo) / max(hi - lo, 1e-6)))
                cv2.circle(img, (int(u), int(w)), 6, col, -1, cv2.LINE_AA)
                cv2.circle(img, (int(u), int(w)), 6, (0, 0, 0), 1, cv2.LINE_AA)
            return f"held-out PSNR: red {lo:.1f} dB … green {hi:.1f} dB ({len(pts)} photos)"
        if kind == "Difix thin spots":
            xy = d["xy"]
            x0, y0, x1, y1 = view.bounds()
            cell = max(view.mpp * 4, 0.25)
            grid, _ = map2d.raster(xy, np.ones(len(xy)), (x0, y0, x1, y1), cell, reduce="sum")
            g = np.flipud(np.nan_to_num(grid))
            alpha = np.clip(g / max(np.percentile(g[g > 0], 90) if (g > 0).any() else 1, 1) * 200, 0, 200)
            col = np.zeros(g.shape + (3,), np.uint8)
            col[:] = (255, 0, 200)
            self._blend_raster(img, view, np.concatenate([col, alpha.astype(np.uint8)[..., None]], 2),
                               (x0, y0, x0 + g.shape[1] * cell, y0 + g.shape[0] * cell), cell)
            return f"Difix thin spots: magenta where the splat is thin on the SfM mesh ({len(xy):,} samples)"
        return ""

    def _draw_tiles(self, img, view, t):
        for c in t["cells"]:
            col = (150, 150, 150) if c.get("skipped") else (255, 160, 0) if c["status"] == "training" else (255, 50, 50)
            rings = c.get("outlines") or []
            if not rings:
                x0, y0, x1, y1 = c["rect"]
                rings = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
            for ring in rings:
                q = view.to_px(np.clip(np.asarray(ring, float), -1e6, 1e6))
                cv2.polylines(img, [_ipts(q)], True, col, 2, cv2.LINE_AA)
            cx, cy = c.get("center") or ((c["rect"][0] + c["rect"][2]) / 2, (c["rect"][1] + c["rect"][3]) / 2)
            u, v = view.to_px([[cx, cy]])[0]
            if 0 <= u < SIZE and 0 <= v < SIZE:
                _text(img, f"{c['tile'] + 1}" + (" (out)" if c.get("skipped") else ""), (int(u) - 8, int(v)), 0.6, col)

    def _draw_plan(self, img, view):
        ing = self.app.ingest
        if not ing.prev or not ing.plan:
            return
        for sp, pv in zip(ing.plan.sources, ing.prev["sources"]):
            t = view.to_px(np.asarray(pv["track"])[:, :2])
            v = np.asarray(pv["track_valid"], bool)
            cv2.polylines(img, [_ipts(t)], False, OUTSIDE_COLOR, 1, cv2.LINE_AA)
            col = source_color(sp.index)
            k = np.flatnonzero(np.diff(np.r_[0, v.astype(np.int8), 0]))
            for a, b in zip(k[::2], k[1::2]):
                cv2.polylines(img, [_ipts(t[a:b])], False, col, 3, cv2.LINE_AA)
            for u, w in view.to_px(np.asarray(pv["samples"])[:, :2]) if len(pv["samples"]) else []:
                cv2.circle(img, (int(u), int(w)), 2, col, -1)

    def _draw_path(self, img, view, c, path, dots):
        C, _, src, tm = c
        P = view.to_px(C[:, :2])
        ok = np.isfinite(P[:, 0])
        for s in np.unique(src):
            k = np.flatnonzero(ok & (src == s))
            k = k[np.argsort(tm[k])]
            col = source_color(int(s))
            if path and len(k) > 1:
                cut = np.flatnonzero(np.diff(tm[k]) > 30.0) + 1  # pauses between pieces: no line across
                for part in np.split(k, cut):
                    cv2.polylines(img, [_ipts(P[part])], False, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.polylines(img, [_ipts(P[part])], False, col, 2, cv2.LINE_AA)
            if dots:
                for u, v in P[k]:
                    cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255), -1)

    def _draw_objects(self, img, view):
        sem = getattr(self.app, "semantic", None)
        objs = getattr(sem, "objs", None) or []
        pins = []
        for k, o in enumerate(objs):
            u, v = view.to_px([o["center"][:2]])[0]
            r = max(5, int(o["radius"] / view.mpp))
            cv2.circle(img, (int(u), int(v)), r, (255, 230, 0), 2, cv2.LINE_AA)
            _text(img, f"#{k + 1}", (int(u) + r + 2, int(v) + 4), 0.45, (255, 230, 0))
            pins.append((u, v, "object", k))
        return pins

    def _draw_views(self, img, view, vs, col, kind):
        pins = []
        for v in vs:
            p, la = np.array(v["position"], float), np.array(v["look_at"], float)
            (u, w), (u2, w2) = view.to_px([p[:2], la[:2]])
            d = np.array([u2 - u, w2 - w])
            n = np.linalg.norm(d)
            if n > 1e-6:
                e = np.array([u, w]) + d / n * 18
                cv2.arrowedLine(img, (int(u), int(w)), tuple(_ipts(e)), col, 2, cv2.LINE_AA, tipLength=0.4)
            cv2.circle(img, (int(u), int(w)), 5, col, -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(w)), 5, (0, 0, 0), 1, cv2.LINE_AA)
            _text(img, v["name"][:18], (int(u) + 7, int(w) - 6), 0.38, col)
            pins.append((u, w, kind, v["name"]))
        return pins

    def _draw_scale(self, img, view):
        m = _nice(view.mpp * SIZE * 0.25)
        px = int(m / view.mpp)
        x, y = 10, SIZE - 24
        cv2.line(img, (x, y), (x + px, y), (0, 0, 0), 5)
        cv2.line(img, (x, y), (x + px, y), (255, 255, 255), 2)
        _text(img, f"{m:g} m" if m < 1000 else f"{m / 1000:g} km", (x + px + 6, y + 4), 0.45)
        _text(img, "N", (SIZE - 22, 22), 0.6)
        cv2.arrowedLine(img, (SIZE - 16, 48), (SIZE - 16, 28), (255, 255, 255), 2, tipLength=0.4)

    def _draw_dynamic(self, img, view):
        mode = CLICK_MODES[self.g_mode.value]
        if self.draft:
            q = view.to_px(np.array(self.draft))
            if mode == "rect" and len(self.draft) == 2:
                (a, b), (c, d) = q
                cv2.rectangle(img, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.polylines(img, [_ipts(q)], mode == "area" and len(q) > 2, (255, 255, 255), 2, cv2.LINE_AA)
            for u, v in q:
                cv2.circle(img, (int(u), int(v)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        sel = self.app.subset.sel.area if hasattr(self.app, "subset") else None
        if sel and not (mode == "area" and self.draft):
            cv2.polylines(img, [_ipts(view.to_px(np.array(sel["polygon"])))], True, (0, 230, 255), 2, cv2.LINE_AA)
        if self.cam is not None:
            pos, la, fov, aspect = self.cam
            u, v = view.to_px([pos[:2]])[0]
            d = la[:2] - pos[:2]
            if np.linalg.norm(d) > 1e-6:
                yaw = np.arctan2(-d[1], d[0])  # image angle (v down)
                half = np.arctan(np.tan(fov / 2) * aspect)
                r = 60
                pts = [(u, v)] + [(u + r * np.cos(a), v + r * np.sin(a))
                                  for a in np.linspace(yaw - half, yaw + half, 12)]
                over = img.copy()
                cv2.fillPoly(over, [_ipts(pts)], (255, 255, 255))
                cv2.addWeighted(over, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [_ipts(pts)], True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 6, (255, 60, 60), -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 6, (255, 255, 255), 2, cv2.LINE_AA)

    def _status(self):
        mode = CLICK_MODES[self.g_mode.value]
        v = self.view
        lines = [f"{v.mpp * SIZE:.0f} m across" + (" · following the camera" if self.follow else "")] if v else []
        lines += getattr(self, "_notes", [])
        if self._msg:
            lines.append(self._msg)
        self.g_md.content = "  \n".join(lines)
        if self._draw_msg:
            self.g_draw_md.content = self._draw_msg
        elif mode == "rect":
            self.g_draw_md.content = (f"click corner {len(self.draft) + 1} of 2 of the rectangle" if len(self.draft) < 2
                                      else "⏳ finding the parts of the flights looking into it …")
        elif mode == "area":
            n = len(self.draft)
            self.g_draw_md.content = (f"{n} corner(s): click more, then **Apply** (≥ 3 corners)" if n
                                      else "click the corners of the area")
        else:
            self.g_draw_md.content = ""

    # --- interaction -----------------------------------------------------------------------------------------
    def _nav(self, what):
        with self._lock:
            if self.view is None:
                self._fit()
            v = self.view
            if what == "＋":
                self.view = map2d.View(v.cx, v.cy, v.mpp / 2, SIZE, SIZE)
            elif what == "－":
                self.view = map2d.View(v.cx, v.cy, v.mpp * 2, SIZE, SIZE)
            elif what == "Fit":
                self._fit()
            elif what == "Follow":
                self.follow = not self.follow
        self.redraw()

    def _mode_changed(self):
        mode = CLICK_MODES[self.g_mode.value]
        self._msg, self._draw_msg = "", ""
        self.draft, self.rect_result = [], None
        drawing = mode in ("rect", "area")
        self.g_draw.visible = drawing
        for g in (self.g_by, self.g_min, self.g_gap):
            g.visible = mode == "rect"
        self.redraw()

    def _clicked(self, value, client):
        try:
            fx, fy = (float(v) for v in value.split(",")[:2])
        except (ValueError, AttributeError):
            return
        if self.view is None or not (0 <= fx <= 1 and 0 <= fy <= 1):
            return
        u, v = fx * SIZE, fy * SIZE
        xy = self.view.to_world(u, v)
        self._msg, self._draw_msg = "", ""
        mode = CLICK_MODES[self.g_mode.value]
        try:
            if mode in ("go", "centre"):
                pin = min(self._pins, key=lambda p: np.hypot(p[0] - u, p[1] - v), default=None)
                if pin is not None and np.hypot(pin[0] - u, pin[1] - v) <= PIN_PX:
                    self._go_pin(pin, client)
                elif mode == "go":
                    self._go(xy, client)
                else:
                    with self._lock:
                        self.view = map2d.View(xy[0], xy[1], self.view.mpp, SIZE, SIZE)
                    self.follow = False
            elif mode == "tile":
                self._tile_click(xy)
            elif mode == "rect":
                self.draft = [list(xy)] if len(self.draft) != 1 else self.draft + [list(xy)]
                self.rect_result = None
                if len(self.draft) == 2:
                    self._rect_preview()
            elif mode in ("start", "end"):
                self._range_click(u, v, mode == "start")
            elif mode == "area":
                self.draft.append(list(xy))
        except Exception as e:  # noqa: BLE001 - shown on the map
            traceback.print_exc()
            self._draw_msg = f"❌ {e}"
        self.redraw()

    def _go(self, xy, client):
        """Glide the clicking browser's camera to look at the spot, keeping its offset (height, direction)."""
        if client is None:
            return
        target = np.array([xy[0], xy[1], self._ground_z(xy)])
        cam = client.camera
        off = np.array(cam.position, float) - np.array(cam.look_at, float)
        if not np.isfinite(off).all() or np.linalg.norm(off) < 1e-3:
            off = np.array([0.0, -30.0, 30.0])
        self.app.glide_to(client, target + off, target)

    def _go_pin(self, pin, client):
        _, _, kind, payload = pin
        if kind == "object":
            self.app.semantic.go(payload, client)
        elif kind == "view":
            self.app.views.go(payload, client)
        elif kind == "difix":
            self.app.difix_views.go(payload, client)

    def _tile_click(self, xy):
        t = self._tiles()
        if not t or "cells" not in t:
            self._msg = "tiles: none planned yet"
            return
        for c in t["cells"]:
            rings = c.get("outlines")
            if rings:
                inside = any(map2d.inside_polygon(np.array([xy]), np.asarray(r))[0] for r in rings)
            else:
                x0, y0, x1, y1 = c["rect"]
                inside = x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1
            if inside:
                self.app._toggle_tile(c["tile"], t["planned"])
                self._cache = {k: v for k, v in self._cache.items() if k[0] != "tiles"}
                return

    def _range_click(self, u, v, start):
        ing = self.app.ingest
        if not ing.prev or not ing.plan:
            raise RuntimeError("no Ingest plan with flight tracks (DJI / ROS bag sources) to pick from")
        best = None
        for sp, pv in zip(ing.plan.sources, ing.prev["sources"]):
            P = self.view.to_px(np.asarray(pv["track"])[:, :2])
            d = np.hypot(P[:, 0] - u, P[:, 1] - v)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (d[i], sp, int(pv["track_frames"][i]))
        if best is None or best[0] > TRACK_PX:
            self._draw_msg = "no source's track near the click"
            return
        _, sp, frame = best
        k = next(i for i, s in enumerate(ing.sources) if s is sp.source)
        w = ing._source_widgets[k]
        t = float(sp.clip.pts[min(frame, len(sp.clip.pts) - 1)])
        a, b = w["range"].value
        w["range"].value = (min(t, b - 0.5), b) if start else (a, max(t, a + 0.5))  # → the Ingest tab replans
        self._msg = f"source {k + 1}: range {'starts' if start else 'ends'} at {t:.1f} s"

    def _rect_preview(self):
        if CLICK_MODES[self.g_mode.value] != "rect" or len(self.draft) != 2:
            return
        ing = self.app.ingest
        origin = self.origin()
        if not ing.sources or origin is None or ing.kind == "video":
            self._draw_msg = "needs Ingest sources with GPS (DJI videos or ROS bags with poses)"
            self.redraw()
            return
        (a, b), (c, d) = self.draft
        poly = [(a, b), (c, b), (c, d), (a, d)]
        by = "look-at" if self.g_by.value == "look-at point" else "position"
        self.rect_result = None

        def work():
            try:
                res = dji_ingest.sources_in_area(list(ing.sources), ing.settings, poly, origin, by=by,
                                                 min_s=float(self.g_min.value), gap_s=float(self.g_gap.value))
                self.rect_result = res
                srcs, found = res
                lines = [f"→ **{len(srcs)} piece(s)** in {len(found)} video(s):"] + [
                    f"- {os.path.basename(vid)}: " + ", ".join(f"{t0:.0f}–{t1:.0f} s" for t0, t1 in p)
                    for vid, p in found.items()]
                self._draw_msg = "  \n".join(lines) + "  \n**Apply** replaces the Ingest sources with them" \
                    if srcs else "no part of the flights looks into the rectangle"
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._draw_msg = f"❌ {e}"
            self.redraw()
        threading.Thread(target=work, daemon=True).start()

    def _draw_action(self, what):
        mode = CLICK_MODES[self.g_mode.value]
        ing = self.app.ingest
        if what == "Clear":
            self.draft, self.rect_result = [], None
            if mode == "area":
                self._set_area(None)
        elif what == "Undo" and mode == "rect":
            if self._undo_sources is None:
                self._draw_msg = "nothing to undo"
            else:
                ing.sources, self._undo_sources = self._undo_sources, None
                ing._rebuild_sources()
                ing.request_plan()
                self._draw_msg = "↶ the previous Ingest sources are back"
        elif what == "Apply" and mode == "rect":
            if self.app.running or ing.running:
                self._draw_msg = "⚪ a job is running: wait for it to finish"
            elif not self.rect_result or not self.rect_result[0]:
                self._draw_msg = "draw a rectangle that the flights look into first"
            else:
                self._undo_sources = list(ing.sources)
                ing.sources = list(self.rect_result[0])
                ing._rebuild_sources()
                ing.request_plan()
                self._draw_msg = (f"✅ the Ingest sources are now the {len(ing.sources)} piece(s) "
                                          "(Undo brings the previous ones back); extract in the Ingest tab")
                self.draft, self.rect_result = [], None
        elif what == "Apply" and mode == "area":
            if len(self.draft) < 3:
                self._draw_msg = "an area needs at least 3 corners"
            else:
                self._set_area(list(self.draft))
                self.draft = []
        elif what == "Undo" and mode == "area":
            self._set_area(None)
        self.redraw()

    def _set_area(self, poly):
        """The area to Images used and Register more photos (their Selection), or cleared."""
        app = self.app
        if poly is None:
            for sel in (app.subset.sel, app.register.sel):
                sel.set_area(None, ())
            app.subset._preview()
            app.register._preview()
            self._draw_msg = "area removed from Images used / Register more photos"
            return
        if not app.path:
            return
        self._draw_msg = "⏳ finding the photos looking into the area …"

        def work():
            try:
                names, _, look = self._pool()
                inside = [n for n, k in zip(names, map2d.inside_polygon(look[:, :2], poly)) if k] if names else []
                for sel in (app.subset.sel, app.register.sel):
                    sel.set_area(poly, inside)
                app.subset._preview()
                app.register._preview()
                self._draw_msg = (f"✅ area applied: {len(inside)} of {len(names)} extracted photos look "
                                          "into it. Sparse tab → Images used / Register more photos now choose only "
                                          "among them (untick \"Only photos looking into the map's area\" there to "
                                          "ignore it)")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._draw_msg = f"❌ {e}"
            self.redraw()
        threading.Thread(target=work, daemon=True).start()

    def _export_geotiff(self):
        path = self.app.path
        if not path or ortho.current(path) is None:
            self._msg = "render the orthophoto first (Render orthophoto)"
            return
        try:
            out, epsg = ortho.geotiff(path, os.path.join(path, "export", "ortho.tif"))
            self._msg = f"✅ `{os.path.relpath(out, path)}` written (EPSG:{epsg}, RGBA)"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._msg = f"❌ {e}"

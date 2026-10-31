"""GUI Search tab (left, after View): compute C-RADIOv3 features for the splat (a job), then type a text and
see where it is in the splat: matching Gaussians tinted red, or only the matches shown (modelforge.semantic).
The splat file is never changed; only the colours / selection sent to the browser.

Objects: the matches grouped into instances (semantic.objects), each marked by a bright cylinder (rings +
struts, slowly turning about the vertical) with a label, listed in a dropdown; Go glides the camera to the
photo viewpoint that saw the object best (looking at it), else to an elevated view that fits it. A new
search goes to the best object at once. `preload` (called when the GUI starts) loads the text model in the
background, so the first search does not wait for it. Every job (SfM, dense, splat, meshes, the features)
unloads it (`unload`, ~2.7 GB RAM + ~1 GB GPU); it loads again on the first keystroke in Query (viser does
not report switching tabs, typing is the first sign of using the tab) or at the latest when searching."""

import contextlib
import threading
import time
import traceback

import numpy as np
import viser

from modelforge import semantic

RED = np.array([1.0, 0.1, 0.1], np.float32)
MODES = ("highlight in red", "matches only")
MARKER_COLOR = (255, 230, 0)  # bright yellow: stands out against terrain and the red highlight
SPIN_DEG_PER_S = 30.0


class SemanticPanel:
    def __init__(self, app, gui):
        self.app = app
        self.feats, self.encoder, self.rel, self.query = None, semantic.TextEncoder(), None, ""
        self.means, self.objs, self._markers, self._frames, self._spin = None, [], [], [], None
        self.surface = None  # semantic.Surface (distances to the mesh, measured for the matches), False: no mesh
        self.tree = None  # semantic_tree.Tree: the search scores only the Gaussians that can match
        self._emb, self._floor, self._scored = None, None, None  # the last query's embeddings, floor, share scored
        with contextlib.nullcontext():  # the panel fills its own tab
            self.g_md = gui.add_markdown("")
            self.g_text_md = gui.add_markdown("")
            self.g_compute = gui.add_button(
                "Compute semantic features", icon=viser.Icon.BRAIN,
                hint="C-RADIOv3 features of every photo (2048 px, text-aligned through SigLIP2), projected onto the "
                     "splat's Gaussians with an occlusion test: splat/semantic/ (the splat itself is unchanged). "
                     "About a minute for ~60 photos; the first run downloads the models (~5 GB).")
            self.g_compute.on_click(lambda _: self.app.run_stage("semantic"))
            self.g_in_all = gui.add_checkbox(
                "Include in Run full pipeline", initial_value=True,
                hint="Run full pipeline computes the semantic features right after the splat (and the mesh from "
                     "it), so a full run ends with a searchable splat. Skipped without the RADIO packages.")
            # multiline only to see Enter: viser's text box has no submit event, but a newline arrives
            self.g_query = gui.add_text("Query", initial_value="", multiline=True,
                                        hint="What to look for, in words: 'a car', 'gravel pile', 'a roof'. "
                                             "Enter searches.")
            self.g_query.on_update(lambda e: self._query_typed(e.client))
            self.g_search = gui.add_button("Search", icon=viser.Icon.SEARCH,
                                           hint="Score the Gaussians against the query: through the features' "
                                                "octree, only those that can match. The first search loads the "
                                                "text model (~1 min); later ones take a moment.")
            self.g_search.on_click(lambda e: self.search(e.client))
            self.g_thr = gui.add_slider(
                "Threshold", min=0.5, max=0.95, step=0.01, initial_value=0.6,
                hint="Relevance (0.5 = no better than generic words like 'object' or 'stuff', 1 = clearly the "
                     "query) above which a Gaussian counts as a match. Raise it when too much lights up.")
            self.g_auto = gui.add_checkbox(
                "Auto threshold", initial_value=True,
                hint="Each search sets the threshold itself: lowered from the best match until noise specks start "
                     "to appear or objects merge (semantic.auto_threshold); a best match below 0.6 means the query "
                     "is probably not in the scene. Move the slider afterwards to adjust.")
            self.g_near = gui.add_checkbox(
                "Near the mesh only", initial_value=True,
                hint="Ignore matching Gaussians farther than the tolerance from the mesh (the LiDAR mesh, else the "
                     "mesh from the splat, the SfM mesh, COLMAP's): floaters, haze and stray specks drop out, objects come out cleaner. Untick "
                     "for things the mesh misses (distant terrain beyond its range, wires, glossy surfaces).")
            self.g_tol = gui.add_number("Mesh tolerance (voxels)", initial_value=3.0, min=0.5, step=0.5,
                                        hint="How far from the mesh a match may be, in the mesh's voxels (the "
                                             "LiDAR / splat mesh's TSDF voxel, e.g. 2 / 11 cm).")
            self.g_mode = gui.add_dropdown("Show", options=MODES, initial_value=MODES[0],
                                           hint="highlight in red: the whole splat, matches tinted by relevance. "
                                                "matches only: hide everything else.")
            self.g_clear = gui.add_button("Clear", hint="Back to the normal splat colours, no markers.")
            self.g_clear.on_click(lambda _: self.clear())
            self.g_result = gui.add_markdown("")
            self.g_objects = gui.add_dropdown(
                "Object", options=("-",),
                hint="The matches grouped into separate objects (connected in 3D), strongest first: size "
                     "(diameter) and mean relevance.")
            self.g_nav = gui.add_button_group("", options=("◀", "Go", "▶"),
                                              hint="Go: fly the camera to the selected object (from the photo "
                                                   "that saw it best). ◀ / ▶: the previous / next object.")
            self.g_nav.on_click(lambda e: self._nav(e.target.value, e.client))
            self.g_markers = gui.add_checkbox("Object markers", initial_value=True,
                                              hint="A bright turning cylinder around each object, with its rank.")
            self.g_min = gui.add_number("Min object size (Gaussians)", initial_value=30, min=1, step=10,
                                        hint="Smaller groups of matches are ignored as specks.")
            for h in (self.g_thr, self.g_mode, self.g_near, self.g_tol):
                h.on_update(lambda _: self._redraw())
            for h in (self.g_thr, self.g_markers, self.g_min, self.g_near, self.g_tol):
                h.on_update(lambda _: self._update_objects())

    # ------------------------------------------------------------ state
    def refresh(self):
        """Status of the project's features (after a project switch, a new splat or a compute)."""
        self.feats, self.rel, self.means, self.surface, self.tree = None, None, None, None, None
        self._set_objects([])
        st = semantic.status(self.app.path) if self.app.path else "none"
        why = semantic.available()
        self.g_compute.disabled = why is not None or not (self.app.dense or {}).get("splat")
        self.g_search.disabled = st != "ready"
        if why:
            self.g_md.content = f"unavailable: {why}"
        elif not (self.app.dense or {}).get("splat"):
            self.g_md.content = "no splat yet (Splat tab)"
        else:
            self.g_md.content = {"none": "no semantic features yet: **Compute semantic features**",
                                 "stale": "⚠️ the splat changed since the features were computed: compute again",
                                 "ready": "features ready: type a query and **Search**"}[st]
        self.g_result.content = ""
        if st == "ready" and why is None:  # load features + splat positions now, not at the first search
            path = self.app.path
            threading.Thread(target=self._warm, args=(path,), daemon=True).start()

    def _warm(self, path):
        try:
            if path == self.app.path:
                self._features()
                self._splat_means()
                self._tree()
                srf = self._surface()
                if srf is not None:
                    srf.index()  # the k-d tree (seconds for large meshes) before the first search
        except Exception:  # noqa: BLE001 - the search loads them again and reports errors
            traceback.print_exc()

    def _features(self):
        if self.feats is None:
            self.feats = semantic.Features(self.app.path)
        return self.feats

    def _tree(self):
        """The features' octree (semantic_tree; built and saved when missing, ~3 s per 200k Gaussians), or None."""
        if self.tree is None:
            from modelforge import semantic_tree
            try:
                self.tree = semantic_tree.load_or_build(self.app.path, self._features(), self._splat_means())
            except Exception:  # noqa: BLE001 - the search then scores every Gaussian
                traceback.print_exc()
                self.tree = False
        return self.tree or None

    def _score(self, floor):
        """The relevance of the last query: through the octree exactly where >= floor (0 below), else every
        Gaussian."""
        tree, feats = self._tree(), self._features()
        self._floor = floor if tree is not None else None
        if tree is None:
            self._scored = None
            return feats.relevance(self._emb[0], self._emb[1:])
        rel, st = tree.relevance(feats, self._emb[0], self._emb[1:], floor)
        self._scored = st["scored"] / max(1, st["seen"])
        return rel

    # ------------------------------------------------------------ search
    def unload(self, reason=""):
        """Stop the text worker (a reconstruction starts and needs the memory)."""
        if self.encoder.proc is not None and self.encoder.proc.poll() is None:
            self.encoder.close()
            self.encoder.proc = None
            self.g_text_md.content = (f"text model: unloaded for {reason or 'a job'} (loads again when you type a "
                                      "query or search)")

    def ensure_loaded(self):
        """Load the text model in the background unless it is loaded or loading."""
        if (self.encoder.proc is None or self.encoder.proc.poll() is not None) and not self.encoder.lock.locked():
            self.preload()

    def preload(self):
        """Load the text model in the background (GUI start), so the first search is quick."""
        if semantic.available() is not None:
            return

        def work():
            self.g_text_md.content = "text model: ⏳ loading in the background …"
            try:
                self.encoder.encode(["warm-up"])
                self.g_text_md.content = "text model: ✅ ready"
            except Exception as e:  # noqa: BLE001 - shown in the GUI; a search retries
                if self.encoder.proc is not None:  # (None: unloaded for a job meanwhile, not a failure)
                    self.g_text_md.content = f"text model: ❌ {e}"
        threading.Thread(target=work, daemon=True).start()

    def _query_typed(self, client):
        """Enter in the Query box (a newline) starts the search; any typing loads the text model if needed."""
        self.ensure_loaded()
        v = self.g_query.value
        if "\n" in v:
            self.g_query.value = " ".join(v.split()).strip()
            self.search(client)

    def search(self, client=None):
        q = self.g_query.value.strip()
        if not q:
            self.g_result.content = "type a query first"
            return
        self.g_search.disabled = True

        def work():
            t0 = time.time()
            try:
                if self.encoder.proc is None or self.encoder.lock.locked():
                    self.g_result.content = "⏳ waiting for the text model to load …"
                emb = self.encoder.encode([q] + list(semantic.NEGATIVES))
                self.g_result.content = "⏳ scoring …"
                self._emb = emb
                self.rel, self.query = self._score(min(float(self.g_thr.value), semantic.AUTO_FLOOR)), q
                absent = False
                if self.g_auto.value:
                    thr, why = semantic.auto_threshold(self._splat_means(), self._rel()[0],
                                                       min_gaussians=int(self.g_min.value))
                    absent = why == "absent"
                    self.g_thr.value = (self.g_thr.max if absent else  # nothing to highlight
                                        float(np.clip(round(thr, 2), self.g_thr.min, self.g_thr.max)))
                self._redraw()
                self._update_objects()
                if absent:
                    self._set_objects([])
                    top = float(self._rel()[0].max())
                    best = f"{top:.2f}" if top >= (self._floor or 0) else f"below {self._floor:.2f}"
                    self.g_result.content = (f"**{q}**: probably not in the scene (best relevance {best} < "
                                             f"{semantic.ABSENT})")
                self.g_result.content += f" · {time.time() - t0:.1f} s"
                if self._scored is not None:
                    self.g_result.content += f" (octree: {100 * self._scored:.1f} % of the Gaussians scored)"
                self.g_text_md.content = "text model: ✅ ready"
                if self.objs and client is not None and not absent:  # a new search: go to the best object
                    self.g_objects.value = self.g_objects.options[0]
                    self.go(0, client)
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                traceback.print_exc()
                self.g_result.content = f"❌ {e}"
            finally:
                self.g_search.disabled = semantic.status(self.app.path) != "ready"
        threading.Thread(target=work, daemon=True).start()

    def clear(self):
        self.rel, self.query, self._emb = None, "", None
        self.g_result.content = ""
        self._set_objects([])
        self._redraw()

    def clear_for_switch(self):
        """Another project opens: drop the search, its markers and the loaded features at once (the markers
        are scene nodes outside the app's layers, so clearing the scene does not remove them)."""
        self.feats, self.rel, self.means, self.surface, self.query = None, None, None, None, ""
        self.tree, self._emb = None, None
        self.g_result.content = ""
        self._set_objects([])

    # ------------------------------------------------------------ objects
    def _surface(self):
        """The Gaussians' distances to the mesh (semantic.Surface: measured for the matches only), or None."""
        if self.surface and not self.surface.current(self.app.path):
            self.surface = None  # the mesh (e.g. patched) or the splat changed
        if self.surface is None:
            try:
                self.surface = semantic.Surface(self.app.path, self._splat_means())
            except (FileNotFoundError, KeyError):
                self.surface = False
        return self.surface or None

    def _rel(self):
        """The relevance in use: matches farther than the tolerance from the mesh count as 0 (Near the mesh
        only, with a mesh). (relevance, Gaussians dropped above the threshold)."""
        if self.rel is None:
            return None, 0
        floor = getattr(self, "_floor", None)
        if floor is not None and self._emb is not None and float(self.g_thr.value) < floor:
            self.rel = self._score(float(self.g_thr.value))  # the slider went below what the octree scored
        srf = self._surface() if self.g_near.value else None
        if srf is None or len(srf.d) != len(self.rel):
            return self.rel, 0
        need = self.rel >= min(float(self.g_thr.value), semantic.AUTO_FLOOR)  # the rest never counts as a match
        far = need & (srf.distance(need) > float(self.g_tol.value) * srf.voxel)
        dropped = int(((self.rel >= self.g_thr.value) & far).sum())
        return np.where(far, 0.0, self.rel), dropped

    def _summary(self, objs=None):
        rel, dropped = self._rel()
        n = int((rel >= self.g_thr.value).sum())
        text = f"**{self.query}**: {n:,} Gaussians above {self.g_thr.value:.2f}"
        if dropped:
            text += f" ({dropped:,} off the mesh ignored)"
        elif self.g_near.value and self._surface() is None:
            text += " (no mesh: Near the mesh only has nothing to prune with)"
        return text + (f" · **{len(objs)} object(s)**" if objs is not None else "")

    def _splat_means(self):
        if self.means is None:
            import plyfile
            v = plyfile.PlyData.read(self.app.dense["splat"])["vertex"].data
            self.means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        return self.means

    def _update_objects(self):
        if self.rel is None or not (self.app.dense or {}).get("splat"):
            self._set_objects([])
            return
        try:
            objs = semantic.objects(self._splat_means(), self._rel()[0], float(self.g_thr.value),
                                    min_gaussians=int(self.g_min.value))
        except Exception as e:  # noqa: BLE001 - shown in the GUI
            traceback.print_exc()
            self.g_result.content += f"  \n❌ objects: {e}"
            return
        self._set_objects(objs)
        if self.rel is not None:
            self.g_result.content = self._summary(objs)

    def _set_objects(self, objs):
        self.objs = objs
        self.g_objects.options = tuple(f"#{k + 1} · {2 * o['radius']:.1f} m · {o['relevance']:.2f} · {o['count']:,} G"
                                       for k, o in enumerate(objs)) or ("-",)
        self._draw_markers()

    def _draw_markers(self):
        scene = self.app.server.scene
        for name in self._markers:
            scene.remove_by_name(name)
        self._markers, self._frames = [], []
        if not self.g_markers.value:
            return
        for k, o in enumerate(self.objs):
            name = f"/semantic/object_{k}"
            r, h = o["radius"] * 1.15, (o["z1"] - o["z0"]) * 1.1 + 0.2 * o["radius"]
            frame = scene.add_frame(name, show_axes=False,
                                    position=(float(o["center"][0]), float(o["center"][1]), float(o["z0"]) - 0.05 * h))
            scene.add_line_segments(f"{name}/cylinder", cylinder_segments(r, h), colors=MARKER_COLOR,
                                    thickness=3.0, thickness_units="screen")
            scene.add_label(f"{name}/label", f"#{k + 1} {self.query} · {o['relevance']:.2f}",
                            position=(0.0, 0.0, h * 1.15))
            self._markers.append(name)
            self._frames.append(frame)
        if self._spin is None:  # one thread for the panel's lifetime (a restart per redraw raced)
            self._spin = threading.Thread(target=self._spin_markers, daemon=True)
            self._spin.start()

    def _spin_markers(self, rate_hz=20.0):
        """Turn the marker cylinders slowly about the vertical (idle while there are none)."""
        t0 = time.time()
        while True:
            if not self._frames:
                time.sleep(0.5)
                continue
            a = np.radians(SPIN_DEG_PER_S * (time.time() - t0)) / 2
            wxyz = (float(np.cos(a)), 0.0, 0.0, float(np.sin(a)))
            for f in list(self._frames):
                try:
                    f.wxyz = wxyz
                except Exception:  # noqa: BLE001 - removed meanwhile
                    pass
            time.sleep(1.0 / rate_hz)

    def _nav(self, action, client):
        if not self.objs:
            return
        opts = self.g_objects.options
        k = opts.index(self.g_objects.value) if self.g_objects.value in opts else 0
        if action in ("◀", "▶"):
            k = (k + (1 if action == "▶" else -1)) % len(self.objs)
            self.g_objects.value = opts[k]
        self.go(k, client)

    def go(self, k, client):
        """Glide the camera to object k: from the photo that saw it best, else an elevated view fitting it."""
        o = self.objs[k]
        target = np.array([o["center"][0], o["center"][1], (o["z0"] + o["z1"]) / 2])
        pos = best_view(self.app.snap, target, self.app.fov, self.app.aspect, o["radius"])
        if pos is None:
            size = max(o["radius"], (o["z1"] - o["z0"]) / 2)
            pos = target + np.array([0.0, -1.0, 0.8]) / np.linalg.norm([0.0, -1.0, 0.8]) * max(4 * size, 2.0)
        self.app.glide_to(client, pos, target)

    def _redraw(self):
        self.app.redraw_splat_decorated()
        if self.rel is not None:
            self.g_result.content = self._summary()

    def decorate(self, centers, cov, rgb, op, idx, parent=None):
        """The splat arrays to send, with the current search applied (idx: Gaussians shown of the file; with parent,
        idx are merged ones: parent[j] = the shown Gaussian idx[j] went into, which gets its best score)."""
        rel = self._rel()[0]
        if rel is None or len(rel) <= (idx.max() if len(idx) else 0):
            return centers, cov, rgb, op
        r = rel[idx]
        if parent is not None:
            best = np.zeros(len(centers), r.dtype)
            np.maximum.at(best, parent, r)
            r = best
        thr = float(self.g_thr.value)
        if self.g_mode.value == "matches only":
            keep = r >= thr
            return centers[keep], cov[keep], rgb[keep], op[keep]
        a = np.clip((r - thr) / max(1e-6, 1.0 - thr) * 0.7 + 0.3, 0, 1) * (r >= thr)  # tint 30-100 %
        out = np.array(rgb, np.float32)  # with SH columns (view-dependent colours) the tint fades them too
        out[:, :3] = rgb[:, :3] * (1 - a[:, None]) + RED * a[:, None]
        out[:, 3:] *= (1 - a[:, None])
        return centers, cov, out, op


def cylinder_segments(r, h, n=48, struts=8):
    """Line segments (N, 2, 3) of an upright cylinder of radius r and height h standing on z = 0: two rings and
    `struts` vertical lines (so its turning is visible)."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    segs = []
    for z in (0.0, h):
        p = np.c_[ring, np.full(n + 1, z)]
        segs.append(np.stack([p[:-1], p[1:]], 1))
    b = np.linspace(0, 2 * np.pi, struts, endpoint=False)
    bottom = np.stack([r * np.cos(b), r * np.sin(b), np.zeros(struts)], 1)
    segs.append(np.stack([bottom, bottom + np.array([0.0, 0.0, h])], 1))
    return np.concatenate(segs).astype(np.float32)


def best_view(snap, target, vfov, aspect, radius):
    """Centre of the registered camera that saw `target` best (in its frame with a margin, near, close to
    its optical axis), pulled back if it is closer than ~2.5 object radii; None if no camera saw it."""
    if snap is None:
        return None
    cams = snap.cams
    ok = np.flatnonzero(cams.valid)
    if not len(ok):
        return None
    C, R = cams.centers[ok], cams.R_wc[ok]
    v = target - C
    pc = np.einsum("nji,nj->ni", R, v)  # camera frame: R_wc^T v
    z = pc[:, 2]
    tx, ty = np.tan(vfov / 2) * aspect, np.tan(vfov / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = pc[:, 0] / z, pc[:, 1] / z
    seen = (z > 0) & (np.abs(xn) < 0.8 * tx) & (np.abs(yn) < 0.8 * ty)
    if not seen.any():
        return None
    dist = np.linalg.norm(v, axis=1)
    off = np.hypot(xn / tx, yn / ty)
    score = np.where(seen, dist * (1 + off), np.inf)
    i = int(np.argmin(score))
    c, d = C[i], dist[i]
    want = max(2.5 * radius, 1.0)
    return c if d >= want else target - v[i] / d * want

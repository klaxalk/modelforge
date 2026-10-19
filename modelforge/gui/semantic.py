"""GUI View tab → "Semantic search": compute C-RADIOv3 features for the splat (a job), then type a text and
see where it is in the splat: matching Gaussians tinted red, or only the matches shown (modelforge.semantic).
The splat file is never changed; only the colours / selection sent to the browser.

Objects: the matches grouped into instances (semantic.objects), each marked by a bright cylinder (rings +
struts, slowly turning about the vertical) with a label, listed in a dropdown; Go glides the camera to the
photo viewpoint that saw the object best (looking at it), else to an elevated view that fits it."""

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
        with gui.add_folder("Semantic search", expand_by_default=False):
            self.g_md = gui.add_markdown("")
            self.g_compute = gui.add_button(
                "Compute semantic features", icon=viser.Icon.BRAIN,
                hint="C-RADIOv3 features of every photo (2048 px, text-aligned through SigLIP2), projected onto the "
                     "splat's Gaussians with an occlusion test: splat/semantic/ (the splat itself is unchanged). "
                     "About a minute for ~60 photos; the first run downloads the models (~5 GB).")
            self.g_compute.on_click(lambda _: self.app.run_stage("semantic"))
            self.g_query = gui.add_text("Query", initial_value="",
                                        hint="What to look for, in words: 'a car', 'gravel pile', 'a roof'.")
            self.g_search = gui.add_button("Search", icon=viser.Icon.SEARCH,
                                           hint="Score every Gaussian against the query. The first search loads "
                                                "the text model (~1 min); later ones take a moment.")
            self.g_search.on_click(lambda _: self.search())
            self.g_thr = gui.add_slider(
                "Threshold", min=0.5, max=0.95, step=0.01, initial_value=0.6,
                hint="Relevance (0.5 = no better than generic words like 'object' or 'stuff', 1 = clearly the "
                     "query) above which a Gaussian counts as a match. Raise it when too much lights up.")
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
            for h in (self.g_thr, self.g_mode):
                h.on_update(lambda _: self._redraw())
            for h in (self.g_thr, self.g_markers, self.g_min):
                h.on_update(lambda _: self._update_objects())

    # ------------------------------------------------------------ state
    def refresh(self):
        """Status of the project's features (after a project switch, a new splat or a compute)."""
        self.feats, self.rel, self.means = None, None, None
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

    def _features(self):
        if self.feats is None:
            self.feats = semantic.Features(self.app.path)
        return self.feats

    # ------------------------------------------------------------ search
    def search(self):
        q = self.g_query.value.strip()
        if not q:
            self.g_result.content = "type a query first"
            return
        self.g_search.disabled = True

        def work():
            t0 = time.time()
            try:
                if self.encoder.proc is None:
                    self.g_result.content = "⏳ loading the text model (first search, about a minute) …"
                emb = self.encoder.encode([q] + list(semantic.NEGATIVES))
                self.g_result.content = "⏳ scoring …"
                self.rel, self.query = self._features().relevance(emb[0], emb[1:]), q
                self._redraw()
                self._update_objects()
                self.g_result.content += f" · {time.time() - t0:.1f} s"
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                traceback.print_exc()
                self.g_result.content = f"❌ {e}"
            finally:
                self.g_search.disabled = semantic.status(self.app.path) != "ready"
        threading.Thread(target=work, daemon=True).start()

    def clear(self):
        self.rel, self.query = None, ""
        self.g_result.content = ""
        self._set_objects([])
        self._redraw()

    # ------------------------------------------------------------ objects
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
            objs = semantic.objects(self._splat_means(), self.rel, float(self.g_thr.value),
                                    min_gaussians=int(self.g_min.value))
        except Exception as e:  # noqa: BLE001 - shown in the GUI
            traceback.print_exc()
            self.g_result.content += f"  \n❌ objects: {e}"
            return
        self._set_objects(objs)
        if self.rel is not None:
            self.g_result.content = (f"**{self.query}**: {int((self.rel >= self.g_thr.value).sum()):,} Gaussians above "
                                     f"{self.g_thr.value:.2f} · **{len(objs)} object(s)**")

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
            n = int((self.rel >= self.g_thr.value).sum())
            self.g_result.content = f"**{self.query}**: {n:,} Gaussians above {self.g_thr.value:.2f}"

    def decorate(self, centers, cov, rgb, op, idx):
        """The splat arrays to send, with the current search applied (idx: Gaussians shown of the file)."""
        if self.rel is None or len(self.rel) <= (idx.max() if len(idx) else 0):
            return centers, cov, rgb, op
        r = self.rel[idx]
        thr = float(self.g_thr.value)
        if self.g_mode.value == "matches only":
            keep = r >= thr
            return centers[keep], cov[keep], rgb[keep], op[keep]
        a = np.clip((r - thr) / max(1e-6, 1.0 - thr) * 0.7 + 0.3, 0, 1) * (r >= thr)  # tint 30-100 %
        rgb = rgb * (1 - a[:, None]) + RED * a[:, None]
        return centers, cov, rgb.astype(np.float32), op


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

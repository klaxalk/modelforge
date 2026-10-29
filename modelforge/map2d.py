"""The 2D map (gui/map.py; the user asked, 2026-10-27): a north-up view of the project's ENU frame with a
background of map tiles (OpenStreetMap streets or Esri aerial imagery, fetched by the server and cached under
ROOT/.map_tiles), and the data its layers show: photo coverage per ground cell, the overlap between neighbouring
photos, held-out quality, Difix's thin spots, photos left out as isolated, photos inside a drawn area, and the
pieces of the flights that look into a drawn rectangle (ingest).

Nothing here draws widgets; gui/map.py composes the image."""

import json
import math
import os
import queue
import threading
import time
import urllib.request

import numpy as np

from modelforge import geo

STYLES = {  # background tiles; attributions are shown under the map (both providers ask for them)
    "streets": {"url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "max_zoom": 19, "ext": "png",
                "attribution": "© OpenStreetMap contributors"},
    "aerial": {"url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               "max_zoom": 19, "ext": "jpg", "attribution": "Imagery © Esri, Maxar, Earthstar Geographics"},
}
USER_AGENT = "modelforge/0.1 (drone photogrammetry GUI; https://github.com/klaxalk/modelforge)"
RETRY_S = 300.0  # a tile that failed to load is asked for again after this long
EMPTY = (226, 226, 226)  # map background where no tile (yet) / no geo reference


class View:
    """The map's window onto ENU x / y: centre (cx, cy), metres per pixel, size in pixels. North up."""

    def __init__(self, cx, cy, mpp, w, h):
        self.cx, self.cy, self.mpp, self.w, self.h = float(cx), float(cy), float(mpp), int(w), int(h)

    def to_px(self, xy):
        xy = np.asarray(xy, float).reshape(-1, np.shape(xy)[-1])
        return np.stack([(xy[:, 0] - self.cx) / self.mpp + self.w / 2,
                         (self.cy - xy[:, 1]) / self.mpp + self.h / 2], 1)

    def to_world(self, u, v):
        return np.array([self.cx + (u - self.w / 2) * self.mpp, self.cy - (v - self.h / 2) * self.mpp])

    def bounds(self):
        """(x0, y0, x1, y1) in ENU."""
        hw, hh = self.w / 2 * self.mpp, self.h / 2 * self.mpp
        return self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh

    def key(self):
        return (round(self.cx, 3), round(self.cy, 3), round(self.mpp, 6), self.w, self.h)

    @classmethod
    def fit(cls, xy, w, h, margin=0.08, min_m=20.0):
        """A view showing all of xy (n, 2) with a margin."""
        xy = np.asarray(xy, float)
        xy = xy[np.isfinite(xy).all(1)] if len(xy) else xy
        if not len(xy):
            return cls(0.0, 0.0, min_m / min(w, h), w, h)
        lo, hi = xy.min(0), xy.max(0)
        span = np.maximum(hi - lo, min_m) * (1 + 2 * margin)
        return cls((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(span[0] / w, span[1] / h), w, h)


class TileCache:
    """Map tiles on disk (ROOT/.map_tiles/<style>/<z>/<x>/<y>.<ext>), fetched in the background by one thread
    (the tile servers' usage policies ask for few parallel requests); on_tile() is called after each arrival."""

    def __init__(self, root, on_tile=None):
        self.root = os.path.join(root, ".map_tiles")
        self.on_tile = on_tile
        self._q, self._wanted, self._failed = queue.Queue(), set(), {}
        self._lock = threading.Lock()
        threading.Thread(target=self._fetcher, daemon=True).start()

    def path(self, style, z, x, y):
        return os.path.join(self.root, style, str(z), str(x), f"{y}.{STYLES[style]['ext']}")

    def get(self, style, z, x, y):
        """The tile as an RGB array, or None (then it is fetched; on_tile follows)."""
        import cv2
        p = self.path(style, z, x, y)
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None:
                return im[:, :, ::-1]
        key = (style, z, x, y)
        with self._lock:
            if key not in self._wanted and time.time() - self._failed.get(key, 0.0) > RETRY_S:
                self._wanted.add(key)
                self._q.put(key)
        return None

    def _fetcher(self):
        while True:
            key = self._q.get()
            style, z, x, y = key
            p = self.path(*key)
            try:
                url = STYLES[style]["url"].format(z=z, x=x, y=y)
                data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                              timeout=15).read()
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p + ".part", "wb") as f:
                    f.write(data)
                os.replace(p + ".part", p)
            except Exception:  # noqa: BLE001 - offline, refused: the map shows grey there
                with self._lock:
                    self._failed[key] = time.time()
            with self._lock:
                self._wanted.discard(key)
            if self.on_tile is not None and self._q.empty():
                try:
                    self.on_tile()
                except Exception:  # noqa: BLE001
                    pass


def basemap(view, origin, style, cache, grid=17):
    """The background of `view` from `style` tiles: (H, W, 3) uint8 and whether every tile was there. The ENU
    pixel grid is mapped to Web Mercator exactly on a coarse grid, a quadratic in the pixel position fitted to it
    (smooth over a view's extent; cv2.resize of the grid would clamp the edges into streaks), then the tiles are
    sampled (cv2.remap, bilinear)."""
    import cv2
    out = np.full((view.h, view.w, 3), EMPTY, np.uint8)
    if origin is None or style not in STYLES:
        return out, True
    lat0 = float(origin[0])
    z = int(np.clip(math.ceil(math.log2(geo.mercator_mpp(lat0, 0) / max(view.mpp, 1e-3))), 1,
                    STYLES[style]["max_zoom"]))
    U, V = np.meshgrid(np.linspace(0, view.w, grid), np.linspace(0, view.h, grid))
    w = view.to_world(U.ravel(), V.ravel())
    lat, lon, _ = geo.enu_to_lla(np.stack(w, 1), origin)
    mx, my = geo.mercator_px(lat, lon, z)

    def terms(u, v):  # a quadratic in the pixel position (exact to far below a pixel over a view)
        u, v = u / view.w - 0.5, v / view.h - 0.5
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)
    A = terms(U.ravel(), V.ravel())
    cx, *_ = np.linalg.lstsq(A, mx - mx.mean(), rcond=None)
    cy, *_ = np.linalg.lstsq(A, my - my.mean(), rcond=None)
    T = terms(*np.meshgrid(np.arange(view.w) + 0.5, np.arange(view.h) + 0.5))
    mx, my = (T @ cx + mx.mean()).astype(np.float32), (T @ cy + my.mean()).astype(np.float32)
    tx0, ty0 = int(np.floor(mx.min() / geo.TILE)), int(np.floor(my.min() / geo.TILE))
    tx1, ty1 = int(np.floor(mx.max() / geo.TILE)), int(np.floor(my.max() / geo.TILE))
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:  # zoomed far out of a tiny view: not worth it
        return out, True
    mosaic = np.full(((ty1 - ty0 + 1) * geo.TILE, (tx1 - tx0 + 1) * geo.TILE, 3), EMPTY, np.uint8)
    complete, n = True, 2 ** z
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            if not 0 <= ty < n:
                continue
            t = cache.get(style, z, tx % n, ty)
            if t is None:
                complete = False
                continue
            t = cv2.resize(t, (geo.TILE, geo.TILE)) if t.shape[:2] != (geo.TILE, geo.TILE) else t
            mosaic[(ty - ty0) * geo.TILE:(ty - ty0 + 1) * geo.TILE, (tx - tx0) * geo.TILE:(tx - tx0 + 1) * geo.TILE] = t
    out = cv2.remap(mosaic, mx - tx0 * geo.TILE, my - ty0 * geo.TILE, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=EMPTY)
    return out, complete


# --- layer data ----------------------------------------------------------------------------------------------

def coverage(centers, R_wc, K, size, ground, bounds, cell, max_cams=4000):
    """Photos seeing each ground cell: (grid of counts (ny, nx), (x0, y0, cell)). centers / R_wc: the registered
    cameras (OpenCV, world_from_cam); K = (fx, fy, cx, cy) at size (w, h); ground(xy) -> z (pairs.Terrain.z or a
    constant). A cell counts a photo when its centre on the ground projects into the frame in front of it (no
    occlusion test, as the splat plan's survey)."""
    x0, y0, x1, y1 = bounds
    nx, ny = max(1, int(math.ceil((x1 - x0) / cell))), max(1, int(math.ceil((y1 - y0) / cell)))
    gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * cell, y0 + (np.arange(ny) + 0.5) * cell)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    P = np.c_[P, ground(P)]
    keep = np.isfinite(centers[:, 0])
    C, R = centers[keep], R_wc[keep]
    if len(C) > max_cams:  # a uniform subset, scaled back
        pick = np.linspace(0, len(C) - 1, max_cams).astype(int)
        C, R, scale = C[pick], R[pick], len(C) / max_cams
    else:
        scale = 1.0
    fx, fy, cx, cy = K
    w, h = size
    count = np.zeros(len(P), np.float32)
    for a in range(0, len(C), 64):
        d = P[None, :, :] - C[a:a + 64, None, :]  # (c, p, 3)
        q = np.einsum("cji,cpj->cpi", R[a:a + 64], d)  # camera frame: R_wc^T d
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = fx * q[..., 0] / z + cx, fy * q[..., 1] / z + cy
        count += ((z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).sum(0)
    return (count * scale).reshape(ny, nx), (x0, y0, cell)


def neighbour_overlap(project, names, centers):
    """Overlap between consecutive photos in use (sfm/overlap.py analysis): [(i, j, overlap)] with indices into
    names (capture order per source), or None without an analysis."""
    from modelforge.sfm import overlap
    data = overlap.load(project)
    if not data:
        return None
    ov = overlap.Overlap(data)
    _, poses = overlap.pool(project)
    try:
        images = json.load(open(poses))["images"]
    except (OSError, ValueError, KeyError):
        return None
    index = {n: i for i, n in enumerate(names)}
    out = []
    for seq in overlap.ordered(images).values():
        s = [n for n in seq if n in index and np.isfinite(centers[index[n], 0])]
        out += [(index[a], index[b], ov(a, b)) for a, b in zip(s, s[1:])]
    return out


def heldout(project):
    """Held-out quality (evaluate.run's evaluation/metrics.json): [(image, psnr)], or None."""
    try:
        per = json.load(open(os.path.join(project, "evaluation", "metrics.json")))["per_image"]
    except (OSError, ValueError, KeyError):
        return None
    return [(m["image"], float(m["psnr"])) for m in per if "image" in m and "psnr" in m]


def pool_lookats(project):
    """Every extracted photo (the subset pool included) with its prior centre and look-at point (ENU): (names,
    centres (n, 3), look-ats (n, 3)). The look-at is where the optical axis meets the terrain of the project's sparse
    points (sparse_pc.ply), else the takeoff plane, as for pairs. Photos without a pose prior are left out."""
    from modelforge.sfm import overlap, pairs, priors as priors_mod
    _, poses = overlap.pool(project)
    if not os.path.exists(poses):
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    pri, meta = priors_mod.load(os.path.dirname(poses))
    pri = {n: p for n, p in pri.items() if np.isfinite(p.center).all()}
    if not pri:
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    terrain = None
    ply = os.path.join(project, "sparse_pc.ply")
    if os.path.exists(ply):
        import plyfile
        v = plyfile.PlyData.read(ply)["vertex"].data
        if len(v) >= 1000:
            terrain = pairs.Terrain(np.stack([v["x"], v["y"], v["z"]], 1))
    ground = meta["ground_z"] if "ground_z" in meta else -meta.get("enu_origin_lla", (0, 0, 0))[2]
    names, look, _, _ = pairs.look_at_points(pri, ground, terrain=terrain)
    return names, np.array([pri[n].center for n in names]), look


def isolated(project):
    """Names of the photos the export left out as isolated (report.json export.isolated.names)."""
    try:
        r = json.load(open(os.path.join(project, "report.json")))
    except (OSError, ValueError):
        return []
    return list(((r.get("export") or {}).get("isolated") or {}).get("names") or [])


def thin_spots(project, seed=0):
    """Difix's thin spots: mesh samples (x, y) where the splat is thin (difix.suggest_views' measure), with their
    density relative to the median; None without an SfM mesh or a splat."""
    from modelforge import difix
    if difix.can_suggest(project) is not None:
        return None
    rng = np.random.default_rng(seed)
    st = json.load(open(os.path.join(project, "sfmmesh", "stats.json")))
    spacing = float(st.get("point_spacing") or 0.1)
    p, _, _ = difix._mesh_samples(project, spacing, rng)
    G, op = difix._splat_centres(project)
    dens = difix._splat_density(G, op, p, 2 * spacing)
    rel = dens / max(float(np.median(dens)), 1e-9)
    low = rel < difix.THIN
    return p[low, :2], rel[low]


def raster(xy, values, bounds, cell, reduce="mean"):
    """Points binned into cells: (grid (ny, nx) with NaN where empty, (x0, y0, cell))."""
    x0, y0, x1, y1 = bounds
    nx, ny = max(1, int(math.ceil((x1 - x0) / cell))), max(1, int(math.ceil((y1 - y0) / cell)))
    xy = np.asarray(xy, float)
    i = np.floor((xy[:, 0] - x0) / cell).astype(int)
    j = np.floor((xy[:, 1] - y0) / cell).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    lin = j[ok] * nx + i[ok]
    s = np.bincount(lin, np.asarray(values, float)[ok], nx * ny)
    c = np.bincount(lin, None, nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = s / c if reduce == "mean" else s
    g[c == 0] = np.nan
    return g.reshape(ny, nx), (x0, y0, cell)


# --- drawn shapes ------------------------------------------------------------------------------------------------

def inside_polygon(xy, poly):
    """Mask of the points (n, 2) inside the polygon (m, 2) (even-odd rule)."""
    xy, poly = np.asarray(xy, float), np.asarray(poly, float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
        cross = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= cross & (x < xs)
    return inside


def runs(mask, t, min_s, gap_s):
    """Time stretches [(t0, t1)] where mask holds: gaps shorter than gap_s joined, stretches shorter than min_s
    dropped. t: the frames' times (ascending)."""
    mask, t = np.asarray(mask, bool), np.asarray(t, float)
    out = []
    k = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    for a, b in zip(k[::2], k[1::2]):
        t0, t1 = float(t[a]), float(t[b - 1])
        if out and t0 - out[-1][1] < gap_s:
            out[-1] = (out[-1][0], t1)
        else:
            out.append((t0, t1))
    return [(a, b) for a, b in out if b - a >= min_s]

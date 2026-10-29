"""Orthophoto of the splat (the map's background layer and a GeoTIFF export; the user asked, 2026-10-27): the splat
rendered straight down with an orthographic camera (export/ortho_render.py in nerfstudio's venv, gsplat), north up,
over the sparse points' extent.

    python -m modelforge.ortho PROJECT [--max-px 4096] [--geotiff OUT.tif]

Output splat/ortho.png (RGBA: alpha = the splat's coverage) + splat/ortho.json (ENU bounds, metres per pixel, the
splat's size + mtime: a new splat makes it stale). The area is the flown one (cameras + their look-at points), and
Gaussians above the sparse points' 99.5 % height + 2 m (floaters under the flight path, sky), outside the area or
very large (the background's far blobs) are left out, so the view reaches the ground. The GeoTIFF is in the UTM zone of the
project's origin (EPSG 326xx / 327xx), the ENU image mapped by an affine fit (ModelTransformationTag; the local
frame is rotated against the UTM grid by the meridian convergence, so a plain pixel scale would not do)."""

import argparse
import json
import os
import time

import numpy as np

from modelforge import jobs

PNG = os.path.join("splat", "ortho.png")
META = os.path.join("splat", "ortho.json")
MAX_PX = 4096
MIN_MPP = 0.01  # 1 cm per pixel: finer gains nothing over the photos' GSD here
BIG = 0.02  # Gaussians wider than this share of the area are left out (keep_mask)


def _key(path):
    st = os.stat(path)
    return [st.st_size, round(st.st_mtime, 3)]


def splat_path(project):
    return os.path.join(project, "splat", "splat_enu.ply")


def current(project):
    """ortho.json when the orthophoto belongs to the current splat, else None."""
    try:
        meta = json.load(open(os.path.join(project, META)))
        if meta.get("splat") == _key(splat_path(project)) and os.path.exists(os.path.join(project, PNG)):
            return meta
    except (OSError, ValueError):
        pass
    return None


def extent(project):
    """(x0, y0, x1, y1, z_cut) of the flown area: the transforms.json cameras and their look-at points (the optical
    axis on the sparse points' median height, at most 3 x the camera's height above it) + 15 % + 5 m; the height cut
    from the sparse points (99.5 % + 2 m). The sparse points' own extent reaches the horizon (fullrun_test: 300 m
    for a 50 m flight)."""
    import plyfile
    v = plyfile.PlyData.read(os.path.join(project, "sparse_pc.ply"))["vertex"].data
    z = np.asarray(v["z"], float)
    ground = float(np.median(z))
    t = json.load(open(os.path.join(project, "transforms.json")))
    M = np.array([f["transform_matrix"] for f in t["frames"]], float)
    C, axis = M[:, :3, 3], -M[:, :3, 2]  # OpenGL camera: looking along -z
    h = np.maximum(C[:, 2] - ground, 1.0)
    down = axis[:, 2] < -1e-3
    depth = np.full(len(C), 0.0)
    depth[down] = np.minimum((C[down, 2] - ground) / -axis[down, 2], 3 * h[down])
    xy = np.r_[C[:, :2], C[:, :2] + depth[:, None] * axis[:, :2]]
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.15 * (hi - lo) + 5.0
    return (*(lo - pad), *(hi + pad), float(np.percentile(z, 99.5)) + 2.0)


def keep_mask(g, bounds, z_cut):
    """Gaussians drawn: centre inside the bounds (+ 10 %) and under the height cut, and no wider (3 sigma) than
    BIG of the area's size, at least 1.5 m: the background's huge far blobs otherwise cover the ground."""
    x0, y0, x1, y1 = bounds
    m = g["means"]
    pad = 0.1 * max(x1 - x0, y1 - y0)
    size = 3 * np.exp(g["scales"].max(1))
    return ((m[:, 0] > x0 - pad) & (m[:, 0] < x1 + pad) & (m[:, 1] > y0 - pad) & (m[:, 1] < y1 + pad)
            & (m[:, 2] <= z_cut) & (size <= max(1.5, BIG * max(x1 - x0, y1 - y0))))


def run(project, max_px=MAX_PX, on_line=print, token=None):
    """Render splat/ortho.png + ortho.json; returns the meta."""
    from modelforge import splat, splat_mesh
    from modelforge.pipeline import Report
    token = token or jobs.Token()
    sp = splat_path(project)
    if not os.path.exists(sp):
        raise RuntimeError("no splat yet: train one first (Splat tab)")
    if not os.path.exists(os.path.join(project, "sparse_pc.ply")):
        raise RuntimeError("no sparse_pc.ply: run export first")
    why = splat.available()
    if why:
        raise RuntimeError(f"the orthophoto renders in nerfstudio's environment: {why}")
    t0 = time.time()
    key = _key(sp)
    x0, y0, x1, y1, z_cut = extent(project)
    mpp = max((x1 - x0) / max_px, (y1 - y0) / max_px, MIN_MPP)
    on_line(f"ortho: reading the splat; {x1 - x0:.0f} x {y1 - y0:.0f} m at {mpp * 100:.1f} cm per pixel")
    g = splat_mesh.load_splat(sp)
    keep = keep_mask(g, (x0, y0, x1, y1), z_cut)
    token.check()
    os.makedirs(os.path.join(project, "splat"), exist_ok=True)
    inp = os.path.join(project, "splat", "ortho_input.npz")
    np.savez(inp, means=g["means"][keep], quats=g["quats"][keep], scales=g["scales"][keep],
             opacities=g["opacities"][keep], sh0=g["sh"][keep, :1], bounds=np.array([x0, y0, x1, y1]), mpp=mpp,
             z_top=float(g["means"][keep, 2].max()) + 10.0)
    del g
    out = os.path.join(project, PNG)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "ortho_render.py")
    try:
        splat._stream([os.path.join(splat.NS_BIN, "python"), script, inp, out + ".part.png"],
                      lambda m: on_line(m.replace("[ortho]", "ortho:")), token)
    finally:
        if os.path.exists(inp):
            os.remove(inp)
    os.replace(out + ".part.png", out)
    meta = {"bounds": [x0, y0, x1, y1], "mpp": mpp, "splat": key, "z_cut": z_cut, "gaussians": int(keep.sum()),
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(project, META), "w"), indent=1)
    Report(project).set("ortho", meta)
    return meta


def load(project, meta=None):
    """(RGBA image (H, W, 4) uint8, meta) of a current orthophoto, or (None, None)."""
    import cv2
    meta = meta or current(project)
    if meta is None:
        return None, None
    im = cv2.imread(os.path.join(project, PNG), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None
    return cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA), meta


def geotiff(project, out, origin=None):
    """Write the orthophoto as a GeoTIFF (RGBA, UTM, see the module doc); returns (path, EPSG code)."""
    from PIL import Image, TiffImagePlugin

    from modelforge import geo
    img, meta = load(project)
    if img is None:
        raise RuntimeError("no current orthophoto: render it first (Map window → Orthophoto)")
    if origin is None:
        origin = json.load(open(os.path.join(project, "poses.json")))["enu_origin_lla"]
    zone, north, epsg = geo.utm_zone(origin[0], origin[1])
    x0, y0, x1, y1 = meta["bounds"]
    H, W = img.shape[:2]
    mpp = meta["mpp"]
    # pixel (col, row) -> ENU (x0 + col mpp, y1 - row mpp) -> UTM: an affine fit over a grid (exact to mm here)
    cc, rr = np.meshgrid(np.linspace(0, W, 9), np.linspace(0, H, 9))
    lat, lon, _ = geo.enu_to_lla(np.stack([x0 + cc.ravel() * mpp, y1 - rr.ravel() * mpp], 1), origin)
    e, n = geo.lla_to_utm(lat, lon, zone, north)
    A = np.c_[cc.ravel(), rr.ravel(), np.ones(cc.size)]
    ce, *_ = np.linalg.lstsq(A, e, rcond=None)
    cn, *_ = np.linalg.lstsq(A, n, rcond=None)
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd.tagtype[34264] = 12  # ModelTransformationTag: 4 x 4 doubles, row-major
    ifd[34264] = (ce[0], ce[1], 0.0, ce[2], cn[0], cn[1], 0.0, cn[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ifd.tagtype[34735] = 3  # GeoKeyDirectoryTag: version 1.1.0, 3 keys
    ifd[34735] = (1, 1, 0, 3,
                  1024, 0, 1, 1,  # GTModelTypeGeoKey = projected
                  1025, 0, 1, 1,  # GTRasterTypeGeoKey = PixelIsArea
                  3072, 0, 1, epsg)  # ProjectedCSTypeGeoKey
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    Image.fromarray(img, "RGBA").save(out, tiffinfo=ifd, compression="tiff_deflate")
    return out, epsg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project")
    ap.add_argument("--max-px", type=int, default=MAX_PX)
    ap.add_argument("--geotiff", help="also write the GeoTIFF here")
    a = ap.parse_args()
    if current(a.project) is None:
        run(a.project, a.max_px)
    if a.geotiff:
        print(geotiff(a.project, a.geotiff))


if __name__ == "__main__":
    main()

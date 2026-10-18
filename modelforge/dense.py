"""Dense reconstruction on the refined poses, using COLMAP's MVS (unmodified).

    undistort  sparse/refined -> dense/ (PINHOLE images + sparse model)     CPU
    stereo     PatchMatch depth/normal maps -> dense/stereo/                 CUDA only
    fusion     depth maps -> dense/fused.ply (+ .vis)                        CPU
    mesh       dense/meshed-{delaunay,poisson}.ply                           CPU
    texture    dense/textured-<method>/ (UV-mapped PLY + texture.png)        CPU
    preview    dense/preview-<method>.ply, simplified for the GUI            CPU

Everything stays in the refined world frame, i.e. ENU like transforms.json, so the
mesh lines up with the cameras (and with the splat after export.splat_to_enu).
Delaunay (visibility-based, keeps open surfaces) suits drone scenes better than
Poisson, which closes the surface; it has no pycolmap binding, so it runs through
the COLMAP CLI. QUALITY holds the speed/detail presets (default "superfast", 320 px).
"""

import os
import re
import shutil
import time

import numpy as np
import pycolmap

from modelforge import jobs

COLMAP_BIN = shutil.which("colmap") or "/opt/colmap/bin/colmap"
MESH_METHODS = ("delaunay", "poisson")

# Quality presets: image size for undistort, PatchMatch options for stereo. PatchMatch cost per
# pixel ~ num_samples x (2 window_radius + 1)^2 / window_step^2 and does not depend on the number
# of source images. Measured on berlin_test (20 views, 640 px, RTX 3090, both passes + fusion):
#   COLMAP defaults            170 s   43.5k fused points
#   num_samples 7               90 s   42.6k  (same accuracy against the SfM points)
#   + window_step 2             39 s   38.2k  (same accuracy, ~12% fewer points)
QUALITY = {
    "superfast": {"max_image_size": 320, "window_step": 2, "num_samples": 7},
    "low": {"max_image_size": 640, "window_step": 2, "num_samples": 7},
    "medium": {"max_image_size": 1200, "num_samples": 7},
    "high": {"max_image_size": -1, "num_samples": 7},
    "max": {"max_image_size": -1},  # COLMAP defaults at full resolution
}
DEFAULT_QUALITY = "superfast"  # the user wants the quickest dense by default


def workspace(project):
    return os.path.join(project, "dense")


def has_cuda():
    return bool(getattr(pycolmap, "has_cuda", False))


def undistort(project, max_image_size=-1):
    ws = workspace(project)
    if os.path.exists(ws):
        shutil.rmtree(ws)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = max_image_size
    t0 = time.time()
    pycolmap.undistort_images(ws, os.path.join(project, "sparse", "refined"), os.path.join(project, "images"),
                              output_type="COLMAP", undistort_options=opts)
    rec = pycolmap.Reconstruction(os.path.join(ws, "sparse"))
    cam = next(iter(rec.cameras.values()))
    return {"images": rec.num_reg_images(), "camera": cam.model.name, "width": int(cam.width),
            "height": int(cam.height), "seconds": round(time.time() - t0, 1)}


_VIEW = re.compile(r"Processing view (\d+) / (\d+)")


def stereo(project, max_image_size=-1, geom_consistency=True, token=None, on_progress=None, **patch_match):
    """PatchMatch on the undistorted workspace; extra keywords are PatchMatchStereo options (see QUALITY).
    Runs the colmap CLI (same code as pycolmap.patch_match_stereo), so a Stop kills it at once.
    on_progress(view, views, pass_index) follows its "Processing view i / n" lines (2 passes with
    geom_consistency: photometric, then geometric)."""
    if not has_cuda():
        raise RuntimeError("COLMAP PatchMatch stereo needs CUDA, and this pycolmap was built without it. "
                           "Run the dense stages in the CUDA container: MODELFORGE_GPU=1 scripts/dev ...")
    cmd = [COLMAP_BIN, "patch_match_stereo", "--workspace_path", workspace(project), "--workspace_format", "COLMAP",
           "--PatchMatchStereo.max_image_size", str(max_image_size),
           "--PatchMatchStereo.geom_consistency", "1" if geom_consistency else "0"]
    for k, v in patch_match.items():
        cmd += [f"--PatchMatchStereo.{k}", str(v)]
    state = {"pass": 0, "last": 0}

    def line(text):
        m = _VIEW.search(text)
        if m and on_progress:
            i, n = int(m.group(1)), int(m.group(2))
            if i < state["last"]:
                state["pass"] += 1
            state["last"] = i
            on_progress(i, n, state["pass"])
    t0 = time.time()
    (token or jobs.Token()).run(cmd, line)
    maps = os.listdir(os.path.join(workspace(project), "stereo", "depth_maps"))
    return {"depth_maps": len(maps), "seconds": round(time.time() - t0, 1)}


def fusion(project, input_type="geometric", min_num_pixels=5):
    opts = pycolmap.StereoFusionOptions()
    opts.min_num_pixels = min_num_pixels
    out = os.path.join(workspace(project), "fused.ply")
    t0 = time.time()
    # only output_type "ply" also writes fused.ply.vis, which the Delaunay mesher needs
    rec = pycolmap.stereo_fusion(out, workspace(project), input_type=input_type, options=opts, output_type="ply")
    return {"points": rec.num_points3D(), "path": out, "seconds": round(time.time() - t0, 1)}


def _read_map(path):
    """A COLMAP depth / normal map (.bin): "w&h&c&" header, then float32, row-major."""
    with open(path, "rb") as f:
        head = b""
        while head.count(b"&") < 3:
            head += f.read(1)
        w, h, c = map(int, head.decode().split("&")[:3])
        return np.fromfile(f, np.float32).reshape(c, h, w).transpose(1, 2, 0).squeeze()


def _read_vis(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), np.uint64)[0])
        out = []
        for _ in range(n):
            k = int(np.frombuffer(f.read(4), np.uint32)[0])
            out.append(np.frombuffer(f.read(4 * k), np.uint32).tolist())
    return out


def _write_vis(path, vis):
    with open(path, "wb") as f:
        f.write(np.uint64(len(vis)).tobytes())
        for v in vis:
            f.write(np.uint32(len(v)).tobytes() + np.asarray(v, np.uint32).tobytes())


def add_lidar(project, max_points=1_000_000, max_cams=5, min_cams=2, max_depth=80.0, tol=0.05, seed=0):
    """Merge the project's LiDAR cloud (kind rosbag) into dense/fused.ply (+ .vis) before meshing, so
    surfaces the images cannot reconstruct (no texture) still get geometry. A LiDAR point lists the
    cameras that see it (inside the image, in front, not behind the PatchMatch depth there by more
    than tol * depth: occluded), up to max_cams nearest; it needs min_cams. Colour from the nearest,
    normal towards them. The image-only cloud stays as dense/fused_mvs.ply. Returns stats."""
    from plyfile import PlyData, PlyElement

    from modelforge.export import lidar as lidar_mod
    ws = workspace(project)
    xyz = lidar_mod.load(project)
    if xyz is None or not len(xyz):
        return {"lidar_points": 0}
    fused, vis_path = os.path.join(ws, "fused.ply"), os.path.join(ws, "fused.ply.vis")
    mvs, mvs_vis = os.path.join(ws, "fused_mvs.ply"), os.path.join(ws, "fused_mvs.ply.vis")
    if not os.path.exists(mvs):  # keep the image-only result (re-running this replaces only the merge)
        shutil.copyfile(fused, mvs)
        shutil.copyfile(vis_path, mvs_vis)
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    rec = pycolmap.Reconstruction(os.path.join(ws, "sparse"))
    ids = list(rec.reg_image_ids())  # the order mvs::Model gives image indices in the .vis file
    n = len(xyz)
    cand_d = np.full((n, max_cams), np.inf)
    cand_i = np.full((n, max_cams), -1, np.int64)
    color = np.zeros((n, 3), np.uint8)
    t0 = time.time()
    from PIL import Image as PILImage
    for idx, image_id in enumerate(ids):
        img = rec.image(image_id)
        cam = rec.cameras[img.camera_id]
        T = img.cam_from_world().matrix()
        Xc = xyz @ T[:, :3].T + T[:, 3]
        sel = np.flatnonzero((Xc[:, 2] > 0.1) & (Xc[:, 2] < max_depth))
        if not len(sel):
            continue
        uv = cam.img_from_cam(Xc[sel])
        inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam.width - 1) & (uv[:, 1] < cam.height - 1)
        sel, uv = sel[inside], uv[inside].astype(int)
        depth_path = os.path.join(ws, "stereo", "depth_maps", f"{img.name}.geometric.bin")
        if os.path.exists(depth_path) and len(sel):  # occlusion: a nearer surface in the depth map
            d = _read_map(depth_path)
            sy, sx = d.shape[0] / cam.height, d.shape[1] / cam.width
            dm = d[np.clip((uv[:, 1] * sy).astype(int), 0, d.shape[0] - 1),
                   np.clip((uv[:, 0] * sx).astype(int), 0, d.shape[1] - 1)]
            z = Xc[sel, 2]
            ok = (dm <= 0) | (z <= dm * (1 + tol))
            sel, uv = sel[ok], uv[ok]
        if not len(sel):
            continue
        dist = np.linalg.norm(xyz[sel] - img.projection_center(), axis=1)
        worst = cand_d[sel].argmax(1)
        better = dist < cand_d[sel, worst]
        rows = sel[better]
        cand_d[rows, worst[better]] = dist[better]
        cand_i[rows, worst[better]] = idx
        nearest = dist <= cand_d[sel].min(1)
        if nearest.any():
            pix = np.asarray(PILImage.open(os.path.join(ws, "images", img.name)).convert("RGB"))
            color[sel[nearest]] = pix[uv[nearest, 1], uv[nearest, 0]]
    keep = (cand_i >= 0).sum(1) >= min_cams
    centers = np.array([rec.image(i).projection_center() for i in ids])
    vis_l = [sorted(int(i) for i in row if i >= 0) for row in cand_i[keep]]
    nrm = np.array([centers[v].mean(0) for v in vis_l]) - xyz[keep] if vis_l else np.zeros((0, 3))
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    base = PlyData.read(mvs)["vertex"].data
    add = np.zeros(int(keep.sum()), dtype=base.dtype)
    for i, c in enumerate("xyz"):
        add[c] = xyz[keep, i]
    for i, c in enumerate(("nx", "ny", "nz")):
        if c in add.dtype.names:
            add[c] = nrm[:, i]
    for i, c in enumerate(("red", "green", "blue")):
        if c in add.dtype.names:
            add[c] = color[keep, i]
    PlyData([PlyElement.describe(np.concatenate([base, add]), "vertex")]).write(fused)
    _write_vis(vis_path, _read_vis(mvs_vis) + vis_l)
    return {"lidar_points": int(keep.sum()), "lidar_candidates": n, "mvs_points": len(base),
            "seconds": round(time.time() - t0, 1)}


def mesh(project, method="delaunay", poisson_depth=13, poisson_trim=10.0, token=None):
    ws = workspace(project)
    out = os.path.join(ws, f"meshed-{method}.ply")
    t0 = time.time()
    if method == "poisson":
        opts = pycolmap.PoissonMeshingOptions()
        opts.depth = poisson_depth
        opts.trim = poisson_trim
        pycolmap.poisson_meshing(os.path.join(ws, "fused.ply"), out, options=opts)
    elif method == "delaunay":
        _colmap("delaunay_mesher", "--input_path", ws, "--input_type", "dense", "--output_path", out, token=token)
    else:
        raise ValueError(f"unknown mesh method {method!r}, expected one of {MESH_METHODS}")
    return {"method": method, "path": out, "bytes": os.path.getsize(out), "seconds": round(time.time() - t0, 1)}


def _colmap(*args, token=None):
    (token or jobs.Token()).run([COLMAP_BIN, *args])


def texture(project, method="delaunay", token=None):
    ws = workspace(project)
    out = os.path.join(ws, f"textured-{method}")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t0 = time.time()
    _colmap("mesh_texturer", "--workspace_path", ws, "--input_path", os.path.join(ws, f"meshed-{method}.ply"),
            "--output_path", out, token=token)
    return {"method": method, "path": out, "files": sorted(os.listdir(out)), "seconds": round(time.time() - t0, 1)}


def preview(project, method="delaunay", max_faces=300_000, token=None):
    """Simplified copy of the mesh, small enough to stream to the browser."""
    import plyfile
    ws = workspace(project)
    src = os.path.join(ws, f"meshed-{method}.ply")
    out = os.path.join(ws, f"preview-{method}.ply")
    n = plyfile.PlyData.read(src)["face"].count
    if n <= max_faces:
        shutil.copyfile(src, out)
    else:
        _colmap("mesh_simplifier", "--input_path", src, "--output_path", out,
                "--MeshSimplification.target_face_ratio", f"{max_faces / n:.6f}", token=token)
    return {"method": method, "path": out, "faces_in": n, "faces_out": plyfile.PlyData.read(out)["face"].count}

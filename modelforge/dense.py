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
the COLMAP CLI.
"""

import os
import shutil
import subprocess
import time

import pycolmap

COLMAP_BIN = shutil.which("colmap") or "/opt/colmap/bin/colmap"
MESH_METHODS = ("delaunay", "poisson")


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


def stereo(project, max_image_size=-1, geom_consistency=True):
    if not has_cuda():
        raise RuntimeError("COLMAP PatchMatch stereo needs CUDA, and this pycolmap was built without it. "
                           "Run the dense stages in the CUDA container: MODELFORGE_GPU=1 scripts/dev ...")
    opts = pycolmap.PatchMatchOptions()
    opts.max_image_size = max_image_size
    opts.geom_consistency = geom_consistency
    t0 = time.time()
    pycolmap.patch_match_stereo(workspace(project), options=opts)
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


def mesh(project, method="delaunay", poisson_depth=13, poisson_trim=10.0):
    ws = workspace(project)
    out = os.path.join(ws, f"meshed-{method}.ply")
    t0 = time.time()
    if method == "poisson":
        opts = pycolmap.PoissonMeshingOptions()
        opts.depth = poisson_depth
        opts.trim = poisson_trim
        pycolmap.poisson_meshing(os.path.join(ws, "fused.ply"), out, options=opts)
    elif method == "delaunay":
        _colmap("delaunay_mesher", "--input_path", ws, "--input_type", "dense", "--output_path", out)
    else:
        raise ValueError(f"unknown mesh method {method!r}, expected one of {MESH_METHODS}")
    return {"method": method, "path": out, "bytes": os.path.getsize(out), "seconds": round(time.time() - t0, 1)}


def _colmap(*args):
    subprocess.run([COLMAP_BIN, *args], check=True, capture_output=True, text=True)


def texture(project, method="delaunay"):
    ws = workspace(project)
    out = os.path.join(ws, f"textured-{method}")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t0 = time.time()
    _colmap("mesh_texturer", "--workspace_path", ws, "--input_path", os.path.join(ws, f"meshed-{method}.ply"),
            "--output_path", out)
    return {"method": method, "path": out, "files": sorted(os.listdir(out)), "seconds": round(time.time() - t0, 1)}


def preview(project, method="delaunay", max_faces=300_000):
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
                "--MeshSimplification.target_face_ratio", f"{max_faces / n:.6f}")
    return {"method": method, "path": out, "faces_in": n, "faces_out": plyfile.PlyData.read(out)["face"].count}

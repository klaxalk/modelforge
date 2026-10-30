"""Fuse LiDAR scans into a mesh (modelforge/lidar_mesh.py; runs in nerfstudio's venv: Open3D on the GPU).

Input SCANS.npz: points (N, 3) in each scan's sensor frame, grey (N,) 0-1, offsets (scans + 1,), poses (scans, 4,
4) T_world_sensor. Each scan becomes six 90° pinhole depth images around the sensor (a cube map of FACE_RES px,
z-depth, the nearest return per pixel), integrated into an Open3D VoxelBlockGrid (TSDF, truncation 4 voxels, the
intensity as grey colour): the rays carve free space, so returns from people walking by or noise that later scans
see through fade. Marching cubes (in pieces, as export/splat_tsdf.py) over voxels seen by >= min_weight scans,
pieces under 0.2 % of the faces dropped.

python export/lidar_tsdf.py SCANS.npz OUT_MESH.ply PARAMS.json   (writes OUT_MESH.ply, preview.ply next to it,
OUT_MESH.ply.json with the stats)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np


def _load_splat_tsdf():
    """export/splat_tsdf.py by its path (this folder on sys.path would shadow nerfstudio)."""
    spec = importlib.util.spec_from_file_location(
        "mf_splat_tsdf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "splat_tsdf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cube_faces():
    """sensor_from_cam rotations of the six faces (OpenCV camera: z forward, y down)."""
    out = []
    for d in np.eye(3).tolist() + (-np.eye(3)).tolist():
        z = np.asarray(d, float)
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(z, up)  # with y = z × x pointing "down" for horizontal faces
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        out.append(np.stack([x, y, z], 1))
    return out


def face_images(pts, grey, R_sc, res):
    """The scan's depth (res, res) float32 and grey (res, res, 3) float32 seen from one face (0 = no return)."""
    q = pts @ R_sc  # camera coordinates
    z = q[:, 2]
    ok = z > 1e-3
    q, z, g = q[ok], z[ok], grey[ok]
    f = res / 2.0
    u = np.floor(f * q[:, 0] / z + f).astype(np.int64)
    v = np.floor(f * q[:, 1] / z + f).astype(np.int64)
    ok = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    u, v, z, g = u[ok], v[ok], z[ok], g[ok]
    order = np.argsort(-z)  # the nearest return written last wins
    dep = np.zeros((res, res), np.float32)
    col = np.zeros((res, res, 3), np.float32)
    dep[v[order], u[order]] = z[order]
    col[v[order], u[order]] = g[order, None]
    return dep, col, int(len(z))


def main():
    scans_path, mesh_path, params = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    t0 = time.time()
    st = _load_splat_tsdf()
    import open3d as o3d
    import open3d.core as o3c
    import torch
    d = np.load(scans_path)
    pts, grey, offs, poses = d["points"], d["grey"], d["offsets"], d["poses"]
    n = len(poses)
    voxel, res, max_range = float(params["voxel"]), int(params["face_res"]), float(params["max_range"])
    min_weight = float(params["min_weight"])
    dev = o3c.Device("CUDA:0")
    # the grid's size: blocks holding returns, x the truncation band (counted on the world points, in chunks)
    block = 8 * voxel
    seen = set()
    for i in range(n):
        w = pts[offs[i]:offs[i + 1]] @ poses[i][:3, :3].T + poses[i][:3, 3]
        seen.update(map(tuple, np.unique(np.floor(w / block).astype(np.int32), axis=0).tolist()))
    count = int(st.BAND_GUESS * len(seen) * 1.5) + 10_000
    free = torch.cuda.mem_get_info()[0]
    if count * st.BLOCK_BYTES > 0.8 * free:
        raise SystemExit(f"[lidar] ~{count:,} voxel blocks at {voxel * 100:.1f} cm need "
                         f"{count * st.BLOCK_BYTES / 1e9:.1f} GB of GPU memory, {free / 1e9:.1f} GB free: "
                         "choose a larger voxel")
    print(f"[lidar] {n} scans, {len(pts):,} returns, voxel {voxel * 100:.1f} cm, {len(seen):,} blocks hold returns "
          f"(grid for {count:,})", flush=True)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=count, device=dev)
    f = res / 2.0
    K = o3c.Tensor(np.array([[f, 0, f], [0, f, f], [0, 0, 1]], np.float64))
    faces = cube_faces()
    used = 0
    for i in range(n):
        p, g = pts[offs[i]:offs[i + 1]], grey[offs[i]:offs[i + 1]]
        for R_sc in faces:
            dep, col, k = face_images(p, g, R_sc, res)
            if not k:
                continue
            T_wc = poses[i].copy()
            T_wc[:3, :3] = poses[i][:3, :3] @ R_sc
            E = o3c.Tensor(np.linalg.inv(T_wc))
            dimg = o3d.t.geometry.Image(o3c.Tensor(dep)).to(dev)
            cimg = o3d.t.geometry.Image(o3c.Tensor(col)).to(dev)
            try:
                bl = vbg.compute_unique_block_coordinates(dimg, K, E, depth_scale=1.0, depth_max=max_range,
                                                          trunc_voxel_multiplier=4.0)
            except RuntimeError as e:  # a few returns may touch no block: Open3D aborts instead of skipping
                if "No block is touched" not in str(e):
                    raise
                continue
            vbg.integrate(bl, dimg, cimg, K, K, E, depth_scale=1.0, depth_max=max_range, trunc_voxel_multiplier=4.0)
            used += k
        print(f"[lidar] scan {i + 1} / {n}", flush=True)
    size = vbg.hashmap().size()
    pruned = st.prune_blocks(vbg, min_weight)
    print(f"[lidar] {size:,} voxel blocks, {pruned:,} without a voxel seen by {min_weight:g} scans; marching cubes",
          flush=True)
    keys, attrs = st.active_blocks(vbg)
    del vbg
    torch.cuda.empty_cache()
    o3c.cuda.release_cache()
    if not len(keys):
        raise SystemExit("[lidar] no surface: no voxel was seen by enough scans")
    v, fc, c = st.marching_cubes(keys, attrs, voxel, min_weight, dev)
    del keys, attrs
    raw = len(fc)
    if raw:
        fc = st.drop_small_pieces(len(v), fc.astype(np.int32), max(0.002 * raw, 50))
    v, fc, c = st.compact(np.asarray(v, np.float32), fc, np.clip(np.round(c * 255), 0, 255).astype(np.uint8))
    nrm = st.vertex_normals(v, fc)
    st.write_ply(mesh_path, v, nrm, c, fc)
    preview = st.preview_mesh(v, fc, c, voxel, 300_000)
    o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(mesh_path), "preview.ply"), preview)
    stats = {"faces": int(len(fc)), "vertices": int(len(v)), "faces_raw": int(raw), "voxel_blocks": int(size),
             "returns_integrated": int(used), "preview_faces": len(preview.triangles),
             "fuse_seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(mesh_path + ".json", "w"), indent=1)
    print(f"[lidar] done: {stats}", flush=True)


if __name__ == "__main__":
    main()

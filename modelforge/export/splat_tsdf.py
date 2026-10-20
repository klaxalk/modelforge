"""Mesh from a Gaussian splat by TSDF fusion of rendered depth. Runs in nerfstudio's venv (torch,
gsplat, open3d); modelforge.splat_mesh prepares the input and calls it.

    python splat_tsdf.py INPUT.npz OUT_DIR PARAMS.json

INPUT.npz (ENU world): means (N,3), quats (N,4, wxyz), scales (N,3, log), opacities (N, logit),
sh (N,K,3), viewmats (V,4,4, world -> OpenCV camera), Ks (V,3,3), width, height.
PARAMS: voxel (m, or null = auto: voxel_px pixels at the median depth), voxel_px, max_depth (m, or
null = auto: depth_factor x median depth), depth_factor, min_alpha, edge_tol, min_weight,
max_preview_faces, min_component.

Per view: render RGB + expected depth + alpha with gsplat at a pinhole camera with the view's
intrinsics (a virtual camera: the real images are never compared, so lens distortion is irrelevant).
A pixel's depth is used where alpha >= min_alpha (opaque enough to be a surface) and its depth differs
from its 4 neighbours by less than edge_tol (relative): expected depth blends foreground and
background across silhouettes, which would fuse into "curtains". The kept depth + colour go into an
Open3D VoxelBlockGrid on the GPU (truncation 4 voxels); marching cubes over voxels seen by at least
min_weight views gives a vertex-coloured mesh. Connected pieces smaller than min_component of the
faces are dropped (floaters).

Known bias: gsplat's expected depth blends the Gaussians' *centre* depths, so on oblique views the
front-most of overlapping opaque Gaussians pulls the depth towards the camera by about the Gaussian
size (synthetic test: 9 cm discs at ~50 deg gave 6.8 cm, 1.1%). Ray-Gaussian intersection depth
(2DGS, RaDe-GS) would remove it; gsplat 1.4 does not render it. (Open3D 0.20's legacy ScalableTSDFVolume returns no surface at all in
this image, even for a synthetic wall, hence the tensor API.)

Prints "[tsdf] view i / n" (progress) and writes OUT_DIR/mesh.ply, preview.ply (<= max_preview_faces,
quadric decimation) and stats.json.
"""

import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
from gsplat import rasterization


def render(g, viewmat, K, W, H):
    colors, alphas, _ = rasterization(
        means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"], colors=g["sh"],
        viewmats=viewmat[None], Ks=K[None], width=W, height=H, sh_degree=g["sh_degree"],
        render_mode="RGB+ED", near_plane=0.01, far_plane=1e10, packed=False)
    rgb = colors[0, ..., :3]
    depth = colors[0, ..., 3]
    alpha = alphas[0, ..., 0]
    rgb = (rgb / alpha.clamp(min=1e-6)[..., None]).clamp(0, 1)  # gsplat's colours are alpha-weighted sums
    return rgb, depth, alpha


def keep_mask(depth, alpha, min_alpha, edge_tol):
    """Opaque pixels whose depth differs from each 4-neighbour by less than edge_tol (relative)."""
    d = depth.clamp(min=1e-6)
    rel = torch.zeros_like(d)
    dx = (d[:, 1:] - d[:, :-1]).abs() / torch.minimum(d[:, 1:], d[:, :-1])
    dy = (d[1:] - d[:-1]).abs() / torch.minimum(d[1:], d[:-1])
    rel[:, 1:], rel[:, :-1] = torch.maximum(rel[:, 1:], dx), torch.maximum(rel[:, :-1], dx)
    rel[1:], rel[:-1] = torch.maximum(rel[1:], dy), torch.maximum(rel[:-1], dy)
    return (alpha >= min_alpha) & (depth > 0) & (rel < edge_tol)


def prune_blocks(vbg, min_weight):
    """Erase the voxel blocks none of whose voxels reached min_weight views: marching cubes skips them anyway,
    but allocates for every block (karlak: 389k blocks did not fit in 11 GB). Returns the number erased."""
    import open3d.core as o3c
    hm = vbg.hashmap()
    active = hm.active_buf_indices().to(o3c.int64)
    if active.shape[0] == 0:
        return 0
    w = torch.utils.dlpack.from_dlpack(vbg.attribute("weight").to_dlpack())  # (capacity, 8, 8, 8, 1)
    idx = torch.utils.dlpack.from_dlpack(active.to_dlpack())
    weak = w[idx].reshape(len(idx), -1).amax(1) < min_weight
    if not bool(weak.any()):
        return 0
    keys = torch.utils.dlpack.from_dlpack(hm.key_tensor().to_dlpack())[idx[weak]].contiguous()
    hm.erase(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys)))
    return int(weak.sum())


def main():
    inp, out_dir, params = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    t0 = time.time()
    d = np.load(inp)
    dev = torch.device("cuda")
    k = d["sh"].shape[1]
    g = {"means": torch.tensor(d["means"], device=dev, dtype=torch.float32),
         "quats": torch.tensor(d["quats"], device=dev, dtype=torch.float32),
         "scales": torch.exp(torch.tensor(d["scales"], device=dev, dtype=torch.float32)),
         "opacities": torch.sigmoid(torch.tensor(d["opacities"], device=dev, dtype=torch.float32)),
         "sh": torch.tensor(d["sh"], device=dev, dtype=torch.float32), "sh_degree": int(round(np.sqrt(k))) - 1}
    viewmats, Ks = d["viewmats"], d["Ks"]
    W, H = int(d["width"]), int(d["height"])
    n = len(viewmats)

    # pass 1 (cheap, ~25 views): the median depth sets the automatic voxel size and depth cut-off
    medians = []
    with torch.no_grad():
        for i in range(0, n, max(1, n // 25)):
            _, depth, alpha = render(g, torch.tensor(viewmats[i], device=dev, dtype=torch.float32),
                                     torch.tensor(Ks[i], device=dev, dtype=torch.float32), W, H)
            v = depth[alpha >= params["min_alpha"]]
            if v.numel():
                medians.append(float(v.median()))
    if not medians:
        raise SystemExit("the splat renders nothing opaque from the cameras")
    median_depth = float(np.median(medians))
    fx = float(np.median(Ks[:, 0, 0]))
    voxel = params["voxel"] or median_depth / fx * params["voxel_px"]
    max_depth = params["max_depth"] or median_depth * params["depth_factor"]
    print(f"[tsdf] median depth {median_depth:.2f} m, voxel {voxel * 100:.1f} cm, max depth {max_depth:.1f} m, "
          f"{n} views at {W}x{H}, {len(d['means']):,} gaussians", flush=True)

    import open3d.core as o3c
    o3dev = o3c.Device("CUDA:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=100_000, device=o3dev)
    used = 0
    with torch.no_grad():
        for i in range(n):
            K = Ks[i]
            rgb, depth, alpha = render(g, torch.tensor(viewmats[i], device=dev, dtype=torch.float32),
                                       torch.tensor(K, device=dev, dtype=torch.float32), W, H)
            m = keep_mask(depth, alpha, params["min_alpha"], params["edge_tol"]) & (depth < max_depth)
            used += int(m.sum())
            dep = torch.where(m, depth, torch.zeros_like(depth)).contiguous().cpu().numpy()
            dimg = o3d.t.geometry.Image(o3c.Tensor(dep)).to(o3dev)
            cimg = o3d.t.geometry.Image(o3c.Tensor(rgb.contiguous().cpu().numpy())).to(o3dev)
            Kt, E = o3c.Tensor(K.astype(np.float64)), o3c.Tensor(viewmats[i].astype(np.float64))
            blocks = vbg.compute_unique_block_coordinates(dimg, Kt, E, depth_scale=1.0, depth_max=max_depth,
                                                          trunc_voxel_multiplier=4.0)
            vbg.integrate(blocks, dimg, cimg, Kt, Kt, E, depth_scale=1.0, depth_max=max_depth,
                          trunc_voxel_multiplier=4.0)
            print(f"[tsdf] view {i + 1} / {n}", flush=True)
    print("[tsdf] marching cubes", flush=True)
    # torch's cached render memory goes back first: on karlak (471 views, 2.1M Gaussians) Open3D's CUDA
    # marching cubes ran out of GPU memory without this (a failed CUDA allocation leaves Open3D's CUDA state
    # broken, so it cannot be retried, and moving a large grid to the CPU crashed Open3D 0.20)
    del g
    torch.cuda.empty_cache()
    n_blocks = vbg.hashmap().size()
    pruned = prune_blocks(vbg, params["min_weight"])
    free, total = torch.cuda.mem_get_info()
    print(f"[tsdf] {n_blocks:,} voxel blocks, {pruned:,} pruned (no voxel seen by {params['min_weight']} views); GPU "
          f"memory free {free / 1e9:.1f} of {total / 1e9:.1f} GB", flush=True)
    try:
        mesh = vbg.extract_triangle_mesh(weight_threshold=params["min_weight"]).to_legacy()
    except RuntimeError as e:  # Open3D's CUDA state is broken now: say why and leave without its teardown
        if "CUDA" not in str(e):
            raise
        print(f"[tsdf] not enough GPU memory for marching cubes over {n_blocks - pruned:,} voxel blocks at "
              f"{voxel * 100:.1f} cm: use a coarser quality (or a larger voxel)", flush=True)
        os._exit(2)
    raw_faces = len(mesh.triangles)
    if raw_faces:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters, counts = np.asarray(clusters), np.asarray(counts)
        small = counts[clusters] < max(params["min_component"] * raw_faces, 50)
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)
    preview = mesh
    if len(mesh.triangles) > params["max_preview_faces"]:
        print("[tsdf] simplifying the preview", flush=True)
        preview = mesh.simplify_quadric_decimation(params["max_preview_faces"])
        preview.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "preview.ply"), preview)
    stats = {"views": n, "width": W, "height": H, "gaussians": int(len(d["means"])),
             "median_depth_m": round(median_depth, 3), "voxel_m": round(voxel, 4), "max_depth_m": round(max_depth, 2),
             "depth_pixels": used, "faces_raw": raw_faces, "faces": len(mesh.triangles),
             "vertices": len(mesh.vertices), "preview_faces": len(preview.triangles),
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"[tsdf] done: {stats}", flush=True)


if __name__ == "__main__":
    main()

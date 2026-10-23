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


def active_blocks(vbg):
    """The active blocks as CPU arrays: keys (n, 3) int32 and every attribute (n, 8, 8, 8, c)."""
    import open3d.core as o3c
    hm = vbg.hashmap()
    idx = torch.utils.dlpack.from_dlpack(hm.active_buf_indices().to(o3c.int64).to_dlpack())
    keys = torch.utils.dlpack.from_dlpack(hm.key_tensor().to_dlpack())[idx].cpu().numpy()
    attrs = {a: torch.utils.dlpack.from_dlpack(vbg.attribute(a).to_dlpack())[idx].cpu().numpy()
             for a in ("tsdf", "weight", "color")}
    return keys, attrs


MC_MAX_BLOCKS = 200_000  # Open3D 0.20's marching cubes crashes from 2^18 = 262,144 blocks on (CUDA and CPU alike)


def marching_cubes(keys, attrs, voxel, min_weight, device, max_blocks=MC_MAX_BLOCKS):
    """Marching cubes over the blocks (keys (n, 3), attrs from active_blocks) in pieces of <= max_blocks: each piece
    is a grid of its blocks + their 26 neighbours, and keeps the triangles of the cubes in its own blocks (a cube's
    triangles lie inside it, so the centroid names its block); the pieces' shared vertices are merged.
    Returns (vertices (V, 3), triangles (F, 3), colours (V, 3)) as numpy arrays."""
    import open3d.core as o3c
    span = int(np.abs(keys).max()) * 2 + 3
    code = lambda k: ((k[:, 0].astype(np.int64) + span // 2) * span + k[:, 1] + span // 2) * span + k[:, 2] + span // 2
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))  # slabs along x
    all_codes = np.sort(code(keys))
    offsets = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)
    n_pieces = max(1, int(np.ceil(len(keys) / max_blocks)))
    verts, tris, cols, n_v = [], [], [], 0
    for i, core in enumerate(np.array_split(order, n_pieces)):
        if n_pieces > 1:
            print(f"[tsdf] marching cubes: piece {i + 1} / {n_pieces}", flush=True)
        core_codes = code(keys[core])
        near = (keys[core][:, None, :].astype(np.int64) + offsets[None]).reshape(-1, 3)
        near = np.unique(code(near))
        pos = np.minimum(np.searchsorted(all_codes, near), len(all_codes) - 1)
        piece = np.flatnonzero(np.isin(code(keys), near[all_codes[pos] == near])) if n_pieces > 1 else core
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=2 * len(piece),
            device=device)  # room to spare: a hash map filled to capacity drops keys on the GPU (stdgpu)
        buf, ok = vbg.hashmap().activate(o3c.Tensor(keys[piece]).to(device))
        if int(ok.cpu().numpy().sum()) != len(piece):
            raise RuntimeError(f"marching cubes: only {int(ok.cpu().numpy().sum())} of {len(piece)} blocks inserted")
        bi = torch.utils.dlpack.from_dlpack(buf.to(o3c.int64).to_dlpack())
        for a in ("tsdf", "weight", "color"):
            t = torch.utils.dlpack.from_dlpack(vbg.attribute(a).to_dlpack())
            t[bi] = torch.from_numpy(attrs[a][piece]).to(t.device)
        del bi, buf
        m = vbg.extract_triangle_mesh(weight_threshold=min_weight).cpu()
        del vbg
        if device.get_type() == o3c.Device.DeviceType.CUDA:
            o3c.cuda.release_cache()
        v, f = m.vertex.positions.numpy(), m.triangle.indices.numpy().astype(np.int64)
        c = m.vertex.colors.numpy() if "colors" in m.vertex else np.zeros_like(v)
        if n_pieces > 1 and len(f):
            owner = np.floor(v[f].mean(1) / voxel / 8).astype(np.int64)
            f = f[np.isin(code(owner), core_codes)]
        verts.append(v), tris.append(f + n_v), cols.append(c)
        n_v += len(v)
    v, f, c = np.concatenate(verts), np.concatenate(tris), np.concatenate(cols)
    if n_pieces > 1:  # a vertex on an edge between pieces exists in both, computed from the same voxels
        v, first, inv = np.unique(v, axis=0, return_index=True, return_inverse=True)
        c, f = c[first], inv.reshape(-1)[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]  # Open3D's own doubled vertices
    return v, f, c


def preview_mesh(mesh, voxel, max_faces, cluster_factor=4):
    """A copy of at most max_faces for the GUI. Above cluster_factor x max_faces, vertex clustering (cheap, colours
    averaged) first brings it near that size, then quadric decimation: Open3D's quadric decimation of the whole
    mesh needs ~1 KB per face (berlin, 24.3M faces: the process was OOM-killed at 27.6 GB)."""
    if len(mesh.triangles) <= max_faces:
        return mesh
    faces = len(mesh.triangles)
    if faces > cluster_factor * max_faces:  # faces scale with 1 / cell^2 on a surface
        cell = voxel * np.sqrt(faces / (cluster_factor * max_faces))
        print(f"[tsdf] simplifying the preview: clustering {faces:,} faces at {cell * 100:.1f} cm", flush=True)
        mesh = mesh.simplify_vertex_clustering(cell, o3d.geometry.SimplificationContraction.Average)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    print(f"[tsdf] simplifying the preview: {len(mesh.triangles):,} faces to {max_faces:,}", flush=True)
    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.compute_vertex_normals()
    return mesh


def main():
    inp, out_dir, params = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    t0 = time.time()
    d = np.load(inp)
    n_gaussians = int(len(d["means"]))
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
    used, skipped = 0, 0
    with torch.no_grad():
        for i in range(n):
            K = Ks[i]
            rgb, depth, alpha = render(g, torch.tensor(viewmats[i], device=dev, dtype=torch.float32),
                                       torch.tensor(K, device=dev, dtype=torch.float32), W, H)
            m = keep_mask(depth, alpha, params["min_alpha"], params["edge_tol"]) & (depth < max_depth)
            used += int(m.sum())
            if not bool(m.any()):  # nothing opaque in range (sky, the scene's edge): Open3D aborts on it
                skipped += 1
                print(f"[tsdf] view {i + 1} / {n}", flush=True)
                continue
            dep = torch.where(m, depth, torch.zeros_like(depth)).contiguous().cpu().numpy()
            dimg = o3d.t.geometry.Image(o3c.Tensor(dep)).to(o3dev)
            cimg = o3d.t.geometry.Image(o3c.Tensor(rgb.contiguous().cpu().numpy())).to(o3dev)
            Kt, E = o3c.Tensor(K.astype(np.float64)), o3c.Tensor(viewmats[i].astype(np.float64))
            try:
                blocks = vbg.compute_unique_block_coordinates(dimg, Kt, E, depth_scale=1.0, depth_max=max_depth,
                                                              trunc_voxel_multiplier=4.0)
            except RuntimeError as e:  # a few stray pixels can touch no block: Open3D aborts instead of skipping
                if "No block is touched" not in str(e):
                    raise
                skipped += 1
                print(f"[tsdf] view {i + 1}: {int(m.sum())} pixels kept but no voxel block touched "
                      f"({vbg.hashmap().size():,} blocks so far): skipped", flush=True)
                print(f"[tsdf] view {i + 1} / {n}", flush=True)
                continue
            vbg.integrate(blocks, dimg, cimg, Kt, Kt, E, depth_scale=1.0, depth_max=max_depth,
                          trunc_voxel_multiplier=4.0)
            print(f"[tsdf] view {i + 1} / {n}", flush=True)
    print(f"[tsdf] marching cubes ({skipped} of {n} views had no surface in range)" if skipped
          else "[tsdf] marching cubes", flush=True)
    # torch's cached render memory goes back first (a failed CUDA allocation leaves Open3D's CUDA state broken,
    # so it cannot be retried); marching cubes then runs in pieces below Open3D's block limit (MC_MAX_BLOCKS)
    del g
    torch.cuda.empty_cache()
    n_blocks = vbg.hashmap().size()
    pruned = prune_blocks(vbg, params["min_weight"])
    torch.cuda.empty_cache()
    o3c.cuda.release_cache()  # Open3D's allocator keeps the integration's freed buffers otherwise
    free, total = torch.cuda.mem_get_info()
    print(f"[tsdf] {n_blocks:,} voxel blocks, {pruned:,} pruned (no voxel seen by {params['min_weight']} views, capacity "
          f"{vbg.hashmap().capacity():,}); GPU memory free {free / 1e9:.1f} of {total / 1e9:.1f} GB", flush=True)
    keys, attrs = active_blocks(vbg)
    del vbg  # the integration grid (capacity grown in doublings) makes room for the pieces
    o3c.cuda.release_cache()
    try:
        v, f, c = marching_cubes(keys, attrs, voxel, params["min_weight"], o3dev,
                                 params.get("mc_max_blocks") or MC_MAX_BLOCKS)
    except RuntimeError as e:  # Open3D's CUDA state is broken now: say why and leave without its teardown
        if "CUDA" not in str(e):
            raise
        print(f"[tsdf] marching cubes failed on the GPU over {len(keys):,} voxel blocks at {voxel * 100:.1f} cm "
              f"({e}): try a coarser quality (or a larger voxel)", flush=True)
        os._exit(2)
    del keys, attrs, d  # the voxel blocks and the splat: the preview's decimation needs the room (berlin: OOM at 28 GB)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                     o3d.utility.Vector3iVector(f.astype(np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1).astype(np.float64))
    del v, f, c
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
    preview = preview_mesh(mesh, voxel, params["max_preview_faces"])
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "preview.ply"), preview)
    stats = {"views": n, "width": W, "height": H, "gaussians": n_gaussians,
             "median_depth_m": round(median_depth, 3), "voxel_m": round(voxel, 4), "max_depth_m": round(max_depth, 2),
             "depth_pixels": used, "faces_raw": raw_faces, "faces": len(mesh.triangles),
             "vertices": len(mesh.vertices), "preview_faces": len(preview.triangles),
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"[tsdf] done: {stats}", flush=True)


if __name__ == "__main__":
    main()

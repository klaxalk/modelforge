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

Large grids go in tiles: when a cheap guess from the Gaussian centres says the grid might not fit in the free GPU
memory (TILE_MEM_SHARE / BLOCK_BYTES; params max_tile_blocks overrides), one pass counts the blocks fusion would
touch exactly (count_blocks) and the scene is cut in x / y into tiles that fit (plan_tiles). Each tile fuses the
pixels whose surface lands in it + TILE_MARGIN blocks, runs marching cubes and keeps its own triangles; seam vertices
are computed from the same voxels in both tiles and merged (the same mesh as one grid). berlin fine (6.6 cm): 3.0M
blocks, 4 tiles, 99M faces in 469 s (one grid ran out of GPU memory). The whole mesh is then handled in numpy
(merge, floaters, normals, PLY), not as an Open3D TriangleMesh, whose cluster_connected_triangles alone needs ~420 B
per face.

Prints "[tsdf] view i / n" (progress) and writes OUT_DIR/mesh.ply, preview.ply (<= ~max_preview_faces: vertex
clustering, then quadric decimation) and stats.json (incl. voxel_blocks, tiles).
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


BLOCK_BYTES = 10_500  # GPU memory per voxel block: 8^3 voxels x (tsdf + weight + rgb) float32 + its hash entry
TILE_MEM_SHARE = 0.6  # of the free GPU memory (after the splat is loaded) for one tile's grid
PREALLOC = 1.3  # a tile's grid is allocated for this x its counted blocks (no growth by doubling)
TILE_MARGIN = 2  # blocks around a tile fused with it: a voxel near its edge sees the same pixels as in the next tile
BAND_GUESS = 2.5  # blocks per block holding Gaussian centres (the truncation band), for the cheap first guess


def count_blocks(n, kept, Ks, viewmats, voxel, max_depth, W, H, device):
    """The voxel blocks fusion would touch, exactly (Open3D's own block coordinates per view into a GPU hash set,
    no voxels allocated), and each view's (lo, hi) block x / y range (None: nothing kept). Renders every view."""
    import open3d.core as o3c
    tmp = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=1000, device=device)
    hs = o3c.HashSet(1_000_000, o3c.int32, (3,), device)
    boxes = []
    with torch.no_grad():
        for i in range(n):
            _, depth, m = kept(i)
            if not bool(m.any()):
                boxes.append(None)
                print(f"[tsdf] counting blocks: view {i + 1} / {n}", flush=True)
                continue
            dep = torch.where(m, depth, torch.zeros_like(depth)).contiguous().cpu().numpy()
            dimg = o3d.t.geometry.Image(o3c.Tensor(dep)).to(device)
            try:
                bl = tmp.compute_unique_block_coordinates(
                    dimg, o3c.Tensor(Ks[i].astype(np.float64)), o3c.Tensor(viewmats[i].astype(np.float64)),
                    depth_scale=1.0, depth_max=max_depth, trunc_voxel_multiplier=4.0)
            except RuntimeError as e:
                if "No block is touched" not in str(e):
                    raise
                boxes.append(None)
                continue
            hs.insert(bl)
            xy = bl[:, :2].cpu().numpy()
            boxes.append((xy.min(0), xy.max(0) + 1))
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"[tsdf] counting blocks: view {i + 1} / {n}", flush=True)
    keys = hs.key_tensor()[hs.active_buf_indices().to(o3c.int64)].cpu().numpy()
    del hs, tmp
    o3c.cuda.release_cache()
    return keys, boxes


def plan_tiles(xy, max_blocks, margin):
    """Cut the blocks' x / y (block units) into boxes (lo, hi), half-open, each holding at most max_blocks blocks
    with its margin: recursively along the longer side at the median. [None] = one grid for everything."""
    if len(xy) <= max_blocks:
        return [None]
    out = []

    def cut(lo, hi):
        inb = np.all((xy >= lo - margin) & (xy < hi + margin), axis=1)
        own = xy[np.all((xy >= lo) & (xy < hi), axis=1)]
        if inb.sum() <= max_blocks or len(own) < 2 or np.all(hi - lo <= 2 * margin + 1):
            out.append((lo, hi))
            return
        ax = int(np.argmax(hi - lo))
        mid = int(np.median(own[:, ax]))
        mid = min(max(mid, lo[ax] + 1), hi[ax] - 1)
        a_hi, b_lo = hi.copy(), lo.copy()
        a_hi[ax], b_lo[ax] = mid, mid
        cut(lo, a_hi)
        cut(b_lo, hi)

    cut(xy.min(0).astype(np.int64), xy.max(0).astype(np.int64) + 1)
    return out


def boxes_meet(view_box, box, margin):
    """Does a view's block range reach the tile's box + margin?"""
    return bool(np.all(view_box[0] < np.asarray(box[1]) + margin) and np.all(view_box[1] > np.asarray(box[0]) - margin))


def in_blocks(depth, viewmat, K, block, lo, hi):
    """Pixels whose surface point (z-depth back-projected) lies in the block x / y range [lo, hi)."""
    Hh, Ww = depth.shape
    v, u = torch.meshgrid(torch.arange(Hh, device=depth.device, dtype=torch.float32) + 0.5,
                          torch.arange(Ww, device=depth.device, dtype=torch.float32) + 0.5, indexing="ij")
    x = (u - float(K[0, 2])) / float(K[0, 0]) * depth
    y = (v - float(K[1, 2])) / float(K[1, 1]) * depth
    R = torch.tensor(viewmat[:3, :3].T, device=depth.device, dtype=torch.float32)  # camera -> world
    c = torch.tensor(-viewmat[:3, :3].T @ viewmat[:3, 3], device=depth.device, dtype=torch.float32)
    wx = R[0, 0] * x + R[0, 1] * y + R[0, 2] * depth + c[0]
    wy = R[1, 0] * x + R[1, 1] * y + R[1, 2] * depth + c[1]
    bx, by = torch.floor(wx / block), torch.floor(wy / block)
    return (bx >= float(lo[0])) & (bx < float(hi[0])) & (by >= float(lo[1])) & (by < float(hi[1]))


def merge_parts(parts, block, voxel):
    """The tiles' meshes (v, f, c, box) as one: only the vertices of a tile's triangles, and a vertex on a seam (within
    a voxel of its tile's x / y edge) merged with its twin from the next tile (both computed from the same voxels, so
    equal). Returns (v float32, f int32, c uint8)."""
    vs, fs, cs, seam, n_v = [], [], [], [], 0
    for v, f, c, box in parts:
        used = np.unique(f)
        remap = np.full(len(v), -1, np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        v, c, f = np.ascontiguousarray(v[used], np.float32), c[used], remap[f]
        if box is not None:
            lo, hi = np.asarray(box[0]) * block, np.asarray(box[1]) * block
            seam.append(np.flatnonzero(np.any((v[:, :2] < lo + voxel) | (v[:, :2] > hi - voxel), axis=1)) + n_v)
        vs.append(v), cs.append(c), fs.append(f + np.int32(n_v))
        n_v += len(v)
    if not vs or n_v >= 2 ** 31:
        assert n_v < 2 ** 31, "more than 2^31 vertices"
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8)
    v, c, f = np.concatenate(vs), np.concatenate(cs), np.concatenate(fs)
    del vs, cs, fs
    seam = np.concatenate(seam) if seam else np.zeros(0, np.int64)
    if len(parts) > 1 and len(seam):
        key = np.ascontiguousarray(v[seam]).view(np.dtype((np.void, 12))).ravel()
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        target = np.arange(len(v), dtype=np.int32)
        target[seam] = seam[first][inv.reshape(-1)]
        f = target[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f, c


def drop_small_pieces(n, f, min_faces):
    """The faces without the connected pieces (sharing vertices) of fewer than min_faces faces (floaters)."""
    parent = np.arange(n, dtype=f.dtype)
    a, b = np.concatenate([f[:, 0], f[:, 1]]), np.concatenate([f[:, 1], f[:, 2]])
    while True:  # hooking to the smaller root, then pointer jumping (as sfm_mesh.union_labels)
        pa, pb = parent[a], parent[b]
        if np.array_equal(pa, pb):
            break
        m = np.minimum(pa, pb)
        np.minimum.at(parent, pa, m)
        np.minimum.at(parent, pb, m)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
    del a, b
    label = parent[f[:, 0]]
    return f[np.bincount(label, minlength=n)[label] >= min_faces]


def compact(v, f, c):
    """Only the vertices the faces use; faces as int32."""
    used = np.unique(f)
    remap = np.full(len(v), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return v[used], remap[f], c[used]


def vertex_normals(v, f):
    """Area-weighted vertex normals (as Open3D's compute_vertex_normals), float32."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    nrm = np.zeros((len(v), 3), np.float32)
    for k in range(3):
        for ax in range(3):
            nrm[:, ax] += np.bincount(f[:, k], weights=fn[:, ax], minlength=len(v)).astype(np.float32)
    del fn
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return nrm


def write_ply(path, v, nrm, c, f, chunk=5_000_000):
    """Binary PLY: x y z, nx ny nz (float), red green blue (uchar); faces as uchar count + int indices. Written in
    chunks (100M faces as one record array would be another 1.3 GB)."""
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}"]
    header += [f"property {'float' if vdt[k].kind == 'f' else 'uchar'} {k}" for k in vdt.names]
    header += [f"element face {len(f)}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        for s in range(0, len(v), chunk):
            rec = np.empty(min(chunk, len(v) - s), vdt)
            for k, src in zip(("x", "y", "z"), v[s:s + chunk].T):
                rec[k] = src
            for k, src in zip(("nx", "ny", "nz"), nrm[s:s + chunk].T):
                rec[k] = src
            for k, src in zip(("red", "green", "blue"), c[s:s + chunk].T):
                rec[k] = src
            fh.write(rec.tobytes())
        for s in range(0, len(f), chunk):
            rec = np.empty(min(chunk, len(f) - s), fdt)
            rec["n"], rec["i"] = 3, f[s:s + chunk]
            fh.write(rec.tobytes())


def cluster_vertices(v, f, c, cell):
    """Vertex clustering in numpy: one vertex per cell (mean position and colour), faces between three different
    cells, each once. Returns (v float64, f int64, c float 0-1)."""
    q = np.floor((v - v.min(0)) / cell).astype(np.int64)
    span = q.max(0) + 1
    _, inv = np.unique((q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2], return_inverse=True)
    inv = inv.reshape(-1)
    del q
    cnt = np.bincount(inv).astype(np.float64)
    pv = np.stack([np.bincount(inv, weights=v[:, a]) for a in range(3)], 1) / cnt[:, None]
    pc = np.stack([np.bincount(inv, weights=c[:, a]) for a in range(3)], 1) / cnt[:, None] / 255.0
    pf = inv[f]
    pf = pf[(pf[:, 0] != pf[:, 1]) & (pf[:, 1] != pf[:, 2]) & (pf[:, 0] != pf[:, 2])]
    _, first = np.unique(np.sort(pf, axis=1), axis=0, return_index=True)
    return pv, pf[np.sort(first)], pc


def preview_mesh(v, f, c, voxel, max_faces, cluster_factor=4):
    """An Open3D mesh of at most max_faces for the GUI. Above cluster_factor x max_faces, vertex clustering (cheap,
    colours averaged) first brings it near that size, then quadric decimation: Open3D's quadric decimation of the
    whole mesh needs ~1 KB per face (berlin, 24.3M faces: the process was OOM-killed at 27.6 GB)."""
    faces = len(f)
    if faces > cluster_factor * max_faces:  # faces scale with 1 / cell^2 on a surface
        cell = voxel * np.sqrt(faces / (cluster_factor * max_faces))
        print(f"[tsdf] simplifying the preview: clustering {faces:,} faces at {cell * 100:.1f} cm", flush=True)
        v, f, c = cluster_vertices(v, f, c, cell)
    else:
        c = c / 255.0
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(v, np.float64)),
                                     o3d.utility.Vector3iVector(np.asarray(f, np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(c, np.float64))
    if len(mesh.triangles) > max_faces:  # can stop short on a clustered mesh (berlin fine: 979k -> 441k for 300k;
        # repeating it stays at ~434k): fine for the View tab
        print(f"[tsdf] simplifying the preview: {len(mesh.triangles):,} faces to {max_faces:,}", flush=True)
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.remove_unreferenced_vertices()
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
    views = lambda i: (torch.tensor(viewmats[i], device=dev, dtype=torch.float32),
                       torch.tensor(Ks[i], device=dev, dtype=torch.float32))

    def kept(i):
        """Render view i: (rgb, depth, keep mask) on the GPU."""
        rgb, depth, alpha = render(g, *views(i), W, H)
        return rgb, depth, keep_mask(depth, alpha, params["min_alpha"], params["edge_tol"]) & (depth < max_depth)

    # tiles: the grid must fit in GPU memory (berlin fine, 6.6 cm: out of memory on 24 GB). A cheap guess from the
    # Gaussian centres decides; when it might not fit, one pass over the views counts the blocks exactly and the
    # scene is cut in x / y into tiles that fit (plan_tiles)
    free = torch.cuda.mem_get_info()[0]
    budget = int(params.get("max_tile_blocks") or TILE_MEM_SHARE * free / BLOCK_BYTES)
    op = 1 / (1 + np.exp(-np.asarray(d["opacities"], np.float64).reshape(-1)))
    guess = BAND_GUESS * len(np.unique(np.floor(np.asarray(d["means"])[op > 0.3] / (8 * voxel)).astype(np.int64), axis=0))
    tiles, view_boxes, blocks = [None], None, None
    if guess > 0.5 * budget:
        print(f"[tsdf] counting voxel blocks (about {guess:,.0f} expected, {budget:,} fit in "
              f"{free / 1e9:.1f} GB)", flush=True)
        blocks, view_boxes = count_blocks(n, kept, Ks, viewmats, voxel, max_depth, W, H, o3dev)
        tiles = plan_tiles(blocks[:, :2], int(budget / PREALLOC), TILE_MARGIN)
        print(f"[tsdf] {len(blocks):,} voxel blocks: {len(tiles)} tile(s) of at most {budget:,}", flush=True)
    tile_views = [list(range(n)) if box is None else
                  [i for i in range(n) if view_boxes[i] is not None and boxes_meet(view_boxes[i], box, TILE_MARGIN)]
                  for box in tiles]
    total, done = sum(len(v) for v in tile_views), 0
    used, skipped, n_blocks, pruned, parts = 0, 0, 0, 0, []
    for k, (box, vis) in enumerate(zip(tiles, tile_views)):
        if box is None:  # one grid: the counted size up front, else grown by doubling from 100k (as before tiles)
            count = int(PREALLOC * len(blocks)) + 1000 if blocks is not None else 100_000
        else:
            lo, hi = np.asarray(box[0]) - TILE_MARGIN, np.asarray(box[1]) + TILE_MARGIN
            inb = np.all((blocks[:, :2] >= lo) & (blocks[:, :2] < hi), axis=1)
            count = int(PREALLOC * inb.sum()) + 1000
            print(f"[tsdf] tile {k + 1} / {len(tiles)}: {int(inb.sum()):,} blocks, {len(vis)} views; GPU memory free "
              f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)
        torch.cuda.empty_cache()
        o3c.cuda.release_cache()
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8, block_count=count, device=o3dev)
        with torch.no_grad():
            for i in vis:
                done += 1
                rgb, depth, m = kept(i)
                if box is not None:  # only the pixels whose surface lies in this tile + margin
                    m &= in_blocks(depth, viewmats[i], Ks[i], 8 * voxel, lo, hi)
                used += int(m.sum())  # with tiles, pixels in a margin count in both tiles
                if not bool(m.any()):  # nothing opaque in range (sky, the scene's edge): Open3D aborts on it
                    skipped += box is None
                    print(f"[tsdf] view {done} / {total}", flush=True)
                    continue
                dep = torch.where(m, depth, torch.zeros_like(depth)).contiguous().cpu().numpy()
                dimg = o3d.t.geometry.Image(o3c.Tensor(dep)).to(o3dev)
                cimg = o3d.t.geometry.Image(o3c.Tensor(rgb.contiguous().cpu().numpy())).to(o3dev)
                Kt, E = o3c.Tensor(Ks[i].astype(np.float64)), o3c.Tensor(viewmats[i].astype(np.float64))
                try:
                    bl = vbg.compute_unique_block_coordinates(dimg, Kt, E, depth_scale=1.0, depth_max=max_depth,
                                                              trunc_voxel_multiplier=4.0)
                except RuntimeError as e:  # a few stray pixels can touch no block: Open3D aborts instead of skipping
                    if "No block is touched" not in str(e):
                        raise
                    skipped += box is None
                    print(f"[tsdf] view {i + 1}: {int(m.sum())} pixels kept but no voxel block touched "
                          f"({vbg.hashmap().size():,} blocks so far): skipped", flush=True)
                    print(f"[tsdf] view {done} / {total}", flush=True)
                    continue
                vbg.integrate(bl, dimg, cimg, Kt, Kt, E, depth_scale=1.0, depth_max=max_depth,
                              trunc_voxel_multiplier=4.0)
                print(f"[tsdf] view {done} / {total}", flush=True)
        print(f"[tsdf] marching cubes ({skipped} of {n} views had no surface in range)" if skipped and k == 0
              else "[tsdf] marching cubes", flush=True)
        # torch's cached render memory goes back first (a failed CUDA allocation leaves Open3D's CUDA state broken,
        # so it cannot be retried); marching cubes then runs in pieces below Open3D's block limit (MC_MAX_BLOCKS)
        torch.cuda.empty_cache()
        size = vbg.hashmap().size()
        n_blocks += size
        pruned += prune_blocks(vbg, params["min_weight"])
        torch.cuda.empty_cache()
        o3c.cuda.release_cache()  # Open3D's allocator keeps the integration's freed buffers otherwise
        free_now, total_mem = torch.cuda.mem_get_info()
        print(f"[tsdf] {size:,} voxel blocks, {pruned:,} pruned so far (no voxel seen by {params['min_weight']} views, "
              f"capacity {vbg.hashmap().capacity():,}); GPU memory free {free_now / 1e9:.1f} of "
              f"{total_mem / 1e9:.1f} GB", flush=True)
        keys, attrs = active_blocks(vbg)
        del vbg  # the integration grid makes room for the pieces
        torch.cuda.empty_cache()  # active_blocks' GPU copies (berlin fine: ~5 GB, the next tile did not fit)
        o3c.cuda.release_cache()
        if not len(keys):
            continue
        try:
            v, f, c = marching_cubes(keys, attrs, voxel, params["min_weight"], o3dev,
                                     params.get("mc_max_blocks") or MC_MAX_BLOCKS)
        except RuntimeError as e:  # Open3D's CUDA state is broken now: say why and leave without its teardown
            if "CUDA" not in str(e):
                raise
            print(f"[tsdf] marching cubes failed on the GPU over {len(keys):,} voxel blocks at {voxel * 100:.1f} cm "
                  f"({e}): try a coarser quality (or a larger voxel)", flush=True)
            os._exit(2)
        del keys, attrs
        if box is not None and len(f):  # this tile's own triangles (a cube's triangles lie in its block)
            own = np.floor(v[f].mean(1)[:, :2] / (8 * voxel))
            f = f[np.all((own >= box[0]) & (own < box[1]), axis=1)]
        # compact per tile: float32 positions, int32 faces, uint8 colours (int64 / float copies of 4 tiles took GBs)
        parts.append((np.asarray(v, np.float32), f.astype(np.int32), np.clip(np.round(c * 255), 0, 255).astype(np.uint8),
                      box))
        del v, f, c
    del g, d  # the splat: the preview's decimation needs the room (berlin: OOM at 28 GB)
    torch.cuda.empty_cache()
    # the whole mesh in numpy, no Open3D TriangleMesh: Open3D's cluster_connected_triangles alone takes ~420 B per
    # face (berlin fine, ~100M faces: OOM-killed at 27 GB)
    import resource
    ram = lambda: f"RAM peak so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB"
    print(f"[tsdf] merging and cleaning the mesh ({ram()})", flush=True)
    v, f, c = merge_parts(parts, 8 * voxel, voxel)
    del parts
    raw_faces = len(f)
    if raw_faces:
        f = drop_small_pieces(len(v), f, max(params["min_component"] * raw_faces, 50))
    v, f, c = compact(v, f, c)
    print(f"[tsdf] {len(f):,} faces after dropping small pieces ({ram()})", flush=True)
    nrm = vertex_normals(v, f)
    os.makedirs(out_dir, exist_ok=True)
    write_ply(os.path.join(out_dir, "mesh.ply"), v, nrm, c, f)
    del nrm
    n_faces, n_verts = len(f), len(v)
    preview = preview_mesh(v, f, c, voxel, params["max_preview_faces"])
    del v, f, c
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "preview.ply"), preview)
    stats = {"views": n, "width": W, "height": H, "gaussians": n_gaussians,
             "median_depth_m": round(median_depth, 3), "voxel_m": round(voxel, 4), "max_depth_m": round(max_depth, 2),
             "depth_pixels": used, "faces_raw": raw_faces, "faces": n_faces,
             "vertices": n_verts, "preview_faces": len(preview.triangles), "voxel_blocks": n_blocks,
             "tiles": len(tiles),
             "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"[tsdf] done: {stats}", flush=True)


if __name__ == "__main__":
    main()

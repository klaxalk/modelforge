"""Direct sunlight per pixel of every training view, from a trained splat (docs/todo_lighting.md, step 3; the SfM
mesh's casters were too rough: on berlin_lighting the mesh maps' shadow pixels were only 0.5 stops darker than their
sunlit ones in the sunny photos, real shadows 1.5-2.5).

Runs in nerfstudio's venv (gsplat). Shadow mapping with the splat as both caster and receiver:
1. the photos grouped by sun direction (within GROUP_DEG); per group a depth map of the splat seen from the sun
   (orthographic camera along the sun direction over the splat's extent, SUN_PX_M per pixel, expected depth);
2. per photo, at the training camera (export/mesh_depth.py's undistorted camera at the training size), the splat's
   expected depth -> the surface point of every pixel; it is in shadow when it lies more than BIAS_M (+ a slope term)
   behind the sun's depth there (the 3 x 3 minimum: no leaks at edges); normals from that depth (smoothed);
   S = lit x max(0, cos(normal, sun)); -1 where the photo's render is transparent (sky) or outside the sun map.
-> OUT/<image name>.npy (float16), as export/sun_vis.py.

python export/sun_vis_splat.py DATASET SPLAT.ply SUN.json OUT --downscale N
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

GROUP_DEG = 0.5
SUN_PX_M = 0.05
MAX_SUN_PX = 8192
BIAS_M = 0.25
CASTER_OPACITY = 0.5
CASTER_MAX_M = 1.0


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"mf_{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                              f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_splat(path, dev):
    """A binary little-endian 3DGS ply -> gsplat tensors (no plyfile in this venv)."""
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode().strip()
            head.append(line)
            if line == "end_header":
                break
        n = next(int(h.split()[-1]) for h in head if h.startswith("element vertex"))
        props = [h.split() for h in head if h.startswith("property")]
        typ = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2"}
        dt = np.dtype([(p[2], typ[p[1]]) for p in props])
        v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    rest = sorted((k for k in v.dtype.names if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None]
    sh = np.concatenate([dc, np.stack([v[k] for k in rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)], 1) if rest else dc
    t = lambda a: torch.tensor(np.ascontiguousarray(a, np.float32), device=dev)  # noqa: E731
    q = t(np.stack([v[f"rot_{i}"] for i in range(4)], 1))
    return {"means": t(np.stack([v["x"], v["y"], v["z"]], 1)), "quats": q / q.norm(dim=1, keepdim=True),
            "scales": torch.exp(t(np.stack([v[f"scale_{i}"] for i in range(3)], 1))),
            "opacities": torch.sigmoid(t(v["opacity"])), "colors": t(sh)}


def sun_vector(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def depth(g, viewmat, K, W, H, ortho=False):
    """(expected depth, alpha) of the splat."""
    out, alpha, _ = rasterization(means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                                  colors=g["colors"], viewmats=viewmat[None], Ks=K[None], width=W, height=H,
                                  sh_degree=int(round(math.sqrt(g["colors"].shape[1]))) - 1, render_mode="ED",
                                  near_plane=-1e4 if ortho else 0.01, far_plane=1e10, packed=False,
                                  camera_model="ortho" if ortho else "pinhole")
    return out[0, ..., 0], alpha[0, ..., 0]


def sun_view(g, s, dev):
    """Orthographic camera looking along -s (from the sun) over the splat: (viewmat, K, W, H)."""
    z = -torch.tensor(s, dtype=torch.float32, device=dev)  # camera z: from the sun into the scene
    up = torch.tensor([0.0, 0.0, 1.0], device=dev) if abs(s[2]) < 0.99 else torch.tensor([0.0, 1.0, 0.0], device=dev)
    x = torch.linalg.cross(up, z)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])  # world -> camera rows
    m = g["means"]
    lo, hi = torch.quantile(m[::max(1, len(m) // 200000)], torch.tensor([0.01, 0.99], device=dev), dim=0)
    c = (lo + hi) / 2
    pc = (m[::max(1, len(m) // 200000)] - c) @ R.T
    ext = torch.quantile(pc[:, :2].abs(), 0.995, dim=0) + 2.0
    px = max(SUN_PX_M, float(2 * ext.max()) / MAX_SUN_PX)
    W, H = int(2 * float(ext[0]) / px), int(2 * float(ext[1]) / px)
    t = -R @ c + torch.tensor([0.0, 0.0, 1000.0], device=dev)  # the scene 1 km in front (ortho: depth = distance)
    V = torch.eye(4, device=dev)
    V[:3, :3], V[:3, 3] = R, t
    K = torch.tensor([[1 / px, 0, W / 2], [0, 1 / px, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
    return V, K, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("splat")
    ap.add_argument("sun")
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda")
    md = _load("mesh_depth")
    t = json.load(open(os.path.join(a.dataset, "transforms.json")))
    suns = json.load(open(a.sun))
    cams, index = md.frame_camera.cameras(t)
    und = [md.undistorted_camera(c, a.downscale, t) for c in cams]
    g = read_splat(a.splat, dev)
    # the casters: fairly opaque, not huge (floaters, haze and sky blobs would shade everything below them)
    keep = (g["opacities"] > CASTER_OPACITY) & (g["scales"].max(dim=1).values < CASTER_MAX_M)
    caster = {k: v[keep] for k, v in g.items()}
    print(f"sun visibility (splat): {int(keep.sum()):,} of {len(keep):,} gaussians cast shadows", flush=True)
    os.makedirs(a.out, exist_ok=True)
    frames = [(f, i) for f, i in zip(t["frames"], index) if os.path.basename(f["file_path"]) in suns]
    groups = {}
    for f, i in frames:
        sv = suns[os.path.basename(f["file_path"])]
        key = (round(sv["azimuth"] / GROUP_DEG), round(sv["elevation"] / GROUP_DEG))
        groups.setdefault(key, []).append((f, i))
    print(f"sun visibility (splat): {len(frames)} photos, {len(groups)} sun directions, {len(g['means']):,} gaussians",
          flush=True)
    t0, done, lit_share = time.time(), 0, []
    for key, items in groups.items():
        s = sun_vector(key[0] * GROUP_DEG, key[1] * GROUP_DEG)
        if s[2] <= 0:
            continue
        Vs, Ks, Ws, Hs = sun_view(g, s, dev)
        with torch.no_grad():
            Ds, As = depth(caster, Vs, Ks, Ws, Hs, ortho=True)  # "ED" is already normalized by the opacity
            Ds = torch.where(As > 0.5, Ds, torch.full_like(Ds, 1e9))
            Dmin = -F.max_pool2d(-Ds[None, None], 3, 1, 1)[0, 0]  # the nearest of 3 x 3: no light leaks at edges
            sv = torch.tensor(s, dtype=torch.float32, device=dev)
            for f, ci in items:
                K, w, h = und[ci]
                Kt = torch.tensor(K, dtype=torch.float32, device=dev)
                c2w = torch.tensor(np.array(f["transform_matrix"], float), dtype=torch.float32, device=dev)
                c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
                Vc = torch.linalg.inv(c2w)
                D, A = depth(g, Vc, Kt, w, h)
                v, u = torch.meshgrid(torch.arange(h, device=dev) + 0.5, torch.arange(w, device=dev) + 0.5,
                                      indexing="ij")
                rays = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
                P = (rays * D[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]  # world points (z-depth along rays)
                # normals from the depth: cross of the neighbours' differences, smoothed, towards the camera
                dx = F.pad(P[:, 2:] - P[:, :-2], (0, 0, 1, 1), mode="replicate")
                dy = F.pad((P[2:] - P[:-2]).permute(2, 0, 1)[None], (0, 0, 1, 1), mode="replicate")[0].permute(1, 2, 0)
                n = torch.linalg.cross(dx, dy)
                n = F.avg_pool2d(n.permute(2, 0, 1)[None], 9, 1, 4, count_include_pad=False)[0].permute(1, 2, 0)
                n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                n = torch.where(((n * (c2w[:3, 3] - P)).sum(-1, keepdim=True) < 0), -n, n)
                cos = (n @ sv).clamp(min=0)
                ps = P @ Vs[:3, :3].T + Vs[:3, 3]  # into the sun view
                us, vs = ps[..., 0] * Ks[0, 0] + Ks[0, 2], ps[..., 1] * Ks[1, 1] + Ks[1, 2]
                inside = (us >= 0) & (us < Ws - 1) & (vs >= 0) & (vs < Hs - 1)
                grid = torch.stack([us / (Ws - 1) * 2 - 1, vs / (Hs - 1) * 2 - 1], -1)[None]
                dsun = F.grid_sample(Dmin[None, None], grid, mode="nearest", align_corners=True)[0, 0]
                bias = BIAS_M + 0.02 * D  # metres: depth noise grows with distance
                lit = (ps[..., 2] <= dsun + bias).float()
                S = lit * cos
                S = torch.where((A > 0.5) & inside, S, torch.full_like(S, -1.0))
                name = os.path.basename(f["file_path"])
                arr = S.cpu().numpy().astype(np.float16)
                np.save(os.path.join(a.out, name + ".part.npy"), arr)
                os.replace(os.path.join(a.out, name + ".part.npy"), os.path.join(a.out, name + ".npy"))
                k = (A > 0.5) & inside
                lit_share.append(float(lit[k].mean()) if k.any() else 0.0)
                done += 1
                if done % 50 == 0 or done == len(frames):
                    print(f"sun visibility (splat): {done} / {len(frames)} ({(time.time() - t0) / done:.2f} s per map; "
                          f"sunlit share median {100 * np.median(lit_share):.0f} %; sun map {Ws} x {Hs})", flush=True)


if __name__ == "__main__":
    main()

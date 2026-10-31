"""Rate objects of a splat by how their render compares with the photos that see them best (modelforge/replace.py;
runs in nerfstudio's venv: gsplat).

Per object: up to PHOTOS photos with it near their centre (not inside it), each rendered at half size with the photo's
camera (pinhole; the photo undistorted to match), both cropped to the object's projected box (its footprint box from the
ground to its top). sharp = Laplacian energy of the render / of the photo, after a 1 px blur against pixel noise (1 = as
sharp; a smeared object << 1). psnr after a per-channel gain fitted to the photo. Medians over the photos.

python replace_rate.py SPEC.json   SPEC: npz (load_splat arrays), transforms, root (the project), objects ([{id, x, y,
ground_z, height, box}]), out (JSON: {id: {sharp, psnr, photos}}), crops (folder for render | photo crops, or null)"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from gsplat import rasterization

PHOTOS = 3
CENTRE = 0.3  # the object within this share of the half-width / -height of the frame from the centre


def main(spec_path):
    spec = json.load(open(spec_path))
    t = json.load(open(spec["transforms"]))
    frames = t["frames"]
    dev = torch.device("cuda")
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    d = np.load(spec["npz"])
    sh = T(d["sh"])
    G = (T(d["means"]), torch.nn.functional.normalize(T(d["quats"]), dim=-1), torch.exp(T(d["scales"])),
         torch.sigmoid(T(d["opacities"]).reshape(-1)), sh)
    deg = int(round(np.sqrt(sh.shape[1]))) - 1

    def cam(f, half=2):
        g = lambda k: float(f.get(k, t.get(k, 0.0)))  # noqa: E731
        W, H = int(g("w")) // half, int(g("h")) // half
        K = np.array([[g("fl_x") / half, 0, g("cx") / half], [0, g("fl_y") / half, g("cy") / half], [0, 0, 1]])
        dist = np.array([g("k1"), g("k2"), g("p1"), g("p2"), g("k3")])
        c2w = np.array(f["transform_matrix"], float)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV axes
        return W, H, K, dist, c2w

    def lap(a):
        g = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    C = np.array([np.array(f["transform_matrix"], float)[:3, 3] for f in frames])
    R = np.array([np.array(f["transform_matrix"], float)[:3, :3] @ np.diag([1, -1, -1]) for f in frames])
    W0, H0, K0, _, _ = cam(frames[0], 1)
    if spec.get("crops"):
        os.makedirs(spec["crops"], exist_ok=True)
    out = {}
    for k, o in enumerate(spec["objects"]):
        at = np.array([o["x"], o["y"], o["ground_z"] + 0.5 * o["height"]])
        rel = np.einsum("nji,nj->ni", R, at - C)
        u = K0[0, 0] * rel[:, 0] / np.maximum(rel[:, 2], 1e-6)
        v = K0[1, 1] * rel[:, 1] / np.maximum(rel[:, 2], 1e-6)
        ok = (rel[:, 2] > 1.5 * o["height"]) & (np.abs(u) < CENTRE * W0) & (np.abs(v) < CENTRE * H0)
        dist3 = np.linalg.norm(C - at, axis=1)
        pick = [int(j) for j in np.argsort(np.where(ok, dist3, np.inf))[:PHOTOS] if ok[j]]
        x0, y0, x1, y1 = o["box"]
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                            for z in (o["ground_z"], o["ground_z"] + o["height"])])
        sharp, psnr = [], []
        for j in pick:
            W, H, K, dist, c2w = cam(frames[j])
            q = (corners - c2w[:3, 3]) @ c2w[:3, :3]
            if (q[:, 2] <= 0.1).any():
                continue
            pu, pv = K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]
            a0, a1 = int(max(pu.min(), 0)), int(min(pu.max(), W))
            b0, b1 = int(max(pv.min(), 0)), int(min(pv.max(), H))
            if a1 - a0 < 24 or b1 - b0 < 24:
                continue
            with torch.no_grad():
                img, _, _ = rasterization(*G, T(np.linalg.inv(c2w))[None], T(K)[None], W, H, sh_degree=deg)
            ren = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            ph = cv2.cvtColor(cv2.imread(os.path.join(spec["root"], frames[j]["file_path"])), cv2.COLOR_BGR2RGB)
            ph = cv2.undistort(cv2.resize(ph, (W, H), interpolation=cv2.INTER_AREA), K, dist)
            r, p = ren[b0:b1, a0:a1], ph[b0:b1, a0:a1]
            sharp.append(lap(r) / max(lap(p), 1e-6))
            rf, pf = r.astype(np.float64) / 255, p.astype(np.float64) / 255
            gain = (rf * pf).sum((0, 1)) / np.maximum((rf * rf).sum((0, 1)), 1e-9)
            psnr.append(float(10 * np.log10(1 / max(((np.clip(rf * gain, 0, 1) - pf) ** 2).mean(), 1e-10))))
            if spec.get("crops") and len(sharp) == 1:
                cv2.imwrite(os.path.join(spec["crops"], f"{o['id']:04d}.jpg"),
                            cv2.cvtColor(np.concatenate([r, p], 1), cv2.COLOR_RGB2BGR))
        out[str(o["id"])] = {"sharp": round(float(np.median(sharp)), 3) if sharp else None,
                             "psnr": round(float(np.median(psnr)), 2) if psnr else None, "photos": len(sharp)}
        print(f"[rate] object {k + 1} / {len(spec['objects'])}", flush=True)
    json.dump(out, open(spec["out"], "w"))


def detail_map(spec_path):
    """A top-down map of how much of the photos' detail the splat keeps (python replace_rate.py --map SPEC.json).
    SPEC: npz, transforms, root, photos (how many, spread over the flight), lo (x, y), cell, shape (H, W), out (.npz:
    ratio (H, W) float32, NaN where nothing was seen; weight). Per photo: render (RGB + expected depth) and photo
    (undistorted) at half size; local detail = the mean squared Laplacian (after a 1 px blur) over WIN px windows; the
    ratio render / photo of each window centre (every STEP px) goes to the cell its rendered surface point lies in,
    weighted by the photo's detail there (flat sky or road counts little)."""
    WIN, STEP = 31, 12
    spec = json.load(open(spec_path))
    t = json.load(open(spec["transforms"]))
    frames = t["frames"]
    dev = torch.device("cuda")
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    d = np.load(spec["npz"])
    sh = T(d["sh"])
    G = (T(d["means"]), torch.nn.functional.normalize(T(d["quats"]), dim=-1), torch.exp(T(d["scales"])),
         torch.sigmoid(T(d["opacities"]).reshape(-1)), sh)
    deg = int(round(np.sqrt(sh.shape[1]))) - 1
    H, W = spec["shape"]
    lo, cell = np.array(spec["lo"], float), float(spec["cell"])
    num, den = np.zeros(H * W), np.zeros(H * W)
    pick = np.linspace(0, len(frames) - 1, min(int(spec["photos"]), len(frames))).astype(int)

    def energy(a):
        g = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
        return cv2.boxFilter(cv2.Laplacian(g, cv2.CV_32F) ** 2, -1, (WIN, WIN))
    for n, j in enumerate(pick):
        f = frames[j]
        g_ = lambda k: float(f.get(k, t.get(k, 0.0)))  # noqa: E731
        w, h = int(g_("w")) // 2, int(g_("h")) // 2
        K = np.array([[g_("fl_x") / 2, 0, g_("cx") / 2], [0, g_("fl_y") / 2, g_("cy") / 2], [0, 0, 1]])
        dist = np.array([g_("k1"), g_("k2"), g_("p1"), g_("p2"), g_("k3")])
        c2w = np.array(f["transform_matrix"], float)
        c2w[:3, 1:3] *= -1
        with torch.no_grad():
            img, alpha, _ = rasterization(*G, T(np.linalg.inv(c2w))[None], T(K)[None], w, h, sh_degree=deg,
                                          render_mode="RGB+ED")
        ren = (img[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = img[0, ..., 3].cpu().numpy()
        a = alpha[0, ..., 0].cpu().numpy()
        ph = cv2.cvtColor(cv2.imread(os.path.join(spec["root"], f["file_path"])), cv2.COLOR_BGR2RGB)
        ph = cv2.undistort(cv2.resize(ph, (w, h), interpolation=cv2.INTER_AREA), K, dist)
        er, ep = energy(ren), energy(ph)
        vv, uu = np.mgrid[WIN // 2:h - WIN // 2:STEP, WIN // 2:w - WIN // 2:STEP]
        z, wt = depth[vv, uu], ep[vv, uu]
        ok = (a[vv, uu] > 0.9) & (z > 0) & (wt > 1e-3)
        r = np.clip(er[vv, uu] / np.maximum(ep[vv, uu], 1e-6), 0, 2)
        x = (uu - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        P = np.stack([x, y, z], -1)[ok] @ c2w[:3, :3].T + c2w[:3, 3]
        ci = ((P[:, 1] - lo[1]) // cell).astype(int)
        cj = ((P[:, 0] - lo[0]) // cell).astype(int)
        inb = (ci >= 0) & (ci < H) & (cj >= 0) & (cj < W)
        c = ci[inb] * W + cj[inb]
        np.add.at(num, c, (r[ok] * np.sqrt(wt[ok]))[inb])
        np.add.at(den, c, np.sqrt(wt[ok])[inb])
        print(f"[rate] photo {n + 1} / {len(pick)}", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(den > 0, num / den, np.nan).reshape(H, W).astype(np.float32)
    np.savez(spec["out"], ratio=ratio, weight=den.reshape(H, W).astype(np.float32))


if __name__ == "__main__":
    if sys.argv[1] == "--map":
        detail_map(sys.argv[2])
    else:
        main(sys.argv[1])

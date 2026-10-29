"""Render a splat straight down with an orthographic camera (an orthophoto). Runs in nerfstudio's venv (torch,
gsplat, opencv); modelforge.ortho prepares the input and calls it.

    python ortho_render.py INPUT.npz OUT.png

INPUT.npz (ENU world): means, quats (wxyz), scales (log), opacities (logit), sh0 (N, 1, 3) as written by
splat_mesh.load_splat (band 0 only: from straight above the view direction hardly changes), bounds (x0, y0, x1, y1),
mpp (metres per pixel), z_top (above every Gaussian kept). Rendered in pieces of at most PIECE px; OUT.png is RGBA
(alpha = the splat's coverage, colour un-premultiplied), north up.
"""

import sys

import cv2
import numpy as np
import torch
from gsplat import rasterization

PIECE = 2048


def main():
    inp, out = sys.argv[1], sys.argv[2]
    d = np.load(inp)
    dev = torch.device("cuda")
    means = torch.tensor(d["means"], device=dev, dtype=torch.float32)
    quats = torch.tensor(d["quats"], device=dev, dtype=torch.float32)
    scales = torch.exp(torch.tensor(d["scales"], device=dev, dtype=torch.float32))
    opac = torch.sigmoid(torch.tensor(d["opacities"], device=dev, dtype=torch.float32))
    sh0 = torch.tensor(d["sh0"], device=dev, dtype=torch.float32)
    x0, y0, x1, y1 = (float(v) for v in d["bounds"])
    mpp, z_top = float(d["mpp"]), float(d["z_top"])
    W, H = int(round((x1 - x0) / mpp)), int(round((y1 - y0) / mpp))
    img = np.zeros((H, W, 4), np.uint8)
    K = torch.tensor([[1 / mpp, 0, 0], [0, 1 / mpp, 0], [0, 0, 1]], device=dev, dtype=torch.float32)[None]
    with torch.no_grad():
        for r in range(0, H, PIECE):
            for c in range(0, W, PIECE):
                w, h = min(PIECE, W - c), min(PIECE, H - r)
                left, top = x0 + c * mpp, y1 - r * mpp
                # camera axes: x east, y south (image rows), z down; u = (x - left) / mpp, v = (top - y) / mpp
                view = torch.tensor([[1, 0, 0, -left], [0, -1, 0, top], [0, 0, -1, z_top], [0, 0, 0, 1]],
                                    device=dev, dtype=torch.float32)[None]
                rgb, alpha, _ = rasterization(means, quats, scales, opac, sh0, view, K, w, h, sh_degree=0,
                                              near_plane=0.01, far_plane=1e7, camera_model="ortho")
                a = alpha[0, ..., 0].clamp(0, 1)
                col = (rgb[0] / a.clamp(min=1e-4)[..., None]).clamp(0, 1)
                piece = torch.cat([col, a[..., None]], -1)
                img[r:r + h, c:c + w] = (piece * 255 + 0.5).byte().cpu().numpy()
                print(f"[ortho] rows {r + h} / {H}", flush=True)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    print(f"[ortho] {W} x {H} px, {mpp * 100:.1f} cm per px", flush=True)


if __name__ == "__main__":
    main()

"""Render a rig's object (its Gaussians, colours as given) from a few cameras on a grey background, for the vision
model (modelforge/rig_vlm.py). Runs in nerfstudio's venv (torch, gsplat, opencv).

    python rig_views.py INPUT.npz OUTDIR

INPUT.npz: means, quats (wxyz), scales (log), opacities (logit), rgbs (V, N, 3) in [0, 1] (the colours per view);
viewmats (V, 4, 4) world -> OpenCV camera; Ks (V, 3, 3); width, height. Writes OUTDIR/view_<k>.png and
view_<k>_depth.npy (float16 expected depth along the optical axis, 0 where the object covers < half the pixel).
"""

import os
import sys

import cv2
import numpy as np
import torch
from gsplat import rasterization

BACKGROUND = 0.55


def main():
    inp, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = np.load(inp)
    dev = torch.device("cuda")
    t = lambda k: torch.tensor(d[k], device=dev, dtype=torch.float32)  # noqa: E731
    means, quats, rgbs = t("means"), t("quats"), t("rgbs")
    scales, opac = torch.exp(t("scales")), torch.sigmoid(t("opacities"))
    W, H = int(d["width"]), int(d["height"])
    for k in range(len(d["viewmats"])):
        with torch.no_grad():
            col, alpha, _ = rasterization(means, quats / quats.norm(dim=1, keepdim=True), scales, opac, rgbs[k],
                                          t("viewmats")[k:k + 1], t("Ks")[k:k + 1], W, H, render_mode="RGB+ED")
        depth = torch.where(alpha[0, ..., 0] > 0.5, col[0, ..., 3], torch.zeros_like(col[0, ..., 3]))
        np.save(os.path.join(out, f"view_{k}_depth.npy"), depth.cpu().numpy().astype(np.float16))
        img = col[0, ..., :3] + (1 - alpha[0]) * BACKGROUND
        cv2.imwrite(os.path.join(out, f"view_{k}.png"),
                    (img.clamp(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
    print(f"[rig_views] {len(d['viewmats'])} views", flush=True)


if __name__ == "__main__":
    main()

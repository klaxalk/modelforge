"""Render a splat at held-out camera poses and score it against the photos. Runs in nerfstudio's venv
(torch, gsplat, torchmetrics, opencv); modelforge.evaluate prepares the input and calls it.

    python splat_eval.py INPUT.npz OUT.json

INPUT.npz (ENU world): means, quats (wxyz), scales (log), opacities (logit), sh (N,K,3) as written by
splat_mesh.load_splat; viewmats (V,4,4 world -> OpenCV camera), K (3x3) and dist (k1 k2 p1 p2) of the
model camera at width x height, render_width, paths (the photos).

Each photo is undistorted to the pinhole K (OpenCV), both are resized to render_width, and the splat
is rendered there on a black background. Scores per image and averaged: PSNR, SSIM, LPIPS (AlexNet,
torchmetrics; skipped if it cannot load). Fully transparent pixels of the render (alpha < 0.05: no
Gaussians there, e.g. sky the SfM never reconstructed) are reported as `empty` and left in the scores.
The first three views are also written as OUT_view<i>.jpg (photo | render) for a visual check.
"""

import json
import sys
import time

import cv2
import numpy as np
import torch
from gsplat import rasterization
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def main():
    inp, out = sys.argv[1], sys.argv[2]
    d = np.load(inp)
    dev = torch.device("cuda")
    k = d["sh"].shape[1]
    g = {"means": torch.tensor(d["means"], device=dev, dtype=torch.float32),
         "quats": torch.tensor(d["quats"], device=dev, dtype=torch.float32),
         "scales": torch.exp(torch.tensor(d["scales"], device=dev, dtype=torch.float32)),
         "opacities": torch.sigmoid(torch.tensor(d["opacities"], device=dev, dtype=torch.float32)),
         "colors": torch.tensor(d["sh"], device=dev, dtype=torch.float32)}
    W0, H0 = int(d["width"]), int(d["height"])
    s = min(1.0, int(d["render_width"]) / W0)
    W, H = int(round(W0 * s)), int(round(H0 * s))
    K0 = d["K"].astype(np.float64)
    K = K0.copy()
    K[:2] *= s
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    except Exception as e:  # noqa: BLE001 - LPIPS is optional
        print(f"[eval] LPIPS unavailable: {e}", flush=True)
        lpips = None
    maps = cv2.initUndistortRectifyMap(K0, d["dist"], None, K0, (W0, H0), cv2.CV_32FC1)
    per, t0 = [], time.time()
    with torch.no_grad():
        for i, path in enumerate(d["paths"]):
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            gt = torch.tensor(img, device=dev, dtype=torch.float32) / 255.0
            colors, alphas, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"], colors=g["colors"],
                viewmats=torch.tensor(d["viewmats"][i], device=dev)[None],
                Ks=torch.tensor(K, device=dev, dtype=torch.float32)[None], width=W, height=H,
                sh_degree=int(round(np.sqrt(k))) - 1, near_plane=0.01, far_plane=1e10, packed=False)
            pred = colors[0, ..., :3].clamp(0, 1)  # black background: the alpha-weighted sum as is
            a = gt.permute(2, 0, 1)[None]
            b = pred.permute(2, 0, 1)[None]
            m = {"psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
                 "ssim": float(structural_similarity_index_measure(b, a, data_range=1.0)),
                 "empty": float((alphas[0, ..., 0] < 0.05).float().mean())}
            if lpips is not None:
                m["lpips"] = float(lpips(b, a))
            per.append({k: round(v, 4) for k, v in m.items()})
            if i < 3:  # side by side (photo | render) for a visual check of the poses
                side = np.concatenate([img, (pred.cpu().numpy() * 255).astype(np.uint8)], 1)
                cv2.imwrite(out.rsplit(".", 1)[0] + f"_view{i}.jpg", cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
            print(f"[eval] view {i + 1} / {len(d['paths'])}: PSNR {m['psnr']:.2f}", flush=True)
    keys = per[0].keys() if per else ()
    res = {k: round(float(np.mean([p[k] for p in per])), 4) for k in keys}
    res.update({"psnr_p10": round(float(np.percentile([p["psnr"] for p in per], 10)), 3) if per else None,
                "views": len(per), "render": [W, H], "per_image": per, "eval_seconds": round(time.time() - t0, 1)})
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

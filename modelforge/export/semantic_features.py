"""Semantic features per Gaussian from C-RADIOv3 (option A: back-projection, no training). Runs in
nerfstudio's venv (torch, gsplat, opencv) with the RADIO packages on PYTHONPATH; modelforge.semantic
prepares the input and calls it.

    python semantic_features.py INPUT.npz OUT_DIR PARAMS.json

INPUT.npz (ENU world): the splat (means, quats wxyz, log-scales, opacity logits, sh) as splat_mesh.load_splat
writes it; viewmats (V,4,4 world -> OpenCV camera), K (3x3), dist (k1 k2 p1 p2) at width x height, paths.
PARAMS: version (e.g. c-radio_v3-l), adaptor (siglip2), width (RADIO input width, multiple of 16, <= 2048),
dims (PCA dimensions kept), pca_images, depth_div (depth render = input / depth_div), depth_tol.

1. RADIO's SigLIP2 adaptor features (one vector per 16 px patch) of every photo, undistorted to the
   pinhole K and resized to `width`, mapped into the text-aligned space by SigLIP2's attention-pooling
   head applied per patch without its attention (value projection, output projection, MLP residual: the
   MaskCLIP trick). The raw patch features mimic SigLIP2's patch tokens, which are not text-aligned (on
   kladno_dopravnik "a machine" was low on the machine); through the head, "a machine", "gravel" and
   "dirt road" light up where they belong. The head comes from the SigLIP2 checkpoint the adaptor uses.
2. PCA on a sample of patches from `pca_images` photos: mean + `dims` directions (stored to map back).
3. Per photo: render the splat's depth, project every Gaussian centre, keep those in front, inside the
   frame and within depth_tol of the rendered surface (occlusion), sample the reduced feature map there
   (bilinear) and add it, weighted by opacity, to the Gaussian's running sum. Streams: no feature maps
   are stored.

OUT_DIR/features.npz: feats (N, dims) float16 = weighted mean reduced feature per Gaussian (0 if never seen),
weight (N,) float32 (sum of weights; 0 = never seen), pca_mean (D,), pca_basis (D, dims), explained.
Prints "[semantic] pca i / n", "[semantic] image i / n".
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


HEAD_REPO = {"siglip2": "google/siglip2-so400m-patch16-naflex"}  # the SigLIP2 model behind RADIO's adaptor


def load_model(version, adaptor):
    model = torch.hub.load("NVlabs/RADIO", "radio_model", version=version, adaptor_names=[adaptor], progress=False,
                           skip_validation=True, trust_repo=True)
    return model.cuda().eval()


class DenseHead:
    """SigLIP2's attention-pooling head per patch, without the attention: x -> out_proj(v_proj(x)) = h,
    h + MLP(LayerNorm(h)). Puts patch features into the space the text embeddings live in."""

    def __init__(self, adaptor):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        path = hf_hub_download(HEAD_REPO[adaptor], "model.safetensors")
        with safe_open(path, "pt") as f:
            self.w = {k.split("vision_model.head.")[1]: f.get_tensor(k).float().cuda()
                      for k in f.keys() if k.startswith("vision_model.head.")}
        self.d = self.w["layernorm.weight"].shape[0]

    def __call__(self, x):
        w, d = self.w, self.d
        h = (x @ w["attention.in_proj_weight"][2 * d:].T + w["attention.in_proj_bias"][2 * d:])
        h = h @ w["attention.out_proj.weight"].T + w["attention.out_proj.bias"]
        y = F.layer_norm(h, (d,), w["layernorm.weight"], w["layernorm.bias"])
        y = F.gelu(y @ w["mlp.fc1.weight"].T + w["mlp.fc1.bias"], approximate="tanh")
        return h + y @ w["mlp.fc2.weight"].T + w["mlp.fc2.bias"]


def photo(path, maps, size):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def features(model, adaptor, img, head):
    """(h, w, D) text-aligned features of an RGB uint8 image (size a multiple of the patch)."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)[adaptor].features
    h, w = img.shape[0] // model.patch_size, img.shape[1] // model.patch_size
    with torch.no_grad():
        return head(out[0].float()).reshape(h, w, -1)


def main():
    inp, out_dir, params = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    t0 = time.time()
    d = np.load(inp)
    dev = torch.device("cuda")
    k = d["sh"].shape[1]
    means = torch.tensor(d["means"], device=dev, dtype=torch.float32)
    opac = torch.sigmoid(torch.tensor(d["opacities"], device=dev, dtype=torch.float32))
    g = {"means": means, "quats": torch.tensor(d["quats"], device=dev, dtype=torch.float32),
         "scales": torch.exp(torch.tensor(d["scales"], device=dev, dtype=torch.float32)), "opacities": opac,
         "colors": torch.tensor(d["sh"], device=dev, dtype=torch.float32)}
    sh_degree = int(round(np.sqrt(k))) - 1
    W0, H0 = int(d["width"]), int(d["height"])
    model = load_model(params["version"], params["adaptor"])
    head = DenseHead(params["adaptor"])
    ps = model.patch_size
    W = min(int(params["width"]), 2048) // ps * ps
    H = int(round(H0 * W / W0 / ps)) * ps
    K0 = d["K"].astype(np.float64)
    maps = cv2.initUndistortRectifyMap(K0, d["dist"], None, K0, (W0, H0), cv2.CV_32FC1)
    sx, sy = W / W0, H / H0
    viewmats, paths = d["viewmats"], d["paths"]
    n = len(paths)
    print(f"[semantic] {params['version']} at {W}x{H} ({W // ps}x{H // ps} patches), {n} photos, "
          f"{len(means):,} gaussians", flush=True)

    # 2. PCA basis from a sample of patches
    pick = np.unique(np.linspace(0, n - 1, min(n, int(params["pca_images"]))).round().astype(int))
    sample, rng = [], np.random.default_rng(0)
    for j, i in enumerate(pick):
        f = features(model, params["adaptor"], photo(paths[i], maps, (W, H)), head)
        f = f.reshape(-1, f.shape[-1])
        sample.append(f[torch.from_numpy(rng.choice(len(f), min(len(f), 4000), replace=False)).to(dev)])
        print(f"[semantic] pca {j + 1} / {len(pick)}", flush=True)
    sample = torch.cat(sample)
    mean = sample.mean(0)
    _, s, vt = torch.linalg.svd(sample - mean, full_matrices=False)
    dims = int(params["dims"])
    basis = vt[:dims].T.contiguous()  # (D, dims)
    explained = float((s[:dims] ** 2).sum() / (s ** 2).sum())
    print(f"[semantic] PCA: {dims} of {sample.shape[1]} dims keep {100 * explained:.1f} % of the variance", flush=True)
    del sample

    # 3. back-projection with an occlusion test against the rendered depth
    acc = torch.zeros((len(means), dims), device=dev)
    wsum = torch.zeros(len(means), device=dev)
    dd = int(params["depth_div"])
    Wd, Hd = max(1, W // dd), max(1, H // dd)
    Kd = torch.tensor([[K0[0, 0] * sx / dd, 0, K0[0, 2] * sx / dd], [0, K0[1, 1] * sy / dd, K0[1, 2] * sy / dd],
                       [0, 0, 1]], device=dev, dtype=torch.float32)
    tol = float(params["depth_tol"])
    ones = torch.ones((len(means), 1), device=dev)
    for i in range(n):
        f = features(model, params["adaptor"], photo(paths[i], maps, (W, H)), head)
        fr = ((f - mean) @ basis).permute(2, 0, 1)[None]  # (1, dims, h, w)
        V = torch.tensor(viewmats[i], device=dev, dtype=torch.float32)
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
                colors=g["colors"], viewmats=V[None], Ks=Kd[None], width=Wd, height=Hd, sh_degree=sh_degree,
                render_mode="ED", near_plane=0.01, far_plane=1e10, packed=False)
            depth, alpha = colors[0, ..., 0], alphas[0, ..., 0]
            pc = (torch.cat([means, ones], 1) @ V.T)[:, :3]
            z = pc[:, 2]
            u = Kd[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + Kd[0, 2]
            v = Kd[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + Kd[1, 2]
            inside = (z > 0.01) & (u >= 0) & (u <= Wd - 1) & (v >= 0) & (v <= Hd - 1)
            idx = torch.nonzero(inside).squeeze(1)
            ui, vi = u[idx].round().long(), v[idx].round().long()
            dz = depth[vi, ui]
            vis = (alpha[vi, ui] > 0.5) & ((z[idx] - dz).abs() <= tol * dz)
            idx, uu, vv = idx[vis], u[idx][vis], v[idx][vis]
            grid = torch.stack([uu / (Wd - 1) * 2 - 1, vv / (Hd - 1) * 2 - 1], -1)[None, None]  # (1, 1, M, 2)
            samp = F.grid_sample(fr, grid, mode="bilinear", align_corners=True)[0, :, 0].T  # (M, dims)
            w = opac[idx]
            acc.index_add_(0, idx, samp * w[:, None])
            wsum.index_add_(0, idx, w)
        print(f"[semantic] image {i + 1} / {n}: {len(idx):,} gaussians seen", flush=True)
    seen = wsum > 0
    feats = torch.where(seen[:, None], acc / wsum.clamp(min=1e-12)[:, None], torch.zeros_like(acc))
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "features.part.npz")
    np.savez(tmp, feats=feats.cpu().numpy().astype(np.float16), weight=wsum.cpu().numpy().astype(np.float32),
             pca_mean=mean.cpu().numpy().astype(np.float32), pca_basis=basis.cpu().numpy().astype(np.float32),
             explained=np.float32(explained))
    os.replace(tmp, os.path.join(out_dir, "features.npz"))
    stats = {"version": params["version"], "adaptor": params["adaptor"], "width": W, "height": H, "dims": dims,
             "explained": round(explained, 4), "photos": n, "gaussians": int(len(means)),
             "seen": int(seen.sum()), "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(f"[semantic] done: {stats}", flush=True)


if __name__ == "__main__":
    main()

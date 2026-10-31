"""Difix refinement of a trained splat (run in nerfstudio's venv with the Difix packages first on PYTHONPATH and the
Difix3D repository importable; modelforge/difix.py starts it).

python difix_refine.py SPEC.json
SPEC: splat_npz (ENU: means, quats wxyz, scales log, opacities logit, sh (N, K, 3)), repo (Difix3D), model, transforms,
images_dirs (where the photos are, the training size first: images_N, images), root (the project: a photo not in
images_N is read at its transforms.json file_path, e.g. registered/images/), downscale, sky_dir (sky masks at the training size, or null), steps, rounds,
shift_m, rot_m (metres per radian in the pose distance), novel_prob, novel_weight, difix_width (0 = the training
size), targets (4 x 4 OpenCV c2w in the transforms' world), out_npz, workdir, cache_gb, flatten, needle,
needle_ratio (the shape terms of the splat's own training, nsplugin/mf_splatfacto_lidar.py; 0 = off), keep_shape (no
Gaussian more elongated than it was, or than needle_ratio; default true), cleaner ("difix" (default) or "fixer"; see
_cleaner), fixer_dir, fixer_timestep. novel_prob 0 trains on the photos only (a control: no views are cleaned).

A fine-tune of the splat as it is (no densification): every steps / rounds steps the pseudo views move shift_m
towards their targets (starting at the nearest training camera), are rendered with that camera's intrinsics, cleaned
by Difix (nvidia/difix_ref, half precision; the nearest training photo as the reference, brought to the median
exposure) and replace the previous set. A step trains on a pseudo view with probability novel_prob, its loss x
novel_weight and kept to the rendered alpha > 0.5 (as Difix3D's gsplat trainer: the sky and empty space stay out),
else on a training photo (exposure gain, sky masks, undistorted; read from disk, an LRU cache of cache_gb). The
splat's shape terms (flatten, no needles) apply as in its training, and keep_shape caps each Gaussian's elongation:
without them the fine-tune turned 27 % of kladno_dopravnik's Gaussians into needles (largest / middle scale > 3:
7.7 % -> 43.8 %; the terms alone: 18 %).
Measured on karlak (docs/todo_splat_quality.md): views aimed at held-out poses +0.65 dB there, LPIPS -0.04."""
import collections
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from pytorch_msssim import SSIM
from scipy.spatial.transform import Rotation, Slerp

LRS = {"means": 2e-4, "quats": 1e-3, "scales": 5e-3, "opacities": 2.5e-2, "sh0": 2.5e-3, "shN": 1.25e-4}


def _load_frame_camera():
    """export/frame_camera.py by its path: this folder on sys.path would shadow nerfstudio (export/nerfstudio.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mf_frame_camera", os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_camera.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pose_dist(a, b, rot_m):
    r = Rotation.from_matrix(a[:3, :3]).inv() * Rotation.from_matrix(b[:3, :3])
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) + rot_m * r.magnitude())


def nearest(train, poses, rot_m):
    """Index of the nearest training pose for each pose (the 32 nearest by position, then with the angle)."""
    tp = np.array([t[:3, 3] for t in train])
    out = []
    for p in poses:
        cand = np.argsort(np.linalg.norm(tp - p[:3, 3], axis=1))[:32]
        out.append(int(min(cand, key=lambda i: pose_dist(train[i], p, rot_m))))
    return out


def shift(cur, target, step_m, rot_m):
    """cur moved step_m towards target (in the pose distance), the rotation slerped in proportion."""
    d = pose_dist(cur, target, rot_m)
    if d <= step_m:
        return target.copy()
    a = step_m / d
    out = np.eye(4)
    out[:3, 3] = cur[:3, 3] + a * (target[:3, 3] - cur[:3, 3])
    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([cur[:3, :3], target[:3, :3]])))([a]).as_matrix()[0]
    return out


class Photos:
    """Training photos undistorted to their own pinhole K at the training size, with a validity / sky mask; read
    on demand, the most recent kept up to a memory budget."""

    def __init__(self, views, budget_gb):
        self.views, self.budget = views, budget_gb * 1e9
        self.cache, self.size, self.maps = collections.OrderedDict(), 0, {}

    def get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        v = self.views[i]
        W, H = v["W"], v["H"]
        key = (v["cam"], W, H)
        if key not in self.maps:
            self.maps[key] = cv2.initUndistortRectifyMap(v["K"], v["dist"], None, v["K"], (W, H), cv2.CV_32FC1)
        mx, my = self.maps[key]
        img = cv2.cvtColor(cv2.imread(v["path"]), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        keep = np.full((H, W), 255, np.uint8)
        if v["sky"] and os.path.exists(v["sky"]):  # 0 = sky: left out
            sk = cv2.imread(v["sky"], cv2.IMREAD_GRAYSCALE)
            keep = cv2.resize(sk, (W, H), interpolation=cv2.INTER_NEAREST) if sk.shape != (H, W) else sk
        out = (torch.from_numpy(cv2.remap(img, mx, my, cv2.INTER_LINEAR)),
               torch.from_numpy(cv2.remap(keep, mx, my, cv2.INTER_NEAREST) > 127))
        self.cache[i] = out
        self.size += out[0].numel() + out[1].numel()
        while self.size > self.budget and len(self.cache) > 1:
            _, (a, b) = self.cache.popitem(last=False)
            self.size -= a.numel() + b.numel()
        return out


def _cleaner(spec):
    """clean(render PIL, reference photo PIL) -> the cleaned render (PIL, the render's size), by spec["cleaner"]:
    "difix" (nvidia/difix_ref, Stable Diffusion turbo with the reference photo; at difix_width, 0 = the render's
    size) or "fixer" (nvidia/Fixer, Cosmos-Predict2 0.6B, one step, no reference; at difix_width or its training
    width 1024, the height by the aspect in multiples of 16; spec["fixer_dir"] holds build/fixer's run_fixer.py and
    models)."""
    w = int(spec.get("difix_width") or 0)
    if spec.get("cleaner", "difix") == "fixer":
        sys.path.insert(0, spec["fixer_dir"])
        import run_fixer
        model = run_fixer.load(int(spec.get("fixer_timestep", 250)))
        w = w or 1024

        def clean(pil, ref):
            size = (w // 16 * 16, max(16, int(round(pil.height * w / pil.width / 16)) * 16))
            return run_fixer.fix(model, pil, size)
        return clean
    sys.path.insert(0, spec["repo"])
    from src.pipeline_difix import DifixPipeline
    difix = DifixPipeline.from_pretrained(spec["model"], trust_remote_code=True, torch_dtype=torch.float16)
    difix.set_progress_bar_config(disable=True)
    difix.to("cuda")

    def clean(pil, ref):
        size = (w, int(round(pil.height * w / pil.width / 8)) * 8) if 0 < w < pil.width else pil.size
        size = (size[0] // 8 * 8, size[1] // 8 * 8)
        return difix(prompt="remove degradation", image=pil.resize(size, Image.LANCZOS),
                     ref_image=ref.resize(size, Image.LANCZOS), num_inference_steps=1, timesteps=[199],
                     guidance_scale=0.0).images[0].resize(pil.size, Image.LANCZOS)
    return clean


def main(spec_path):
    spec = json.load(open(spec_path))
    dev = torch.device("cuda")
    fc = _load_frame_camera()
    os.makedirs(spec["workdir"], exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()

    d = np.load(spec["splat_npz"])
    sh = torch.tensor(d["sh"], dtype=torch.float32, device=dev)
    P = {"means": d["means"], "quats": d["quats"], "scales": d["scales"], "opacities": d["opacities"]}
    P = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=dev)) for k, v in P.items()}
    P["sh0"] = torch.nn.Parameter(sh[:, :1].contiguous())
    P["shN"] = torch.nn.Parameter(sh[:, 1:].contiguous())
    del sh
    opt = {k: torch.optim.Adam([P[k]], lr=lr, eps=1e-15) for k, lr in LRS.items()}
    sh_degree = int(round(np.sqrt(P["sh0"].shape[1] + P["shN"].shape[1]))) - 1

    t = json.load(open(spec["transforms"]))
    ds = int(spec["downscale"])
    views = []
    cams, index = fc.cameras(t)
    for f, ci in zip(t["frames"], index):
        cam = fc.train_camera(t, cams[ci], ds)
        name = os.path.basename(f["file_path"])
        c2w = np.array(f["transform_matrix"], float)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
        views.append({"name": name, "c2w": c2w, "cam": ci, "W": int(cam["w"]), "H": int(cam["h"]),
                      "K": np.array([[cam["fl_x"], 0, cam["cx"]], [0, cam["fl_y"], cam["cy"]], [0, 0, 1]]),
                      "dist": np.array([cam.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")] + [cam.get("k3", 0.0)]),
                      "gain": float(f.get("mf_exposure", 1.0)),
                      "path": next((q for q in [os.path.join(dd, name) for dd in spec["images_dirs"][:1]]
                                    + ([os.path.join(spec["root"], f["file_path"])] if spec.get("root") else [])
                                    + [os.path.join(dd, name) for dd in spec["images_dirs"][1:]]
                                    if os.path.exists(q)), os.path.join(spec["images_dirs"][-1], name)),
                      "sky": os.path.join(spec["sky_dir"], name + ".png") if spec.get("sky_dir") else None})
    photos = Photos(views, float(spec.get("cache_gb", 4.0)))
    print(f"[difix] {len(views)} training photos, {len(P['means']):,} gaussians, {len(spec['targets'])} target "
          f"views", flush=True)
    train_poses = [v["c2w"] for v in views]
    targets = [np.array(x, float) for x in spec["targets"]]
    rot_m = float(spec.get("rot_m", 10.0))
    cur = [train_poses[i].copy() for i in nearest(train_poses, targets, rot_m)]

    def render(c2w, K, W, H):
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]
        img, alpha, _ = rasterization(
            means=P["means"], quats=P["quats"] / P["quats"].norm(dim=-1, keepdim=True), scales=torch.exp(P["scales"]),
            opacities=torch.sigmoid(P["opacities"]), colors=torch.cat([P["sh0"], P["shN"]], 1), viewmats=vm,
            Ks=torch.tensor(K, dtype=torch.float32, device=dev)[None], width=W, height=H, sh_degree=sh_degree,
            near_plane=0.01, far_plane=1e10, packed=False)
        return img[0].clamp(0, 1), alpha[0, ..., 0]

    clean = _cleaner(spec)
    print(f"[difix] {spec.get('cleaner', 'difix')} loaded ({time.time() - t0:.0f} s)", flush=True)

    novel = []
    rounds, steps = max(1, int(spec["rounds"])), int(spec["steps"])

    def fix(rnd):
        nonlocal novel, cur
        cur = [shift(c, tg, float(spec["shift_m"]), rot_m) for c, tg in zip(cur, targets)]
        refs = nearest(train_poses, cur, rot_m)
        sdir = os.path.join(spec["workdir"], f"round{rnd}")
        os.makedirs(sdir, exist_ok=True)
        out, t1 = [], time.time()
        for k, (c, r) in enumerate(zip(cur, refs)):
            ref = views[r]
            with torch.no_grad():
                img, alpha = render(c, ref["K"], ref["W"], ref["H"])
            photo, _ = photos.get(r)
            ref_lin = (photo.float() / 255).clamp(1e-6, 1) ** 2.2 / ref["gain"]  # to the median exposure
            ref_img = Image.fromarray((ref_lin.clamp(0, 1) ** (1 / 2.2) * 255).byte().numpy())
            pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
            torch.cuda.empty_cache()
            fixed = clean(pil, ref_img)
            if k < 8:  # render | Difix | reference, for a look
                Image.fromarray(np.concatenate([np.asarray(pil), np.asarray(fixed), np.asarray(ref_img)], 1)).resize(
                    (3 * 640, round(640 * pil.height / pil.width))).save(os.path.join(sdir, f"{k:03d}.jpg"), quality=85)
            out.append((c, ref["K"], ref["W"], ref["H"], torch.from_numpy(np.asarray(fixed).copy()),
                        (alpha > 0.5).cpu()))
            print(f"[difix] round {rnd} / {rounds}: view {k + 1} / {len(cur)}", flush=True)
        novel = out
        left = [pose_dist(c, tg, rot_m) for c, tg in zip(cur, targets)]
        print(f"[difix] round {rnd} / {rounds}: {len(out)} views cleaned in {time.time() - t1:.0f} s; still "
              f"{np.median(left):.1f} m (median) / {np.max(left):.1f} m from their targets", flush=True)

    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    p_novel, w_novel = float(spec.get("novel_prob", 0.3)), float(spec.get("novel_weight", 0.3))
    every = max(1, steps // rounds)
    flatten, needle = float(spec.get("flatten", 0.0)), float(spec.get("needle", 0.0))
    needle_ratio = float(spec.get("needle_ratio", 3.0))
    cap = None
    if spec.get("keep_shape", True):  # the largest / middle log-scale each Gaussian may reach
        ls = torch.sort(P["scales"].detach(), dim=-1).values
        cap = torch.clamp(ls[:, 2] - ls[:, 1], min=float(np.log(needle_ratio)))
        del ls
    t1 = time.time()
    for step in range(steps):
        if step % every == 0 and step // every < rounds and p_novel > 0:
            fix(step // every + 1)
        if novel and rng.random() < p_novel:
            c2w, K, W, H, target, mask = novel[rng.integers(len(novel))]
            gain, weight = 1.0, w_novel
        else:
            i = int(rng.integers(len(views)))
            v = views[i]
            target, mask = photos.get(i)
            c2w, K, W, H, gain, weight = v["c2w"], v["K"], v["W"], v["H"], v["gain"], 1.0
        img, _ = render(c2w, K, W, H)
        if gain != 1.0:  # the photo's exposure (as the plugin's use_exposure)
            img = ((img.clamp(1e-6, 1) ** 2.2) * gain).clamp(1e-6, 1) ** (1 / 2.2)
        gt = target.to(dev).float() / 255
        m = mask.to(dev)[..., None].float()
        l1 = ((img - gt).abs() * m).sum() / (m.sum() * 3).clamp(min=1)
        s = 1 - ssim((img * m).permute(2, 0, 1)[None], (gt * m).permute(2, 0, 1)[None])
        loss = weight * (0.8 * l1 + 0.2 * s)
        if flatten > 0 or needle > 0:  # as the splat's training (mf_splatfacto_lidar.get_loss_dict)
            ls = torch.sort(P["scales"], dim=-1).values  # log scales
            if flatten > 0:  # only the smallest axis may give
                loss = loss + flatten * torch.exp(ls[:, 0] - ls[:, 1].detach()).mean()
            if needle > 0:
                loss = loss + needle * torch.relu(ls[:, 2] - ls[:, 1] - np.log(needle_ratio)).mean()
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if cap is not None:  # no Gaussian more elongated than it was (or than needle_ratio)
            with torch.no_grad():
                ls, order = torch.sort(P["scales"], dim=-1)
                longest = torch.minimum(ls[:, 2], ls[:, 1] + cap)
                P["scales"].scatter_(1, order[:, 2:], longest[:, None])
        if (step + 1) % 100 == 0:
            print(f"[difix] step {step + 1} / {steps} ({time.time() - t1:.0f} s)", flush=True)
    np.savez(spec["out_npz"], means=P["means"].detach().cpu().numpy(), quats=P["quats"].detach().cpu().numpy(),
             scales=P["scales"].detach().cpu().numpy(), opacities=P["opacities"].detach().cpu().numpy(),
             sh=torch.cat([P["sh0"], P["shN"]], 1).detach().cpu().numpy())
    json.dump({"seconds": round(time.time() - t0, 1), "steps": steps, "rounds": rounds, "targets": len(targets),
               "photos": len(views), "gaussians": int(len(P["means"]))},
              open(os.path.join(spec["workdir"], "stats.json"), "w"), indent=1)
    print(f"[difix] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

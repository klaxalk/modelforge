"""splatfacto-lidar / splatfacto-big-lidar: splatfacto with LiDAR depth supervision (project kind rosbag).

Runs inside nerfstudio's venv (/opt/nerfstudio), never imported by modelforge itself. splat.py puts
this folder on PYTHONPATH and registers the methods and the dataparser through nerfstudio's
NERFSTUDIO_METHOD_CONFIGS / NERFSTUDIO_DATAPARSER_CONFIGS, so nothing is installed into the image.

  dataparser  lidar-data (mf_lidar_dataparser.py, kept apart: nerfstudio discovers dataparsers while
              its datamanager module is half imported, so that module must not import the model)
              = nerfstudio-data + transforms.json "lidar_ply_file_path" (the project's
              lidar/cloud.ply, world frame) mapped into nerfstudio's normalized frame
              (metadata["lidar_xyz"]), the same way nerfstudio maps ply_file_path.
  model       splatfacto, plus per training step: the LiDAR points are projected into the training
              camera (as undistorted by the datamanager, at the current training resolution, with the
              optimized pose) into a nearest-depth image. A return is kept where it is no more than
              lidar_occlusion_tol behind the nearest return in a lidar_occlusion_window window, so
              returns from behind a surface (seen through the gaps of a sparse cloud) are dropped.
              Loss: lidar_depth_lambda * mean(min(|D - z| / z, lidar_depth_clamp)) over those pixels,
              D = splatfacto's expected depth. The relative error weighs near and far alike; the clamp
              bounds the pull of wrong returns (moving objects, leaks).
Dense targets instead (dataparser --depth-maps: metadata["depth_files"], one z-depth map per training image at the
undistorted training camera, e.g. the SfM mesh rendered by modelforge/export/mesh_depth.py): the map of the training
image (camera.metadata["cam_idx"]), scaled into nerfstudio's units and resized (nearest) to the current training
resolution, replaces the projected points; the same loss over every pixel with a depth > 0.
Loss masks (dataparser --loss-masks: metadata["mask_files"], same camera as the depth maps): pixels with mask 0 are
left out of splatfacto's photometric loss (its batch["mask"]) and of the depth loss: a tile's pixels that show other
tiles, which it has no points for (it would paint them as fog in front of the camera; berlin tile 0).
Every lidar_report_every steps it prints "[lidar] step N: depth error median x% p90 y% within 5% z%"
(over the steps since the last report), which splat.py puts into report.json.

Regularization for flights that see the ground from one side only (a single straight pass at ~45 deg: the user
asked, 2026-10-22; a far-field mask tried alongside did not work well and was removed):
  mesh seeds      starting points from the SfM mesh (dataparser --seed-ply: metadata "mf_seed_normals", the last
                  points): they start flat in the surface, the size splatfacto gives them (mean distance to the 3
                  nearest points) as the width, SEED_THICKNESS of it along the normal.
  flatten_lambda  flatten_lambda x mean(smallest / middle scale) over the Gaussians, the middle scale held fixed
                  (only the smallest axis shrinks): flat discs on the surface (2DGS / PGSR style), and no needles
                  (two small axes) along the few viewing directions. 0 = off. It does not stop ribbons: a disc
                  whose largest axis runs away from the middle one (kladno_dira 2026-10-24: largest / middle
                  median 3.2, 30 % above 5).
  needle_lambda   needle_lambda x mean(relu(log(largest / middle scale) - log(needle_ratio))): ribbons / needles
                  longer than needle_ratio (3) x their width are pulled back (both axes may give). 0 = off.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from mf_lidar_dataparser import LidarDataParserConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.plugins.types import MethodSpecification


SEED_THICKNESS = 0.1  # mesh seeds start this thin along the surface normal (x their width)


def flat_quats(n):
    """wxyz quaternions of rotations taking the local z axis to the unit normals n (N, 3)."""
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype, device=n.device).expand_as(n)
    axis = torch.cross(z, n, dim=1)
    w = 1.0 + (z * n).sum(1, keepdim=True)  # half-angle form: q = (1 + z.n, z x n), normalized
    q = torch.cat([w, axis], 1)
    flip = w[:, 0] < 1e-6  # n = -z: any half turn about x
    q[flip] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
    return q / q.norm(dim=1, keepdim=True)


def lidar_depth_image(xyz, viewmat, K, W, H, window=5, tol=0.05):
    """(H, W, 1) depth of the nearest visible LiDAR return per pixel, 0 where there is none."""
    Xc = xyz @ viewmat[:3, :3].T + viewmat[:3, 3]
    Xc = Xc[Xc[:, 2] > 0.01]
    z = Xc[:, 2]
    u = torch.floor(K[0, 0] * Xc[:, 0] / z + K[0, 2]).long()
    v = torch.floor(K[1, 1] * Xc[:, 1] / z + K[1, 2]).long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zbuf = torch.full((H * W,), float("inf"), device=xyz.device)
    zbuf.scatter_reduce_(0, v[inside] * W + u[inside], z[inside], reduce="amin")
    zbuf = zbuf.view(1, 1, H, W)
    near = -F.max_pool2d(-zbuf, window, stride=1, padding=window // 2)
    valid = torch.isfinite(zbuf) & (zbuf <= near * (1 + tol))
    return torch.where(valid, zbuf, torch.zeros_like(zbuf))[0, 0, :, :, None]


def depth_loss(depth, z, clamp=1.0, gate=0.0):
    """mean(min(|D - z| / z, clamp)) over the targets, a target counting 0 where the render lies more than gate
    behind it (gate > 0; the photos see past the mesh there); a render in front of it (fog) always counts.
    Returns (loss, |D - z| / z of all, share gated)."""
    signed = (depth - z) / z
    rel = signed.abs()
    term = rel.clamp(max=clamp)
    if gate > 0:
        use = signed.detach() <= gate
        return (term * use).sum() / max(len(rel), 1), rel, 1.0 - use.float().mean().item()
    return term.mean(), rel, 0.0


@dataclass
class SplatfactoLidarModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoLidarModel)
    lidar_depth_lambda: float = 0.2
    """weight of the LiDAR depth loss (0: only report the depth error)"""
    lidar_depth_clamp: float = 1.0
    """relative depth errors are clamped to this in the loss"""
    lidar_depth_gate: float = 0.5
    """dense mesh maps only: after lidar_depth_gate_after steps, a pixel whose rendered depth lies more than this
    fraction *behind* the mesh gets no target (the photos see past the mesh there: Poisson bubbles in halls, closures
    above the horizon, blobs at crowns); a render in front of the mesh (fog) is always pulled back. 0 = off"""
    lidar_depth_gate_after: int = 2000
    lidar_occlusion_window: int = 5
    """window (px, at the training resolution) for dropping returns behind a nearer one"""
    lidar_occlusion_tol: float = 0.05
    """a return may lie this fraction behind the nearest return in its window"""
    lidar_max_points: int = 3_000_000
    """random subset of the cloud kept on the GPU"""
    lidar_report_every: int = 500
    flatten_lambda: float = 0.0
    """weight of mean(smallest / middle scale): flat Gaussians, no needles (0 = off)"""
    needle_lambda: float = 0.0
    """weight of mean(relu(log(largest / middle scale) - log(needle_ratio))): no needles (0 = off). Flattening only
    shrinks the smallest axis; nothing kept the largest from running away from the middle one"""
    needle_ratio: float = 3.0
    sky_alpha_lambda: float = 0.0
    """weight of the mean rendered opacity over sky pixels (batch["mask"] 0: the dataparser's sky masks, already out
    of the photometric loss), so no Gaussians stand in for the sky (0 = off)"""
    sky_core_margin: float = 0.01
    """the opacity loss only where the sky is this share of the width away from anything else"""
    use_exposure: bool = False
    """scale each training render by the photo's relative exposure (dataparser metadata "exposure_gains", from the
    DJI metadata), in linear light: the Gaussians hold the colours at the median exposure"""


class SplatfactoLidarModel(SplatfactoModel):
    config: SplatfactoLidarModelConfig

    def populate_modules(self):
        super().populate_modules()
        nrm = (self.kwargs.get("metadata") or {}).get("mf_seed_normals")
        if nrm is not None and len(nrm) and self.seed_points is not None:  # the mesh seeds: flat in the surface
            m, n = len(nrm), len(self.gauss_params["means"])
            with torch.no_grad():
                self.gauss_params["quats"][n - m:] = flat_quats(nrm.float())
                s = self.gauss_params["scales"][n - m:]  # log(mean distance to the 3 nearest points), x3
                s[:, 2] = s[:, 0] + float(np.log(SEED_THICKNESS))
            print(f"[lidar] {m:,} of {n:,} starting Gaussians from the SfM mesh, flat in its surface", flush=True)
        xyz = (self.kwargs.get("metadata") or {}).get("lidar_xyz")
        if xyz is not None and len(xyz) > self.config.lidar_max_points:
            xyz = xyz[torch.randperm(len(xyz), generator=torch.Generator().manual_seed(0))[
                :self.config.lidar_max_points]]
        meta = self.kwargs.get("metadata") or {}
        self.depth_files, self.depth_scale = meta.get("depth_files"), float(meta.get("depth_scale", 1.0))
        self.mask_files = meta.get("mask_files")
        self.exposure_gains = meta.get("exposure_gains") if self.config.use_exposure else None
        if self.config.use_exposure:
            g = np.array(self.exposure_gains or [1.0])
            print(f"[lidar] exposure gains for {len(self.exposure_gains or [])} training images (x{g.min():.2f} .. "
                  f"x{g.max():.2f})" if self.exposure_gains else "[lidar] use_exposure: no exposure gains in "
                  "transforms.json (frames mf_exposure)", flush=True)
        if self.mask_files:
            print(f"[lidar] loss masks for {sum(f is not None for f in self.mask_files)} of {len(self.mask_files)} "
                  "training images", flush=True)
        self.has_lidar = (xyz is not None and len(xyz) > 0) or bool(self.depth_files)
        # not persistent: checkpoints stay as splatfacto's (the live preview and ns-export read them)
        self.register_buffer("lidar_xyz", xyz.float() if xyz is not None and len(xyz) else torch.zeros(0, 3),
                             persistent=False)
        if self.has_lidar:
            self.config.output_depth_during_training = True
        self._lidar_errors, self._lidar_gated = [], []
        print(f"[lidar] dense depth maps for {sum(f is not None for f in self.depth_files)} of "
              f"{len(self.depth_files)} training images (nerfstudio holds out every 10th photo for evaluation), "
              f"depth loss weight {self.config.lidar_depth_lambda}"
              if self.depth_files else
              f"[lidar] {len(self.lidar_xyz)} LiDAR points, depth loss weight {self.config.lidar_depth_lambda}"
              if self.has_lidar else "[lidar] no LiDAR points (transforms.json lidar_ply_file_path): "
              "training as plain splatfacto", flush=True)

    def _loss_mask(self, camera):
        """(H, W) bool loss mask of the training image at the undistorted training camera, or None."""
        idx = (camera.metadata or {}).get("cam_idx")
        f = self.mask_files[int(idx)] if self.mask_files and idx is not None else None
        if f is None:
            return None
        import cv2
        return torch.from_numpy(cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127)

    def get_outputs(self, camera):
        out = super().get_outputs(camera)
        if self.training and self.exposure_gains:
            idx = (camera.metadata or {}).get("cam_idx")
            if idx is not None:
                g = float(self.exposure_gains[int(idx)])
                lin = out["rgb"].clamp(1e-6, 1) ** 2.2  # sRGB-like camera response, approximated by a gamma
                out["rgb"] = (lin * g).clamp(1e-6, 1) ** (1 / 2.2)
        if self.training and self.mask_files:
            m = self._loss_mask(camera)
            if m is not None:
                out["mf_mask"] = m  # full training size, as batch["image"]: splatfacto downscales it
        if self.training and self.depth_files and out.get("depth") is not None:
            idx = (camera.metadata or {}).get("cam_idx")
            f = self.depth_files[int(idx)] if idx is not None else None
            if f is not None:
                with torch.no_grad():
                    d = torch.from_numpy(np.load(f).astype(np.float32)).to(out["depth"].device) * self.depth_scale
                    H, W = out["depth"].shape[:2]
                    if d.shape != (H, W):
                        d = F.interpolate(d[None, None], size=(H, W), mode="nearest")[0, 0]
                    out["lidar_depth"] = d[:, :, None]
        elif self.training and self.has_lidar and out.get("depth") is not None:
            with torch.no_grad():
                c2w = self.camera_optimizer.apply_to_camera(camera)
                d = self._get_downscale_factor()
                camera.rescale_output_resolution(1 / d)
                K = camera.get_intrinsics_matrices()[0].to(self.lidar_xyz.device)
                W, H = int(camera.width.item()), int(camera.height.item())
                camera.rescale_output_resolution(d)
                out["lidar_depth"] = lidar_depth_image(self.lidar_xyz, get_viewmat(c2w)[0], K, W, H,
                                                       self.config.lidar_occlusion_window,
                                                       self.config.lidar_occlusion_tol)
        return out

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        sky_loss = None
        if self.training and self.config.sky_alpha_lambda > 0 and "mask" in batch and "accumulation" in outputs:
            keep = self._downscale_if_required(batch["mask"].float()).to(outputs["accumulation"].device)
            # the core of the sky: CORE px (1 % of the width) away from anything else, so Gaussians straddling the
            # horizon (a treeline) are not thinned
            r = max(1, round(self.config.sky_core_margin * keep.shape[1]))
            near = F.max_pool2d(keep[..., 0][None, None], 2 * r + 1, 1, r)[0, 0]
            sky = near < 0.5
            if sky.shape == outputs["accumulation"].shape[:2] and sky.any():
                sky_loss = self.config.sky_alpha_lambda * outputs["accumulation"][..., 0][sky].mean() * sky.float().mean()
        m = outputs.get("mf_mask")
        if m is not None and m.shape[:2] == batch["image"].shape[:2]:
            mm = m[:, :, None].float()
            if "mask" in batch:  # the sky masks too: both leave pixels out
                mm = mm * batch["mask"].float().to(mm.device)
            batch = dict(batch, mask=mm)
            target = outputs.get("lidar_depth")
            if target is not None:
                H, W = target.shape[:2]
                mm = F.interpolate(m[None, None].float(), size=(H, W), mode="nearest")[0, 0].to(target.device) > 0.5
                outputs = dict(outputs, lidar_depth=torch.where(mm[:, :, None], target, torch.zeros_like(target)))
        loss = super().get_loss_dict(outputs, batch, metrics_dict)
        if sky_loss is not None:
            loss["sky_alpha"] = sky_loss
        if self.training and self.config.flatten_lambda > 0:
            # only the smallest axis may give: with the middle one free, the optimizer shrank the ratio by
            # widening the Gaussians into huge sheets (berlin tile 1: 1.8 -> 21.7 GB of GPU memory, out of memory)
            s = torch.sort(torch.exp(self.scales), dim=-1).values
            loss["flatten"] = self.config.flatten_lambda * (s[:, 0] / s[:, 1].detach().clamp(min=1e-12)).mean()
        if self.training and self.config.needle_lambda > 0:
            ls = torch.sort(self.scales, dim=-1).values  # log scales
            loss["needle"] = self.config.needle_lambda * torch.relu(
                ls[:, 2] - ls[:, 1] - float(np.log(self.config.needle_ratio))).mean()
        target = outputs.get("lidar_depth")
        if target is None:
            return loss
        m = target > 0
        if m.any():
            z = target[m]
            gate = self.config.lidar_depth_gate if (
                self.depth_files and self.step >= self.config.lidar_depth_gate_after) else 0.0
            term, rel, gated = depth_loss(outputs["depth"][m], z, self.config.lidar_depth_clamp, gate)
            if self.config.lidar_depth_lambda > 0:
                loss["lidar_depth_loss"] = self.config.lidar_depth_lambda * term
            self._report(rel.detach(), gated)
        return loss

    def _report(self, rel, gated=0.0):
        if len(rel) > 4000:
            rel = rel[torch.randint(len(rel), (4000,), device=rel.device)]
        self._lidar_errors.append(rel.float().cpu())
        self._lidar_gated.append(gated)
        every = self.config.lidar_report_every
        if every and (self.step + 1) % every == 0:
            e = torch.cat(self._lidar_errors).numpy()
            g = float(np.mean(self._lidar_gated))
            self._lidar_errors, self._lidar_gated = [], []
            print(f"[lidar] step {self.step + 1}: depth error median {100 * np.median(e):.2f}% "
                  f"p90 {100 * np.percentile(e, 90):.2f}% within 5% {100 * np.mean(e < 0.05):.1f}%"
                  + (f" gated {100 * g:.1f}%" if g > 0 else ""), flush=True)


def _method(base, name, description):
    from nerfstudio.configs.method_configs import method_configs  # filled before plugins are discovered
    cfg = deepcopy(method_configs[base])
    cfg.method_name = name
    m = cfg.pipeline.model
    cfg.pipeline.model = SplatfactoLidarModelConfig(
        **{f.name: getattr(m, f.name) for f in fields(m) if f.init and f.name != "_target"})
    cfg.pipeline.datamanager.dataparser = LidarDataParserConfig()
    return MethodSpecification(config=cfg, description=description)


def splatfacto_lidar():
    return _method("splatfacto", "splatfacto-lidar", "splatfacto with LiDAR depth supervision")


def splatfacto_big_lidar():
    return _method("splatfacto-big", "splatfacto-big-lidar", "splatfacto-big with LiDAR depth supervision")

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
Every lidar_report_every steps it prints "[lidar] step N: depth error median x% p90 y% within 5% z%"
(over the steps since the last report), which splat.py puts into report.json.

Regularization for flights that see the ground from one side only (a single straight pass at ~45 deg: the user
asked, 2026-10-22):
  far_mask        with the dense depth maps: pixels deeper than far_mask x the image's median depth leave the colour
                  and depth losses (the rendered colour is replaced by the photo's there: no loss, no gradient, no
                  densification). The far, grazing part of an oblique photo is low resolution and smeared along the
                  flight; it pulls the Gaussians the sharp near views refine towards a blur. 0 = off.
  flatten_lambda  flatten_lambda x mean(smallest / middle scale) over the Gaussians: flat discs on the surface
                  (2DGS / PGSR style), and no needles (two small axes) along the few viewing directions. 0 = off.
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


@dataclass
class SplatfactoLidarModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoLidarModel)
    lidar_depth_lambda: float = 0.2
    """weight of the LiDAR depth loss (0: only report the depth error)"""
    lidar_depth_clamp: float = 1.0
    """relative depth errors are clamped to this in the loss"""
    lidar_occlusion_window: int = 5
    """window (px, at the training resolution) for dropping returns behind a nearer one"""
    lidar_occlusion_tol: float = 0.05
    """a return may lie this fraction behind the nearest return in its window"""
    lidar_max_points: int = 3_000_000
    """random subset of the cloud kept on the GPU"""
    lidar_report_every: int = 500
    far_mask: float = 0.0
    """with dense depth maps: leave pixels deeper than this x the image's median depth out of the losses (0 = off)"""
    flatten_lambda: float = 0.0
    """weight of mean(smallest / middle scale): flat Gaussians, no needles (0 = off)"""


class SplatfactoLidarModel(SplatfactoModel):
    config: SplatfactoLidarModelConfig

    def populate_modules(self):
        super().populate_modules()
        xyz = (self.kwargs.get("metadata") or {}).get("lidar_xyz")
        if xyz is not None and len(xyz) > self.config.lidar_max_points:
            xyz = xyz[torch.randperm(len(xyz), generator=torch.Generator().manual_seed(0))[
                :self.config.lidar_max_points]]
        meta = self.kwargs.get("metadata") or {}
        self.depth_files, self.depth_scale = meta.get("depth_files"), float(meta.get("depth_scale", 1.0))
        self.has_lidar = (xyz is not None and len(xyz) > 0) or bool(self.depth_files)
        # not persistent: checkpoints stay as splatfacto's (the live preview and ns-export read them)
        self.register_buffer("lidar_xyz", xyz.float() if xyz is not None and len(xyz) else torch.zeros(0, 3),
                             persistent=False)
        if self.has_lidar:
            self.config.output_depth_during_training = True
        self._lidar_errors, self._far_shares = [], []
        print(f"[lidar] dense depth maps for {sum(f is not None for f in self.depth_files)} of "
              f"{len(self.depth_files)} training images (nerfstudio holds out every 10th photo for evaluation), "
              f"depth loss weight {self.config.lidar_depth_lambda}"
              if self.depth_files else
              f"[lidar] {len(self.lidar_xyz)} LiDAR points, depth loss weight {self.config.lidar_depth_lambda}"
              if self.has_lidar else "[lidar] no LiDAR points (transforms.json lidar_ply_file_path): "
              "training as plain splatfacto", flush=True)

    def get_outputs(self, camera):
        out = super().get_outputs(camera)
        if self.training and self.depth_files and out.get("depth") is not None:
            idx = (camera.metadata or {}).get("cam_idx")
            f = self.depth_files[int(idx)] if idx is not None else None
            if f is not None:
                with torch.no_grad():
                    d = torch.from_numpy(np.load(f).astype(np.float32)).to(out["depth"].device) * self.depth_scale
                    H, W = out["depth"].shape[:2]
                    if d.shape != (H, W):
                        d = F.interpolate(d[None, None], size=(H, W), mode="nearest")[0, 0]
                    if self.config.far_mask > 0 and bool((d > 0).any()):
                        far = d > self.config.far_mask * d[d > 0].median()
                        out["mf_far"] = far[:, :, None]
                        d = torch.where(far, torch.zeros_like(d), d)  # the mesh is least reliable out there too
                        self._far_shares.append(float(far.float().mean()))
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
        far = outputs.get("mf_far")
        if far is not None and bool(far.any()):  # the photo's own colour there: no loss, no gradient
            gt = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            outputs = dict(outputs, rgb=torch.where(far, gt, outputs["rgb"]))
        loss = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.config.flatten_lambda > 0:
            s = torch.sort(torch.exp(self.scales), dim=-1).values
            loss["flatten"] = self.config.flatten_lambda * (s[:, 0] / s[:, 1].clamp(min=1e-12)).mean()
        target = outputs.get("lidar_depth")
        if target is None:
            return loss
        m = target > 0
        if m.any():
            z = target[m]
            rel = (outputs["depth"][m] - z).abs() / z
            if self.config.lidar_depth_lambda > 0:
                loss["lidar_depth_loss"] = self.config.lidar_depth_lambda * rel.clamp(
                    max=self.config.lidar_depth_clamp).mean()
            self._report(rel.detach())
        return loss

    def _report(self, rel):
        if len(rel) > 4000:
            rel = rel[torch.randint(len(rel), (4000,), device=rel.device)]
        self._lidar_errors.append(rel.float().cpu())
        every = self.config.lidar_report_every
        if every and (self.step + 1) % every == 0 and self._far_shares:
            print(f"[lidar] step {self.step + 1}: far-field mask {100 * np.mean(self._far_shares):.1f}% of the "
                  "pixels", flush=True)
            self._far_shares = []
        if every and (self.step + 1) % every == 0:
            e = torch.cat(self._lidar_errors).numpy()
            self._lidar_errors = []
            print(f"[lidar] step {self.step + 1}: depth error median {100 * np.median(e):.2f}% "
                  f"p90 {100 * np.percentile(e, 90):.2f}% within 5% {100 * np.mean(e < 0.05):.1f}%", flush=True)


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

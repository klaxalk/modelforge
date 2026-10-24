"""The lidar-data dataparser of splatfacto-lidar (see mf_splatfacto_lidar.py): nerfstudio-data plus the
LiDAR cloud of transforms.json "lidar_ply_file_path" (else the SfM cloud, "ply_file_path": the SfM points as depth
targets; --depth-ply overrides both, e.g. the SfM mesh's vertices), in nerfstudio's normalized frame as metadata["lidar_xyz"]. Imports only the dataparser side of nerfstudio
(no circular import)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json


@dataclass
class LidarDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: LidarDataParser)
    load_3D_points: bool = True
    depth_ply: Optional[Path] = None
    """Depth points from this file instead (absolute, or relative to the data folder; e.g. the SfM mesh's
    vertices, modelforge/sfm_mesh.py)."""
    depth_maps: Optional[Path] = None
    """Dense depth targets instead of points: a folder of <image name>.npy z-depth maps (world units, 0 = none) at
    the undistorted training camera (modelforge/export/mesh_depth.py: the SfM mesh rendered)."""
    loss_masks: Optional[Path] = None
    """Per-image loss masks: a folder of <image name>.png (255 = in the losses, 0 = left out) at the undistorted
    training camera, as the depth maps (modelforge splat_tiles.loss_masks: a tile's pixels showing other tiles)."""
    sky_masks: Optional[Path] = None
    """Sky masks: a folder of <image name>.png (0 = sky, 255 = the rest) at the training images' size, before
    undistortion (modelforge/sky.py): nerfstudio's own masks (undistorted with the photo, left out of the photometric
    loss); the model adds its opacity loss on them (sky_alpha_lambda)."""
    seed_ply: Optional[Path] = None
    """Extra starting points with normals and colours (modelforge sfm_mesh.seed_points: the SfM mesh where the SfM
    points are thin), appended to the SfM points; the model starts them flat in the surface (metadata
    "mf_seed_normals", the last len of the points)."""


class LidarDataParser(Nerfstudio):
    config: LidarDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        out = super()._generate_dataparser_outputs(split)
        data = self.config.data
        meta = load_from_json(data if data.suffix == ".json" else data / "transforms.json")
        data_dir = data.parent if data.suffix == ".json" else data
        T = out.dataparser_transform.float()
        if self.config.seed_ply and "points3D_xyz" in out.metadata:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(data_dir / self.config.seed_ply))
            if len(pcd.points):
                xyz = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
                rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).round().astype(np.uint8))
                nrm = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
                out.metadata["points3D_xyz"] = torch.cat([out.metadata["points3D_xyz"],
                                                          (xyz @ T[:, :3].T + T[:, 3]) * out.dataparser_scale])
                out.metadata["points3D_rgb"] = torch.cat([out.metadata["points3D_rgb"], rgb])
                out.metadata["mf_seed_normals"] = nrm @ T[:, :3].T  # rotation only: directions
        # relative exposure per photo (modelforge/exposure.py, transforms.json frames "mf_exposure"; 1 = the median)
        gain = {Path(f["file_path"]).name: float(f["mf_exposure"]) for f in meta.get("frames", []) if "mf_exposure" in f}
        if gain:
            out.metadata["exposure_gains"] = [gain.get(Path(f).name, 1.0) for f in out.image_filenames]
        if self.config.sky_masks:
            folder = Path(self.config.sky_masks)
            files = [folder / (Path(f).name + ".png") for f in out.image_filenames]
            missing = [f.name for f in files if not f.exists()]
            if missing:
                raise FileNotFoundError(f"sky masks missing for {len(missing)} photos, e.g. {missing[0]} in {folder}")
            out.mask_filenames = files
        if self.config.loss_masks:
            folder = Path(self.config.loss_masks)
            files = [str(folder / (Path(f).name + ".png")) for f in out.image_filenames]
            out.metadata["mask_files"] = [f if Path(f).exists() else None for f in files]
        if self.config.depth_maps:
            folder = Path(self.config.depth_maps)
            files = [str(folder / (Path(f).name + ".npy")) for f in out.image_filenames]
            out.metadata["depth_files"] = [f if Path(f).exists() else None for f in files]
            out.metadata["depth_scale"] = float(out.dataparser_scale)  # world units -> nerfstudio's
            return out
        cloud = self.config.depth_ply or meta.get("lidar_ply_file_path", meta.get("ply_file_path"))
        if cloud:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(data_dir / cloud))
            xyz = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
            # world (= transforms.json frame; modelforge writes no applied_transform) -> normalized
            out.metadata["lidar_xyz"] = (xyz @ T[:, :3].T + T[:, 3]) * out.dataparser_scale
        return out


def lidar_dataparser():
    return DataParserSpecification(config=LidarDataParserConfig(),
                                   description="nerfstudio-data + LiDAR points (lidar_ply_file_path)")

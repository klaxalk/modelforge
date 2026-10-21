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


class LidarDataParser(Nerfstudio):
    config: LidarDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        out = super()._generate_dataparser_outputs(split)
        data = self.config.data
        meta = load_from_json(data if data.suffix == ".json" else data / "transforms.json")
        data_dir = data.parent if data.suffix == ".json" else data
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
            T = out.dataparser_transform.float()
            out.metadata["lidar_xyz"] = (xyz @ T[:, :3].T + T[:, 3]) * out.dataparser_scale
        return out


def lidar_dataparser():
    return DataParserSpecification(config=LidarDataParserConfig(),
                                   description="nerfstudio-data + LiDAR points (lidar_ply_file_path)")

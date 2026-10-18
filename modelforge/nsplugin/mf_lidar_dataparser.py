"""The lidar-data dataparser of splatfacto-lidar (see mf_splatfacto_lidar.py): nerfstudio-data plus the
LiDAR cloud of transforms.json "lidar_ply_file_path", in nerfstudio's normalized frame as
metadata["lidar_xyz"]. Imports only the dataparser side of nerfstudio (no circular import)."""

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json


@dataclass
class LidarDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: LidarDataParser)
    load_3D_points: bool = True


class LidarDataParser(Nerfstudio):
    config: LidarDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        out = super()._generate_dataparser_outputs(split)
        data = self.config.data
        meta = load_from_json(data if data.suffix == ".json" else data / "transforms.json")
        data_dir = data.parent if data.suffix == ".json" else data
        if "lidar_ply_file_path" in meta:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(data_dir / meta["lidar_ply_file_path"]))
            xyz = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
            # world (= transforms.json frame; modelforge writes no applied_transform) -> normalized
            T = out.dataparser_transform.float()
            out.metadata["lidar_xyz"] = (xyz @ T[:, :3].T + T[:, 3]) * out.dataparser_scale
        return out


def lidar_dataparser():
    return DataParserSpecification(config=LidarDataParserConfig(),
                                   description="nerfstudio-data + LiDAR points (lidar_ply_file_path)")

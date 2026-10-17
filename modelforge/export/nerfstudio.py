"""nerfstudio dataset export (transforms.json + sparse_pc.ply).

Follows nerfstudio/process_data/colmap_utils.py:colmap_to_json, except that no
applied_transform is needed: our world is already ENU (z up, metric), whereas
nerfstudio permutes axes on its COLMAP path because COLMAP's world is arbitrary.
"""

import json
import os

import numpy as np


def export(rec, project, image_dir="images", ply_name="sparse_pc.ply", extra=None):
    cams = list(rec.cameras.values())
    if len(cams) != 1:
        raise ValueError(f"expected one shared camera, got {len(cams)}")
    cam = cams[0]
    if cam.model.name != "OPENCV":
        raise ValueError(f"expected OPENCV camera, got {cam.model.name}")
    fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in cam.params)

    frames = []
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.image(i).name):
        img = rec.image(image_id)
        w2c = np.eye(4)
        w2c[:3] = img.cam_from_world().matrix()
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1  # OpenCV -> OpenGL camera axes
        frames.append({"file_path": f"{image_dir}/{img.name}", "transform_matrix": c2w.tolist(),
                       "colmap_im_id": int(image_id)})

    rec.export_PLY(os.path.join(project, ply_name))
    out = {"camera_model": "OPENCV", "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
           "w": int(cam.width), "h": int(cam.height), "k1": k1, "k2": k2, "p1": p1, "p2": p2,
           "frames": frames, "ply_file_path": ply_name}
    if extra:
        out["modelforge"] = extra
    path = os.path.join(project, "transforms.json")
    json.dump(out, open(path, "w"), indent=1)
    return path

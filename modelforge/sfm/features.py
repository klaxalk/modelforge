"""SIFT feature extraction into a COLMAP database (one shared OPENCV camera)."""

import json
import os
import time

import pycolmap


def extract(project, max_num_features=8192, camera_model="OPENCV", upright=True):
    """Write PROJECT/features.db (keypoints/descriptors only) for the images in poses.json.

    Matching works on a copy, so pairs can be re-matched without re-extracting.
    """
    poses = json.load(open(os.path.join(project, "poses.json")))
    names = [r["image"] for r in poses["images"]]
    db_path = os.path.join(project, "features.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = max_num_features
    # The gimbal keeps the horizon level (camera roll ~0, max ~7 deg), so one fixed
    # orientation per keypoint; otherwise each keypoint may get 2 (~16k descriptors).
    extraction.sift.upright = upright
    t0 = time.time()
    pycolmap.extract_features(db_path, os.path.join(project, "images"), image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction)
    db = pycolmap.Database.open(db_path)
    stats = {"images": db.num_images(), "keypoints": db.num_keypoints(), "cameras": db.num_cameras(),
             "seconds": round(time.time() - t0, 1)}
    db.close()
    return stats

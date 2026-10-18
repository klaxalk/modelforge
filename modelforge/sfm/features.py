"""SIFT feature extraction into a COLMAP database (one shared camera; OPENCV for DJI, RADIAL for videos
without metadata). Options: EXTRACTORS, max features, upright."""

import json
import os
import time

import pycolmap


EXTRACTORS = {
    "sift": "SIFT on the GPU: fast, the default",
    "sift-robust": "DSP-SIFT (domain-size pooling) + affine shape estimation: better matches across large viewpoint "
                   "changes, runs on the CPU (slower)",
}


def extract(project, max_num_features=8192, camera_model="OPENCV", upright=True, extractor="sift", camera_params=None):
    """Write PROJECT/features.db (keypoints/descriptors only) for the images in poses.json.

    Matching works on a copy, so pairs can be re-matched without re-extracting.
    extractor: EXTRACTORS (learned ones such as ALIKED need COLMAP built with ONNX, which this image is not).
    """
    if extractor not in EXTRACTORS:
        raise ValueError(f"extractor must be one of {tuple(EXTRACTORS)}")
    poses = json.load(open(os.path.join(project, "poses.json")))
    names = [r["image"] for r in poses["images"]]
    db_path = os.path.join(project, "features.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if camera_params:  # known intrinsics (ROS camera_info), kept fixed in BA
        reader.camera_params = ",".join(f"{x:.10g}" for x in camera_params)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = max_num_features
    # The gimbal keeps the horizon level (camera roll ~0, max ~7 deg), so one fixed
    # orientation per keypoint; otherwise each keypoint may get 2 (~16k descriptors).
    extraction.sift.upright = upright
    if extractor == "sift-robust":  # GPU SIFT supports neither, so this runs on the CPU
        extraction.sift.domain_size_pooling = True
        extraction.sift.estimate_affine_shape = True
        extraction.use_gpu = False
    t0 = time.time()
    pycolmap.extract_features(db_path, os.path.join(project, "images"), image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction)
    db = pycolmap.Database.open(db_path)
    stats = {"images": db.num_images(), "keypoints": db.num_keypoints(), "cameras": db.num_cameras(),
             "extractor": extractor, "max_num_features": max_num_features, "upright": upright,
             "seconds": round(time.time() - t0, 1)}
    db.close()
    return stats

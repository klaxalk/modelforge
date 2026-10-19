"""SIFT feature extraction into a COLMAP database (one shared camera; OPENCV for DJI, RADIAL for videos
without metadata). Options: EXTRACTORS, max features, upright."""

import json
import os
import sqlite3
import threading
import time

import pycolmap


EXTRACTORS = {
    "sift": "SIFT on the GPU: fast, the default",
    "sift-robust": "DSP-SIFT (domain-size pooling) + affine shape estimation: better matches across large viewpoint "
                   "changes, runs on the CPU (slower)",
}


def _watch(db_path, total, progress, done, every=0.25):
    """progress(images done, total, last image name) while COLMAP fills the database: extract_features
    has no callback, but each image is committed as it finishes."""
    last = -1
    while not done.wait(every):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.1)
            try:
                n = con.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
                row = con.execute("SELECT name FROM images ORDER BY image_id DESC LIMIT 1").fetchone()
            finally:
                con.close()
        except sqlite3.Error:  # not created yet, or busy
            continue
        if n != last:
            last = n
            progress(n, total, row[0] if row else "")


def extract(project, max_num_features=8192, camera_model="OPENCV", upright=True, extractor="sift", camera_params=None,
            progress=None):
    """Write PROJECT/features.db (keypoints/descriptors only) for the images in poses.json.

    Matching works on a copy, so pairs can be re-matched without re-extracting.
    extractor: EXTRACTORS (learned ones such as ALIKED need COLMAP built with ONNX, which this image is not).
    progress(done, total, name): images finished so far (polled from the database a few times a second).
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
    done = threading.Event()
    if progress:
        threading.Thread(target=_watch, args=(db_path, len(names), progress, done), daemon=True).start()
    try:
        pycolmap.extract_features(db_path, os.path.join(project, "images"), image_names=names,
                                  camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                  extraction_options=extraction)
    finally:
        done.set()
    if progress:
        progress(len(names), len(names), names[-1] if names else "")
    db = pycolmap.Database.open(db_path)
    stats = {"images": db.num_images(), "keypoints": db.num_keypoints(), "cameras": db.num_cameras(),
             "extractor": extractor, "max_num_features": max_num_features, "upright": upright,
             "seconds": round(time.time() - t0, 1)}
    db.close()
    return stats

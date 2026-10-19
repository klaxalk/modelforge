"""Reading ROS 2 bags (sqlite3 or MCAP) with the rosbags library: topic inventory, TF, images,
camera_info, odometry and point clouds. Times are in seconds (header stamps)."""

import os
from functools import lru_cache

import cv2
import numpy as np

from modelforge.rosbag.tf import TFBuffer, make, quat_to_rot

IMAGE_TYPES = ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
INFO_TYPE = "sensor_msgs/msg/CameraInfo"
ODOM_TYPE = "nav_msgs/msg/Odometry"
CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
TF_TYPE = "tf2_msgs/msg/TFMessage"


def is_bag(path):
    """A ROS 2 bag: a folder with metadata.yaml, or a single .mcap file."""
    return (os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.yaml"))) or \
        (os.path.isfile(path) and path.lower().endswith(".mcap"))


def find_bags(root):
    out = []
    for d, dirs, files in os.walk(root):
        if os.path.exists(os.path.join(d, "metadata.yaml")):
            out.append(d)
            dirs[:] = []
            continue
        if any(f in files for f in ("poses.json", "ingest.json")):  # a project folder
            dirs[:] = []
            continue
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(".mcap")]
    return out


def reader(path):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path
    return AnyReader([Path(path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


@lru_cache(maxsize=16)
def _inventory(path, mtime):
    with reader(path) as r:
        topics = {c.topic: {"type": c.msgtype, "count": c.msgcount} for c in r.connections}
        start, end = r.start_time * 1e-9, r.end_time * 1e-9
        frames = {}
        for c in r.connections:  # frame_id of the first message of the sensor topics
            if c.msgtype in IMAGE_TYPES + (CLOUD_TYPE, ODOM_TYPE, INFO_TYPE):
                for conn, _, raw in r.messages(connections=[c]):
                    m = r.deserialize(raw, conn.msgtype)
                    frames[c.topic] = m.header.frame_id.lstrip("/")
                    if c.msgtype == ODOM_TYPE:
                        frames[c.topic + "#child"] = m.child_frame_id.lstrip("/")
                    break
    return {"topics": topics, "start": start, "end": end, "frame_of": frames}


def inventory(path):
    """{"topics": {name: {type, count}}, "start", "end" [s], "frame_of": {topic: frame_id}}."""
    return _inventory(os.path.abspath(path), os.path.getmtime(path))


def topics_of(inv, types):
    return sorted(t for t, d in inv["topics"].items() if d["type"] in (types if isinstance(types, tuple) else (types,)))


@lru_cache(maxsize=8)
def _tf(path, mtime):
    buf = TFBuffer()
    with reader(path) as r:
        conns = [c for c in r.connections if c.msgtype == TF_TYPE]
        for conn, _, raw in r.messages(connections=conns):
            static = conn.topic.endswith("tf_static")
            for ts in r.deserialize(raw, conn.msgtype).transforms:
                tr, q = ts.transform.translation, ts.transform.rotation
                buf.add(ts.header.frame_id, ts.child_frame_id, stamp(ts.header), (tr.x, tr.y, tr.z),
                        (q.x, q.y, q.z, q.w), static=static)
    return buf


def tf_buffer(path):
    return _tf(os.path.abspath(path), os.path.getmtime(path))


@lru_cache(maxsize=8)
def _image_index(path, mtime, topic):
    """Header stamps of an image topic, in recording order."""
    with reader(path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        out = []
        for conn, t, raw in r.messages(connections=conns):
            out.append(stamp(r.deserialize(raw, conn.msgtype).header))
    return np.array(out)


def image_stamps(path, topic):
    return _image_index(os.path.abspath(path), os.path.getmtime(path), topic)


def decode_image(msg, msgtype):
    """RGB uint8 from a CompressedImage or a raw Image (rgb8 / bgr8 / mono8 / mono16)."""
    if msgtype.endswith("CompressedImage"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot decode a compressed image ({msg.format})")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGB)
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img if enc == "rgb8" else img[:, :, ::-1].copy()
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2RGB)
    if enc in ("mono16", "16uc1"):
        g = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return cv2.cvtColor((g / 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def iter_images(path, topic, indices):
    """(index, stamp, rgb) for the given topic message indices (recording order), in order."""
    want = set(int(i) for i in indices)
    last = max(want) if want else -1
    with reader(path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for k, (conn, _, raw) in enumerate(r.messages(connections=conns)):
            if k > last:
                break
            if k in want:
                m = r.deserialize(raw, conn.msgtype)
                yield k, stamp(m.header), decode_image(m, conn.msgtype)


def camera_info(path, topic):
    """First calibrated CameraInfo: {"width", "height", "model", "params", "distortion_model"} as a COLMAP
    camera. plumb_bob / rational_polynomial -> OPENCV (k1 k2 p1 p2; higher terms dropped), equidistant ->
    OPENCV_FISHEYE (k1..k4), none -> PINHOLE. None if the topic has none: uncalibrated drivers publish
    all zeros (size 0 x 0, K = 0), which is no calibration (the intrinsics are then estimated by SfM)."""
    with reader(path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            K, d, dm = np.asarray(m.k, float).reshape(3, 3), list(np.asarray(m.d, float)), m.distortion_model
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if m.width <= 0 or m.height <= 0 or fx <= 0 or fy <= 0:  # not calibrated: try the next message
                continue
            if dm == "equidistant":
                model, params = "OPENCV_FISHEYE", [fx, fy, cx, cy] + (d + [0.0] * 4)[:4]
            elif dm in ("plumb_bob", "rational_polynomial") and d:
                model, params = "OPENCV", [fx, fy, cx, cy] + (d + [0.0] * 4)[:4]
            else:
                model, params = "PINHOLE", [fx, fy, cx, cy]
            return {"width": int(m.width), "height": int(m.height), "model": model,
                    "params": [float(x) for x in params], "distortion_model": dm,
                    "dropped_terms": [float(x) for x in d[4:]] if dm != "equidistant" else []}
    return None


def odometry(path, topic):
    """(stamps, T_frame_child poses (N, 4, 4), frame_id, child_frame_id)."""
    ts, Ts, frame, child = [], [], None, None
    with reader(path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            ts.append(stamp(m.header))
            Ts.append(make(quat_to_rot((q.x, q.y, q.z, q.w)), (p.x, p.y, p.z)))
            frame, child = m.header.frame_id.lstrip("/"), m.child_frame_id.lstrip("/")
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(Ts)[order], frame, child


_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_xyz(msg):
    """(N, 3) float64 finite xyz (+ intensity or None) from a PointCloud2."""
    fields = {f.name: f for f in msg.fields}
    n = msg.width * msg.height
    names, formats, offsets = [], [], []
    for name in ("x", "y", "z", "intensity"):
        if name in fields:
            f = fields[name]
            names.append(name)
            formats.append(np.dtype(_PF[f.datatype]).newbyteorder(">" if msg.is_bigendian else "<"))
            offsets.append(f.offset)
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("PointCloud2 without x / y / z fields")
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    arr = np.frombuffer(np.asarray(msg.data, np.uint8).tobytes(), dtype=dt, count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    ok = np.isfinite(xyz).all(1)
    inten = arr["intensity"].astype(np.float32)[ok] if "intensity" in names else None
    return xyz[ok], inten


def iter_clouds(path, topic, t0=-np.inf, t1=np.inf, every=1):
    """(stamp, frame_id, xyz, intensity) for messages with t0 <= stamp <= t1, every n-th."""
    with reader(path) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for k, (conn, _, raw) in enumerate(r.messages(connections=conns)):
            if k % every:
                continue
            m = r.deserialize(raw, conn.msgtype)
            t = stamp(m.header)
            if t0 <= t <= t1:
                xyz, inten = cloud_xyz(m)
                yield t, m.header.frame_id.lstrip("/"), xyz, inten

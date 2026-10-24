"""Decoder for the DJI `djmd` ("CAM meta") MP4 data track.

The track holds one protobuf packet per video frame. DJI does not publish the
schema (the header names it `dvtm_<MODEL>.proto`), so packets are parsed
schema-less and fields are mapped by their protobuf path. Field meanings were
reverse engineered on a DJI NEO2 clip and checked on a Mavic 4 Pro clip; see
docs/djmd_format.md. The camera / aircraft / gimbal records are found by the type
in their header (`.1.3` = 1 / 2 / 3), since their positions differ between models
(NEO2: 3.2 / 3.4 / 3.5, Mavic 4: 3.2 / 3.3 / 3.4).

Usage:
    python -m modelforge.dji.djmd data/clip.MP4 -o data/clip_djmd.csv
"""

import argparse
import csv
import json
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, fields


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            if v >= 1 << 63:
                v -= 1 << 64
        elif wt == 1:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        elif wt == 5:
            v = struct.unpack_from("<f", b, i)[0]
            i += 4
        elif wt == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield f, wt, v


def _msg(b):
    """Last-wins dict of a message's fields (repeated fields are not used here)."""
    return {f: v for f, _, v in _fields(b)}


def _packed(b):
    out, i = [], 0
    while i < len(b):
        v, i = _varint(b, i)
        out.append(v)
    return out


def _sub(m, *path):
    """Follow nested submessages; a missing field means proto3 default (empty)."""
    for f in path:
        m = _msg(m.get(f, b"")) if isinstance(m, dict) else _msg(m)
    return m


@dataclass
class FrameMeta:
    frame: int  # 0-based video frame index
    pts: float  # presentation time [s]
    timestamp_us: int  # device clock [us], shared with the aircraft record
    # camera (29.97 Hz)
    iso: float
    shutter_s: float  # exposure time [s]
    fnum: float
    color_temp: int  # [K], SRT `ct`
    # aircraft (10 Hz, repeated on the frames in between)
    lat: float  # WGS84 [deg]
    lon: float  # WGS84 [deg]
    abs_alt_m: float  # SRT `abs_alt`; not plain MSL (negative on the sample clip)
    rel_alt_m: float  # above takeoff
    vel_n: float  # NED velocity [m/s]
    vel_e: float
    vel_d: float  # positive = descending
    ac_pitch_deg: float
    ac_roll_deg: float
    ac_yaw_deg: float  # true heading, matches GPS course
    # gimbal (10 Hz); equals SRT `pp_current` quaternion (w,x,y,z)
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    gimbal_yaw_deg: float  # NEO2 gimbal cannot yaw: this is ac_yaw - ~36.5 deg (offset heading reference)
    gimbal_timestamp_ms: int  # separate clock, not aligned with timestamp_us
    zoom: float = 1.0  # Mavic 4 camera `.23.1` (1.0 on every clip seen; probably the zoom ratio); NEO2: none


CAMERA, AIRCRAFT, GIMBAL = 1, 2, 3  # record types (header field .1.3)
# camera record field numbers per layout: (iso, shutter, f-number, colour temperature)
CAMERA_FIELDS = {"neo2": (3, 4, 14, 6), "mavic4": (9, 10, 11, 24)}


def records(rec):
    """{record type: message} of one packet's field 3 (the first record of each type)."""
    out = {}
    for v in rec.values():
        if not isinstance(v, (bytes, bytearray)):
            continue
        try:
            sub = _msg(v)
        except (ValueError, IndexError, struct.error):
            continue
        hdr = sub.get(1)
        if isinstance(hdr, (bytes, bytearray)):
            t = _msg(hdr).get(3)
            if t in (CAMERA, AIRCRAFT, GIMBAL):
                out.setdefault(t, sub)
    return out


def decode_packet(frame, pts, pkt):
    top = _msg(pkt)
    rec = _msg(top[3])
    r = records(rec)
    cam, ac, gim = r.get(CAMERA, {}), r.get(AIRCRAFT, {}), r.get(GIMBAL, {})
    # NEO2 has ISO in camera field 3; the Mavic 4 moved ISO / shutter / f-number / colour temperature
    f_iso, f_shutter, f_fnum, f_ct = CAMERA_FIELDS["neo2" if 3 in cam or 4 in cam else "mavic4"]

    shutter = _packed(_sub(cam, f_shutter).get(1, b""))
    fnum = _packed(_sub(cam, f_fnum).get(1, b""))
    att = _sub(ac, 3)
    vel = _sub(ac, 2)
    pos = _sub(ac, 4)
    latlon = _sub(pos, 1)
    gatt = _sub(gim, 3) if 3 in gim else _sub(gim, 2)  # gimbal attitude: Mavic 4 field 3, NEO2 field 2

    return FrameMeta(
        frame=frame,
        pts=pts,
        timestamp_us=_sub(rec, 1).get(2, 0),
        iso=_sub(cam, f_iso).get(1, 0.0),
        shutter_s=shutter[0] / shutter[1] if len(shutter) == 2 and shutter[1] else float("nan"),
        fnum=fnum[0] / fnum[1] if len(fnum) == 2 and fnum[1] else float("nan"),
        color_temp=_sub(cam, f_ct).get(1, 0),
        lat=latlon.get(2, float("nan")),
        lon=latlon.get(3, float("nan")),
        abs_alt_m=pos.get(2, 0) / 1000.0,
        rel_alt_m=_sub(ac, 5).get(1, 0.0) / 1000.0,
        vel_n=vel.get(1, 0.0),
        vel_e=vel.get(2, 0.0),
        vel_d=vel.get(3, 0.0),
        ac_pitch_deg=att.get(1, 0) / 10.0,
        ac_roll_deg=att.get(2, 0) / 10.0,
        ac_yaw_deg=att.get(3, 0) / 10.0,
        gimbal_pitch_deg=gatt.get(1, 0) / 10.0,
        gimbal_roll_deg=gatt.get(2, 0) / 10.0,
        gimbal_yaw_deg=gatt.get(3, 0) / 10.0,
        gimbal_timestamp_ms=_sub(gim, 1).get(6, 0),
        zoom=float(_sub(cam, 23).get(1, 1.0)) if f_iso == 9 else 1.0,
    )


def find_djmd_stream(mp4):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_tag_string",
         "-of", "json", mp4], capture_output=True, text=True, check=True).stdout
    for s in json.loads(out)["streams"]:
        if s.get("codec_tag_string") == "djmd":
            return s["index"]
    raise RuntimeError(f"no djmd track in {mp4}")


def read_packets(mp4, stream):
    """Yield (pts, packet_bytes) for one data stream."""
    sizes = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", str(stream),
         "-show_entries", "packet=pts_time,size", "-of", "csv=p=0", mp4],
        capture_output=True, text=True, check=True).stdout.split()
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", mp4, "-map", f"0:{stream}", "-c", "copy", "-f", "data", "-"],
        capture_output=True, check=True).stdout
    off = 0
    for line in sizes:
        pts, n = line.split(",")
        n = int(n)
        yield float(pts), data[off:off + n]
        off += n
    if off != len(data):
        raise RuntimeError(f"packet sizes ({off} B) do not cover stream ({len(data)} B)")


def model_name(mp4):
    """The drone model from the first packet's header (`1.1.10`, e.g. "DJI NEO2", "DJI Mavic4 Pro L3A"),
    else the schema name (`1.1.1`, "dvtm_<model>.proto"), else ""."""
    try:
        stream = find_djmd_stream(mp4)
        sizes = subprocess.run(["ffprobe", "-v", "error", "-select_streams", str(stream), "-read_intervals", "%+#1",
                                "-show_entries", "packet=size", "-of", "csv=p=0", mp4],
                               capture_output=True, text=True, check=True).stdout.split()
        data = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-map", f"0:{stream}", "-frames:d", "1",
                               "-c", "copy", "-f", "data", "-"], capture_output=True, check=True).stdout
        hdr = _msg(_msg(_msg(data[:int(sizes[0])]).get(1, b"")).get(1, b""))
        for f in (10, 1):
            v = hdr.get(f)
            if isinstance(v, (bytes, bytearray)) and v:
                return v.decode("utf-8", "replace")
    except (subprocess.SubprocessError, RuntimeError, ValueError, IndexError, struct.error, KeyError):
        pass
    return ""


def decode(mp4):
    stream = find_djmd_stream(mp4)
    return [decode_packet(i, pts, pkt) for i, (pts, pkt) in enumerate(read_packets(mp4, stream))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("-o", "--output", help="CSV path (default: stdout)")
    args = ap.parse_args()

    frames = decode(args.video)
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    w = csv.DictWriter(out, fieldnames=[f.name for f in fields(FrameMeta)])
    w.writeheader()
    for fm in frames:
        w.writerow(asdict(fm))
    if args.output:
        out.close()
        print(f"wrote {len(frames)} frames to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""djmd decoding across DJI models (NEO2 and Mavic 4 record layouts), the auto camera-heading source and
the windowed optical-flow lead. Packets are built with a minimal protobuf encoder in the layouts seen on
the NEO2 sample and a Mavic 4 Pro clip (docs/djmd_format.md)."""

import struct
from types import SimpleNamespace

import numpy as np

from modelforge.dji import djmd
from modelforge.dji import extract as ex
from modelforge.dji.ingest import Clip


def varint(v):
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def field(f, v):
    """One protobuf field: int -> varint, float -> f32, ('d', x) -> f64, bytes -> length-delimited."""
    if isinstance(v, tuple) and v[0] == "d":
        return varint(f << 3 | 1) + struct.pack("<d", v[1])
    if isinstance(v, float):
        return varint(f << 3 | 5) + struct.pack("<f", v)
    if isinstance(v, int):
        return varint(f << 3 | 0) + varint(v)
    return varint(f << 3 | 2) + varint(len(v)) + v


def msg(*fields):
    return b"".join(field(f, v) for f, v in fields)


def header(kind, hz):
    return msg((1, 1), (2, 1), (3, kind), (4, b"cam"), (5, float(hz)), (6, 1000))


def aircraft():
    return msg((1, header(2, 10)), (2, msg((1, 0.5), (2, -1.5), (3, 0.25))),
               (3, msg((1, -12), (2, 41), (3, -1419))),
               (4, msg((1, msg((2, ("d", 50.1530449)), (3, ("d", 14.1097698)))), (2, 419688))),
               (5, msg((1, 2741.5))))


def packet(layout):
    if layout == "neo2":
        cam = msg((1, header(1, 29.97)), (3, msg((1, 400.0))), (4, msg((1, varint(1) + varint(304)))),
                  (6, msg((1, 5420))), (14, msg((1, varint(22) + varint(10)))))
        gim = msg((1, header(3, 10)), (2, msg((1, -207), (3, -1014))))
        rec = msg((1, msg((2, 123456))), (2, cam), (4, aircraft()), (5, gim))
    else:  # Mavic 4: aircraft at 3.3, gimbal at 3.4 with its attitude in field 3 and a quaternion in 4
        cam = msg((1, header(1, 29.97)), (9, msg((1, 100.0))), (10, msg((1, varint(1) + varint(167)))),
                  (11, msg((1, varint(28) + varint(10)))), (24, msg((1, 5412))))
        gim = msg((1, header(3, 10)), (3, msg((1, -201), (3, -1402))),
                  (4, msg((1, 0.335), (2, -0.164), (3, -0.060), (4, -0.926))))
        rec = msg((1, msg((2, 123456))), (2, cam), (3, aircraft()), (4, gim))
    return msg((3, rec))


def test_decode_neo2_and_mavic4_layouts():
    n = djmd.decode_packet(0, 0.0, packet("neo2"))
    m = djmd.decode_packet(0, 0.0, packet("mavic4"))
    for f in (n, m):  # the aircraft record is the same in both, only its position differs
        assert np.isclose(f.lat, 50.1530449) and np.isclose(f.lon, 14.1097698)
        assert f.abs_alt_m == 419.688 and np.isclose(f.rel_alt_m, 2.7415)
        assert (f.vel_n, f.vel_e, f.vel_d) == (0.5, -1.5, 0.25)
        assert (f.ac_pitch_deg, f.ac_roll_deg, f.ac_yaw_deg) == (-1.2, 4.1, -141.9)
        assert f.timestamp_us == 123456
    assert (n.iso, n.fnum, n.color_temp) == (400.0, 2.2, 5420) and np.isclose(n.shutter_s, 1 / 304)
    assert (n.gimbal_pitch_deg, n.gimbal_roll_deg, n.gimbal_yaw_deg) == (-20.7, 0.0, -101.4)
    assert (m.iso, m.fnum, m.color_temp) == (100.0, 2.8, 5412) and np.isclose(m.shutter_s, 1 / 167)
    assert (m.gimbal_pitch_deg, m.gimbal_roll_deg, m.gimbal_yaw_deg) == (-20.1, 0.0, -140.2)


def _streams(ac_yaw, gimbal_yaw):
    n = len(ac_yaw)
    meta = [SimpleNamespace(lat=50.0, lon=14.0, rel_alt_m=10.0, abs_alt_m=400.0, vel_n=0.0, vel_e=0.0, vel_d=0.0,
                            ac_pitch_deg=0.0, ac_roll_deg=0.0, ac_yaw_deg=float(ac_yaw[k]), gimbal_pitch_deg=-30.0,
                            gimbal_roll_deg=0.0, gimbal_yaw_deg=float(gimbal_yaw[k])) for k in range(n)]
    return ex.build_streams(meta)


def test_auto_yaw_source():
    """auto: by the model when known (NEO: no gimbal yaw axis; Mavic 4: 360 deg gimbal), else the gimbal
    unless its yaw sits over 20 deg from the aircraft yaw (an offset reference, as on the NEO2)."""
    yaw = np.linspace(-170, 170, 500)
    mavic_like = yaw + 5.1 + np.sin(np.arange(500))  # a real small yaw offset (Mavic 4 spodek clips: 5.1 deg)
    neo_like = yaw - 36.5  # aircraft yaw in an offset reference
    for model, gimbal_yaw, expect in (("DJI NEO2", mavic_like, "aircraft"), ("DJI Mavic4 Pro L3A", neo_like, "gimbal"),
                                      ("", mavic_like, "gimbal"), ("", neo_like, "aircraft"),
                                      ("DJI Unknown 9", neo_like, "aircraft")):
        a, g = _streams(yaw, (gimbal_yaw + 180) % 360 - 180)
        clip = SimpleNamespace(n=500, aircraft=a, gimbal=g, model=model)
        assert Clip.yaw_source(clip, "auto") == expect, (model, expect)
        assert Clip.yaw_source(clip, "aircraft") == "aircraft" and Clip.yaw_source(clip, "gimbal") == "gimbal"


def test_flow_windows():
    assert ex.flow_windows(5000, 29.97) == [(0, 5000)]  # short: the whole clip
    w = ex.flow_windows(59025, 29.97)  # the 33 min Mavic clip: 4 windows of 45 s spread over it
    m = int(round(45 * 29.97))
    assert len(w) == 4 and all(b - a == m for a, b in w) and w[0][0] == 0 and w[-1][1] == 59025

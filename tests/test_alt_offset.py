"""The automatic altitude offset between flights (dji.ingest.takeoff_offset): the takeoff altitudes' difference,
refused when the takeoffs are close but abs_alt says tens of metres apart (karlak3: 240 m low, 10 m away)."""
import numpy as np

from modelforge.dji import ingest


class FakeClip:
    def __init__(self, takeoff_abs_alt, lat, lon, start_alt=0.5):
        self.takeoff_abs_alt = takeoff_abs_alt
        a = np.zeros((50, 10))
        a[:, 0], a[:, 1] = lat, lon
        a[:, 2] = np.linspace(start_alt, 30.0, 50)  # rel_alt: from (near) the ground up
        self._a = a

    def telemetry(self, yaw_source="auto"):
        return {"a": self._a}


LAT, LON = 50.1, 14.1
M_LAT = 1 / 111_200  # ~1 m in latitude


def test_takeoff_offset():
    ref = FakeClip(331.8, LAT, LON)
    assert ingest.takeoff_offset(ref, ref) == (0.0, None)
    alt, note = ingest.takeoff_offset(FakeClip(333.5, LAT + 38 * M_LAT, LON), ref)  # karlak_2: kept
    assert abs(alt - 1.7) < 1e-9 and note is None
    alt, note = ingest.takeoff_offset(FakeClip(91.8, LAT + 10 * M_LAT, LON), ref)  # karlak3: refused
    assert alt == 0.0 and "implausible" in note
    alt, note = ingest.takeoff_offset(FakeClip(91.8, LAT + 2000 * M_LAT, LON), ref)  # far away: a real valley
    assert abs(alt + 240.0) < 1e-9 and note is None
    alt, note = ingest.takeoff_offset(FakeClip(91.8, LAT + 10 * M_LAT, LON, start_alt=40), ref)  # starts in the air
    assert abs(alt + 240.0) < 1e-9 and note is None
    alt, note = ingest.takeoff_offset(FakeClip(None, LAT, LON), ref)
    assert alt == 0.0 and "no usable abs_alt" in note

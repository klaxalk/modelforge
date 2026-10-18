"""Clips without an SRT: one UTC clock for ordering, and the optical-flow lead cached on disk."""

import datetime
import os
import subprocess

from modelforge.dji import extract as ex
from modelforge.dji import ingest


def make_clip(path, creation_time=None):
    meta = ["-metadata", f"creation_time={creation_time}"] if creation_time else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                    "-pix_fmt", "yuv420p", *meta, path], check=True)


def utc(s):
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()


def test_wall_start_is_utc_with_or_without_srt(tmp_path):
    clip = str(tmp_path / "a.mp4")
    make_clip(clip, "2026-07-15T18:20:43Z")
    srt = [{"datetime": "2026-07-15 20:20:43.396"}]  # SRT: local time (UTC+2 here), ms resolution
    assert abs(ingest._wall_start(srt, clip) - utc("2026-07-15 18:20:43.396")) < 1e-3  # zone removed, ms kept
    assert ingest._wall_start(None, clip) == utc("2026-07-15 18:20:43")  # no SRT: MP4 tag (UTC)
    bare = str(tmp_path / "b.mp4")
    make_clip(bare)
    assert ingest._wall_start(srt, bare) == utc("2026-07-15 20:20:43.396")  # nothing to compare: SRT as is
    assert ingest._wall_start(None, bare) is None


def test_flow_lead_is_cached_on_disk(tmp_path, monkeypatch):
    clip = str(tmp_path / "c.mp4")
    make_clip(clip)
    calls = []

    def fake(video, gimbal, n, pts=None):
        calls.append(video)
        return {"method": "optical_flow", "lead_frames": 21.5, "corr": 0.55}

    monkeypatch.setattr(ex, "estimate_lead_flow", fake)
    a = ingest._cached_flow_lead(clip, None, 10)
    b = ingest._cached_flow_lead(clip, None, 10)
    assert a == b == {"method": "optical_flow", "lead_frames": 21.5, "corr": 0.55} and len(calls) == 1
    os.rename(clip, str(tmp_path / "renamed.mp4"))  # size + mtime are the key: survives renames
    assert ingest._cached_flow_lead(str(tmp_path / "renamed.mp4"), None, 10)["lead_frames"] == 21.5
    assert len(calls) == 1
    other = str(tmp_path / "d.mp4")
    make_clip(other, "2026-01-01T00:00:00Z")  # a failed estimate is remembered too
    monkeypatch.setattr(ex, "estimate_lead_flow", lambda *a, **k: calls.append("d") or None)
    assert ingest._cached_flow_lead(other, None, 10) is None
    assert ingest._cached_flow_lead(other, None, 10) is None and calls.count("d") == 1

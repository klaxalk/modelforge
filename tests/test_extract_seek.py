"""Extraction of a range late in a clip seeks instead of decoding from the start (extract.decode_from): the same
frames, with the same indices, as decoding everything (trim by frame index)."""
import os
import subprocess

import numpy as np
import pytest

from modelforge.dji import extract as ex
from tests.sample_clip import sample_video

VIDEO = sample_video()
pytestmark = pytest.mark.skipif(not os.path.exists(VIDEO), reason="sample clip not found")


def _frames(cmd, w, h, n=3):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = [np.frombuffer(p.stdout.read(w * h * 3), np.uint8) for _ in range(n)]
    p.kill()
    p.wait()
    return out


@pytest.mark.parametrize("k0", [5, 4321, 9000])
def test_seek_gives_the_same_frames(k0):
    w, h = ex.video_size(VIDEO)
    ref = _frames(["ffmpeg", "-v", "error", "-i", VIDEO, "-map", "0:v:0", "-fps_mode", "passthrough",
                   "-vf", f"trim=start_frame={k0}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], w, h)
    for pts in (None, ex.frame_times(VIDEO)):  # probed from the packets / given (DJI: the djmd times)
        cmd = ex.decode_from(VIDEO, k0, pts)
        assert ("-ss" in cmd) == (k0 >= ex.SEEK_FROM)
        assert all(np.array_equal(a, b) for a, b in zip(ref, _frames(cmd, w, h)))


def test_stopping_early_ends_ffmpeg():
    """An abandoned extraction (Stop) must not leave its decoder running."""
    gen = ex.select_sharpest(VIDEO, [(9000, 9002), (9100, 9102)], 0)
    next(gen)
    gen.close()
    left = subprocess.run(["pgrep", "-f", f"ffmpeg .*-i {VIDEO}"], capture_output=True, text=True).stdout
    assert not left.strip()

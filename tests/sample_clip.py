"""Where the sample clip (berlin_3_1.MP4) lives: $MODELFORGE_TEST_VIDEO, else anywhere in data/
(the Videos tab can move it into a folder)."""

import glob
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def sample_video(name="berlin_3_1.MP4"):
    env = os.environ.get("MODELFORGE_TEST_VIDEO")
    if env:
        return env
    hits = sorted(glob.glob(os.path.join(DATA, "**", name), recursive=True))
    return hits[0] if hits else os.path.join(DATA, name)

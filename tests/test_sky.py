"""Sky masks for splat training (modelforge/sky.py): segmented once per photo at the training size, kept for the next
run; the training flags (test_rosbag_kind.test_splat_lidar_command)."""
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from modelforge import sky

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "transforms.json")) or sky.available() is not None,
                    reason="needs a project with transforms.json and nerfstudio's venv with transformers")
def test_masks_at_the_training_size(tmp_path):
    t = json.load(open(os.path.join(REAL, "transforms.json")))
    t["frames"] = t["frames"][:3]
    p = tmp_path / "proj"
    p.mkdir()
    (p / "images").mkdir()
    for f in t["frames"]:  # copies: the masks go next to the images' folder (a tile's links lead to its project)
        shutil.copy2(os.path.join(REAL, f["file_path"]), p / f["file_path"])
    json.dump(t, open(p / "transforms.json", "w"))
    tile = tmp_path / "tile"
    tile.mkdir()
    os.symlink(p / "images", tile / "images")
    json.dump(t, open(tile / "transforms.json", "w"))
    lines = []
    out = sky.masks(str(tile), 2, on_line=lines.append)
    assert os.path.realpath(out) == os.path.realpath(p / "sky" / "masks_x2")
    for f in t["frames"]:
        name = os.path.basename(f["file_path"])
        w, h = Image.open(p / f["file_path"]).size
        m = np.asarray(Image.open(os.path.join(out, name + ".png")))
        assert m.shape == (h // 2, w // 2) and set(np.unique(m)) <= {0, 255}
    lines.clear()
    sky.masks(str(p), 2, on_line=lines.append)
    assert any("made already" in line for line in lines)

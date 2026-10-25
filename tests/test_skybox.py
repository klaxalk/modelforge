"""Sky boxes for the 3D view (modelforge/gui/skybox.py): the sun's place in the sky and in the panorama."""
import datetime

import numpy as np

from modelforge.gui import skybox


def test_sun_position():
    # Berlin, 2026-06-21 11:07 UTC (~local solar noon): due south, 90 - 52.5 + 23.4 = 60.9° up
    t = datetime.datetime(2026, 6, 21, 11, 7, tzinfo=datetime.timezone.utc).timestamp()
    az, el = skybox.sun_position(52.52, 13.40, t)
    assert abs(az - 180) < 3 and abs(el - 60.9) < 1
    # evening: west-north-west, low
    az, el = skybox.sun_position(52.52, 13.40, t + 8 * 3600)
    assert 290 < az < 320 and el < 10


def test_panorama_layout():
    """Column u = azimuth u x 360° clockwise from north (viser's z-up root in three.js); row 0 = zenith."""
    for kind in ("sunny", "cloudy", "overcast"):
        img = skybox.render(kind, sun_az=90, sun_el=30, width=256)
        assert img.shape == (128, 256, 3) and 0 <= img.min() and img.max() <= 1
        assert img[:60].mean() > img[70:].mean()  # sky above, darker ground below
    img = skybox.render("sunny", sun_az=90, sun_el=30, width=512)
    r, c = np.unravel_index(np.argmax(img.sum(-1)[:128]), (128, 512))
    assert abs(c - 512 / 4) <= 3 and abs(r - (90 - 30) / 180 * 256) <= 3
    assert skybox.jpeg("cloudy", 90, 30, 256)[:2] == b"\xff\xd8"

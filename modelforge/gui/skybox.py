"""Sky boxes for the 3D view (View → Sky box; the user asked, 2026-10-25: splats trained with sky masks leave the sky
empty, and they are meant to be seen in front of a sky).

Procedural equirectangular panoramas (numpy, no assets): sunny (clear blue, sun disc and glow), cloudy (scattered
clouds on a flat cloud layer, so they shrink towards the horizon), overcast (grey, brighter at the zenith, soft
darker patches). Below the horizon a hazy ground colour, for views past the edge of the scene.

The sun stands where it stood during the flight when that is known (the ENU origin's latitude / longitude and the
first video's UTC creation time + the middle photo's time), else south at 40°; the view can still turn the sky.

viser shows it as its environment map's background (a plain JPEG: the client's gain-map loader falls back to SDR).
Column u of the image is the azimuth u x 360° clockwise from north: viser's +z-up root maps world (x, y, z) to
three.js (-y, z, -x), and three.js samples equirect maps at u = atan2(dir.z, dir.x) / 2π + 0.5.
"""
import datetime
import functools
import io
import json
import math
import os

import numpy as np

from modelforge import sun

KINDS = ("off", "sunny", "cloudy", "overcast")
WIDTH = 2048  # px of the panorama (height half of it)
DEFAULT_SUN = (180.0, 40.0)  # azimuth (° from north, clockwise), elevation (°) without a known flight time
MIN_ELEVATION = 6.0  # a sun lower than this is drawn at it (dusk flights would get a sky half below the horizon)


def project_sun(project):
    """(azimuth, elevation, note) of the sun during PROJECT's flight, or the default with a note why not."""
    try:
        p = json.load(open(os.path.join(project, "poses.json")))
        lat, lon = p["enu_origin_lla"][:2]
        video = p["sources"][0]["video"]
        if not os.path.isabs(video):  # stored relative to the working directory or the project
            video = next((c for c in (video, os.path.join(project, video)) if os.path.exists(c)), video)
        from modelforge.dji import ingest
        start = ingest._mp4_creation_time(video)
        if start is None:
            raise ValueError("no creation time")
        times = sorted(im["time_s"] for im in p["images"] if im.get("time_s") is not None)
        utc = start + (times[len(times) // 2] if times else 0.0)
    except Exception:  # noqa: BLE001 - no poses, no video, a video kind: the default sun
        return (*DEFAULT_SUN, "sun: south at 40° (flight time unknown)")
    az, el = sun.position(lat, lon, utc)
    when = datetime.datetime.fromtimestamp(utc, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = f"sun as during the flight ({when}): azimuth {az:.0f}°, elevation {el:.0f}°"
    if el < MIN_ELEVATION:
        note += f", drawn at {MIN_ELEVATION:.0f}°"
    return az, max(el, MIN_ELEVATION), note


def _noise(shape_xy, x, y, octaves, seed, gain=0.5):
    """fBm of periodic value noise (smoothstep-interpolated 256 x 256 lattices) at float coordinates x, y."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape_xy, np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        lat = rng.random((256, 256), dtype=np.float32)
        f = 2.0 ** o
        xs, ys = x * f + 17.3 * o, y * f + 31.7 * o
        x0, y0 = np.floor(xs), np.floor(ys)
        tx, ty = xs - x0, ys - y0
        tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
        i0, j0 = x0.astype(np.int64) % 256, y0.astype(np.int64) % 256
        i1, j1 = (i0 + 1) % 256, (j0 + 1) % 256
        a = lat[j0, i0] + (lat[j0, i1] - lat[j0, i0]) * tx
        b = lat[j1, i0] + (lat[j1, i1] - lat[j1, i0]) * tx
        out += amp * (a + (b - a) * ty)
        norm += amp
        amp *= gain
    return out / norm


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def render(kind, sun_az=DEFAULT_SUN[0], sun_el=DEFAULT_SUN[1], width=WIDTH):
    """The panorama as float32 sRGB in [0, 1], H x W x 3 (row 0 = zenith, column 0 = north, east at W / 4)."""
    h = width // 2
    az = (np.arange(width, dtype=np.float32) + 0.5) / width * 2 * np.pi
    el = np.pi / 2 - (np.arange(h, dtype=np.float32) + 0.5) / h * np.pi
    AZ, EL = np.meshgrid(az, el)
    ce = np.cos(EL)
    d = np.stack([ce * np.sin(AZ), ce * np.cos(AZ), np.sin(EL)], -1)  # east, north, up
    sa, se = math.radians(sun_az), math.radians(sun_el)
    sun = np.array([math.cos(se) * math.sin(sa), math.cos(se) * math.cos(sa), math.sin(se)], np.float32)
    gamma = np.arccos(np.clip(d @ sun, -1, 1))  # angle to the sun
    up = np.clip(d[..., 2], 0, 1)
    # cloud-layer coordinates: where the view ray meets a flat layer (kept finite near the horizon)
    k = 1.0 / np.maximum(up, 0.04)
    px, py = d[..., 0] * k, d[..., 1] * k
    horizon_fade = _smooth(0.015, 0.12, up)  # clouds melt into the haze near the horizon

    if kind == "overcast":
        lum = (1 + 2 * up) / 3  # CIE overcast sky: the zenith 3x the horizon
        base = np.array([0.80, 0.81, 0.83], np.float32)
        sky = base * (0.74 + 0.26 * lum)[..., None]
        n = _noise(up.shape, px * 1.3, py * 1.3, 6, seed=3)
        patches = (n - 0.5) * horizon_fade
        sky = sky * (1 + 0.38 * patches)[..., None]
        sky = sky + (0.06 * np.exp(-gamma / 0.6))[..., None]  # the sun's place, a brighter smudge
        horizon = np.array([0.82, 0.83, 0.84], np.float32)
        sky = sky + (horizon - sky) * (1 - _smooth(0.0, 0.08, up))[..., None]
    else:
        zenith = np.array([0.16, 0.38, 0.76], np.float32)
        horizon = np.array([0.72, 0.82, 0.92], np.float32)
        t = (1 - up) ** 3
        sky = zenith + (horizon - zenith) * t[..., None]
        glow = 0.45 * np.exp(-gamma / 0.10) + 0.20 * np.exp(-gamma / 0.45)
        warm = np.array([1.0, 0.96, 0.86], np.float32)
        sky = sky + (warm - sky) * np.clip(glow, 0, 1)[..., None]
        disc = _smooth(math.radians(0.75), math.radians(0.45), gamma)
        if kind == "cloudy":
            n = _noise(up.shape, px * 0.9, py * 0.9, 7, seed=7)
            dens = _smooth(0.50, 0.70, n) * horizon_fade
            core = _smooth(0.55, 0.85, n)  # thicker parts are darker underneath
            lit = np.array([0.97, 0.97, 0.96], np.float32)
            shade = np.array([0.62, 0.65, 0.71], np.float32)
            cloud = lit + (shade - lit) * (core * 0.8)[..., None]
            cloud = cloud + (warm - cloud) * np.clip(0.5 * np.exp(-gamma / 0.25), 0, 1)[..., None]  # silver lining
            sky = sky + (cloud - sky) * dens[..., None]
            disc = disc * (1 - dens)
        sky = sky + (1 - sky) * disc[..., None]
    # below the horizon: haze into a muted ground
    down = np.clip(-d[..., 2], 0, 1)
    ground = np.array([0.42, 0.41, 0.38], np.float32) if kind != "overcast" else np.array([0.45, 0.45, 0.44],
                                                                                            np.float32)
    g = (1 - np.exp(-down / 0.06))[..., None]
    sky = np.where((d[..., 2] < 0)[..., None], horizon + (ground - horizon) * g, sky)
    return np.clip(sky, 0, 1)


@functools.lru_cache(maxsize=8)
def jpeg(kind, sun_az=DEFAULT_SUN[0], sun_el=DEFAULT_SUN[1], width=WIDTH):
    """render() as JPEG bytes (the sun's angles rounded by the caller keep the cache useful)."""
    from PIL import Image
    img = Image.fromarray((render(kind, sun_az, sun_el, width) * 255 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def show(server, kind, sun_az=DEFAULT_SUN[0], sun_el=DEFAULT_SUN[1], turn_deg=0.0):
    """Set viser's background to the sky box (every browser, also those connecting later), or back to viser's
    default lighting without a background for "off". `turn_deg` turns the sky about the vertical (counter-clockwise
    seen from above)."""
    from viser import _messages
    if kind not in KINDS or kind == "off":
        server.scene.configure_environment_map("city", background=False)  # viser's default lighting
        return
    data = jpeg(kind, round(sun_az, 1) % 360, round(sun_el, 1))
    a = math.radians(turn_deg) / 2
    wxyz = (math.cos(a), 0.0, 0.0, math.sin(a))
    server.scene._websock_interface.queue_message(_messages.EnvironmentMapMessage(
        hdri_data=data, background=True, background_blurriness=0.0, background_intensity=1.0,
        background_wxyz=wxyz, environment_intensity=1.0, environment_wxyz=wxyz))

"""Geodesy for the map (gui/map.py) and the orthophoto's GeoTIFF (modelforge/ortho.py): the project's local ENU frame
(origin enu_origin_lla) to and from WGS84, Web Mercator tile pixels (OpenStreetMap / Esri tiles) and UTM (the
GeoTIFF's projection). Vectorized over points."""

import math

import numpy as np

from modelforge.dji.extract import _A, _E2, lla_to_ecef

TILE = 256  # px per Web Mercator tile


def _r_enu(origin):
    lat, lon = np.radians(origin[0]), np.radians(origin[1])
    return np.array([[-np.sin(lon), np.cos(lon), 0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def lla_to_enu(lat, lon, alt, origin):
    """WGS84 (degrees, m) -> ENU (n, 3) about origin (lat, lon, alt)."""
    p = lla_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float))
    return (_r_enu(origin) @ (p.reshape(3, -1) - lla_to_ecef(*origin)[:, None])).T


def enu_to_lla(xyz, origin):
    """ENU (n, 3) or (n, 2) (height 0) -> (lat, lon, alt) arrays (degrees, m); Bowring's method (< 1 mm)."""
    xyz = np.asarray(xyz, float).reshape(-1, np.shape(xyz)[-1])
    if xyz.shape[1] == 2:
        xyz = np.c_[xyz, np.zeros(len(xyz))]
    x, y, z = (_r_enu(origin).T @ xyz.T) + lla_to_ecef(*origin)[:, None]
    b = _A * math.sqrt(1 - _E2)
    ep2 = (_A * _A - b * b) / (b * b)
    p = np.hypot(x, y)
    th = np.arctan2(z * _A, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - _E2 * _A * np.cos(th) ** 3)
    lon = np.arctan2(y, x)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def mercator_px(lat, lon, zoom):
    """Global Web Mercator pixel coordinates (x right, y down) at a zoom level."""
    n = TILE * 2.0 ** zoom
    lat = np.clip(np.asarray(lat, float), -85.05112878, 85.05112878)
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    s = np.sin(np.radians(lat))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def mercator_mpp(lat, zoom):
    """Ground metres per Web Mercator pixel at a latitude and zoom."""
    return 156543.03392804097 * math.cos(math.radians(lat)) / 2.0 ** zoom


# --- UTM (Krueger series, as in the Transverse Mercator of most GIS libraries; mm accuracy within the zone) ---

def utm_zone(lat, lon):
    """(zone number, north, EPSG code) of a WGS84 point (no Norway / Svalbard exceptions)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    north = lat >= 0
    return zone, north, (32600 if north else 32700) + zone


def lla_to_utm(lat, lon, zone, north=True):
    """WGS84 degrees -> UTM easting / northing (m) in a given zone."""
    f = 1 - math.sqrt(1 - _E2)
    n = f / (2 - f)
    big_a = _A / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n * n / 3 + 5 * n ** 3 / 16, 13 * n * n / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    k0, lon0 = 0.9996, math.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)) - lon0
    e = math.sqrt(_E2)
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    E = eta + sum(a * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, a in enumerate(alpha, 1))
    N = xi + sum(a * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, a in enumerate(alpha, 1))
    return 500000.0 + k0 * big_a * E, (0.0 if north else 10000000.0) + k0 * big_a * N

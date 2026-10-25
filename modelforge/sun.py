"""The sun's position in the sky (the sky box, modelforge/lighting.py)."""
import datetime
import math


def position(lat, lon, utc):
    """Sun azimuth (° from north, clockwise) and elevation (°) at latitude / longitude (°) and POSIX time `utc`
    (NOAA's low-precision formulas, within ~0.5°)."""
    t = datetime.datetime.fromtimestamp(utc, datetime.timezone.utc)
    day = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))  # minutes
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)  # hour angle
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    el = math.asin(max(-1.0, min(1.0, cz)))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))
    return (math.degrees(az) + 180) % 360, math.degrees(el)

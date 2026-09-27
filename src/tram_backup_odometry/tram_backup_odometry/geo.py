"""WGS-84: геодезические координаты ↔ ECEF ↔ локальная ENU (east, north, up), как в pymap3d."""
from __future__ import annotations

import math

import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
B = A * (1.0 - F)
EP2 = (A * A - B * B) / (B * B)


def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    sl, cl = np.sin(lat), np.cos(lat)
    n = A / np.sqrt(1.0 - E2 * sl * sl)
    return (n + h) * cl * np.cos(lon), (n + h) * cl * np.sin(lon), (n * (1.0 - E2) + h) * sl


def ecef_to_geodetic(x, y, z):
    """Замкнутая формула Боуринга; точность ~мм у поверхности Земли."""
    p = np.hypot(x, y)
    th = np.arctan2(z * A, p * B)
    lat = np.arctan2(z + EP2 * B * np.sin(th) ** 3, p - E2 * A * np.cos(th) ** 3)
    lon = np.arctan2(y, x)
    sl = np.sin(lat)
    n = A / np.sqrt(1.0 - E2 * sl * sl)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


class EnuFrame:
    """Локальная касательная плоскость с началом в (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self.x0, self.y0, self.z0 = geodetic_to_ecef(self.lat0, self.lon0, self.h0)
        la, lo = math.radians(self.lat0), math.radians(self.lon0)
        sla, cla, slo, clo = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
        # строки: e, n, u
        self.R = np.array([[-slo, clo, 0.0],
                           [-sla * clo, -sla * slo, cla],
                           [cla * clo, cla * slo, sla]])

    def to_enu(self, lat, lon, h):
        x, y, z = geodetic_to_ecef(lat, lon, h)
        d = np.stack([np.asarray(x) - self.x0, np.asarray(y) - self.y0, np.asarray(z) - self.z0])
        e, n, u = np.tensordot(self.R, d, axes=1)
        return e, n, u

    def to_geodetic(self, e, n, u):
        d = np.tensordot(self.R.T, np.stack([np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)]), axes=1)
        return ecef_to_geodetic(d[0] + self.x0, d[1] + self.y0, d[2] + self.z0)


# ---------------------------------------------------------------------------- UTM / MGRS (плоские координаты судьи)
_N = F / (2.0 - F)
_A1 = A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5, 56 * _N ** 3 / 15)
K0 = 0.9996


def utm_forward(lat, lon, zone: int):
    """WGS-84 → UTM (северное полушарие), ряды Крюгера (точность ~мм в пределах зоны)."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    lon0 = math.radians(6.0 * zone - 183.0)
    e = math.sqrt(E2)
    t = np.sinh(np.arctanh(np.sin(lat)) - e * np.arctanh(e * np.sin(lat)))
    dl = lon - lon0
    xi = np.arctan2(t, np.cos(dl))
    eta = np.arctanh(np.sin(dl) / np.sqrt(1.0 + t * t))
    x, y = eta.copy(), xi.copy()
    for j, a in enumerate(_ALPHA, 1):
        x = x + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y = y + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + K0 * _A1 * x, K0 * _A1 * y


def utm_inverse(east, north, zone: int):
    xi = np.asarray(north, float) / (K0 * _A1)
    eta = (np.asarray(east, float) - 500000.0) / (K0 * _A1)
    xp, ep = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, 1):
        xp = xp - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        ep = ep - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xp) / np.cosh(ep))
    lat = chi.copy()
    for j, d in enumerate(_DELTA, 1):
        lat = lat + d * np.sin(2 * j * chi)
    lon = math.radians(6.0 * zone - 183.0) + np.arctan2(np.sinh(ep), np.cos(xp))
    return np.degrees(lat), np.degrees(lon)


class MgrsFrame:
    """Плоские координаты судьи: UTM зоны `zone` минус угол 100-км квадрата MGRS (по умолчанию 37U CB:
    x = E − 300 000, y = N − 6 100 000, без переноса через границу квадратов); z = эллипсоидальная высота.
    Интерфейс как у EnuFrame: to_enu → (x, y, z), to_geodetic → (lat, lon, h)."""

    def __init__(self, zone: int = 37, east0: float = 300000.0, north0: float = 6100000.0):
        self.zone, self.east0, self.north0 = int(zone), float(east0), float(north0)

    def to_enu(self, lat, lon, h):
        e, n = utm_forward(lat, lon, self.zone)
        return e - self.east0, n - self.north0, np.asarray(h, float) + 0.0

    def to_geodetic(self, x, y, z):
        lat, lon = utm_inverse(np.asarray(x, float) + self.east0, np.asarray(y, float) + self.north0, self.zone)
        return lat, lon, np.asarray(z, float) + 0.0

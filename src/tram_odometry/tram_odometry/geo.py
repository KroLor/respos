"""Перевод геодезических координат WGS84 в локальную метрическую систему ENU.

ENU (East-North-Up): x — на восток, y — на север, z — вверх, в метрах,
начало — в заданной точке (обычно первая точка GNSS прогона).
Путь расчёта: широта/долгота/высота → геоцентрические ECEF → поворот в ENU.
"""
import math

WGS84_A = 6378137.0                      # большая полуось эллипсоида, м
WGS84_F = 1.0 / 298.257223563            # сжатие
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # квадрат первого эксцентриситета


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt: float):
    """Широта, долгота (градусы) и высота над эллипсоидом (м) → ECEF (x, y, z), м."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    # Радиус кривизны первого вертикала
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * math.cos(lon)
    y = (n + alt) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


class LocalEnu:
    """Локальная система ENU с началом в точке (lat0, lon0, alt0)."""

    def __init__(self, lat0_deg: float, lon0_deg: float, alt0: float) -> None:
        self.origin = (lat0_deg, lon0_deg, alt0)
        self._x0, self._y0, self._z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0)
        lat0 = math.radians(lat0_deg)
        lon0 = math.radians(lon0_deg)
        self._sin_lat0 = math.sin(lat0)
        self._cos_lat0 = math.cos(lat0)
        self._sin_lon0 = math.sin(lon0)
        self._cos_lon0 = math.cos(lon0)

    def to_enu(self, lat_deg: float, lon_deg: float, alt: float):
        """Геодезические координаты → (east, north, up), м относительно начала."""
        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt)
        dx = x - self._x0
        dy = y - self._y0
        dz = z - self._z0
        east = -self._sin_lon0 * dx + self._cos_lon0 * dy
        north = (-self._sin_lat0 * self._cos_lon0 * dx
                 - self._sin_lat0 * self._sin_lon0 * dy
                 + self._cos_lat0 * dz)
        up = (self._cos_lat0 * self._cos_lon0 * dx
              + self._cos_lat0 * self._sin_lon0 * dy
              + self._sin_lat0 * dz)
        return east, north, up

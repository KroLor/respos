"""Карта маршрута: замкнутое кольцо в порядке движения и положение на нём по пройденному пути.

Кольцо (config/route_map.csv, строится tools/build_route_map.py из GNSS обучающих прогонов):
разворотная петля у одной конечной → путь туда → петля у другой → путь обратно. Трамвай не
ездит задним ходом, поэтому положение = начальная точка на кольце u₀ + пройденный путь s.
Модуль не зависит от ROS.
"""
import bisect
import math
from typing import List, Optional, Tuple

from tram_odometry.geo import LocalEnu


def load_route_csv(path: str) -> List[Tuple[float, float, float]]:
    """Точки кольца (широта, долгота, высота) в порядке движения; строки '#' — комментарии."""
    points = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('u,'):
                continue
            _, lat, lon, alt = (float(v) for v in line.split(','))
            points.append((lat, lon, alt))
    if len(points) < 10:
        raise ValueError(f'в карте {path} слишком мало точек: {len(points)}')
    return points


def angle_diff(a: float, b: float) -> float:
    """Разность углов в диапазоне [-π, π]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class LocalRoute:
    """Кольцо в локальной системе ENU прогона (начало — первая точка GNSS)."""

    def __init__(self, geodetic_points, enu: LocalEnu) -> None:
        pts = [enu.to_enu(lat, lon, alt) for lat, lon, alt in geodetic_points]
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.z = [p[2] for p in pts]
        n = len(pts)
        # Сегмент i — от точки i к точке i+1 (последний замыкает кольцо)
        self.u = [0.0] * n
        self.seg_len = [0.0] * n
        self.seg_yaw = [0.0] * n
        for i in range(n):
            j = (i + 1) % n
            dx, dy = self.x[j] - self.x[i], self.y[j] - self.y[i]
            self.seg_len[i] = math.hypot(dx, dy)
            self.seg_yaw[i] = math.atan2(dy, dx)
            if i + 1 < n:
                self.u[i + 1] = self.u[i] + self.seg_len[i]
        self.length = self.u[-1] + self.seg_len[-1]

    def wrap(self, u: float) -> float:
        return u % self.length

    def delta(self, u_to: float, u_from: float) -> float:
        """Кратчайшая разность координат на кольце, в диапазоне [-L/2, L/2]."""
        half = self.length / 2.0
        return (u_to - u_from + half) % self.length - half

    def pose(self, u: float) -> Tuple[float, float, float, float]:
        """Координата вдоль кольца → (x, y, z, курс)."""
        u = self.wrap(u)
        i = max(0, bisect.bisect_right(self.u, u) - 1)
        j = (i + 1) % len(self.x)
        t = (u - self.u[i]) / self.seg_len[i] if self.seg_len[i] > 0 else 0.0
        return (self.x[i] + t * (self.x[j] - self.x[i]),
                self.y[i] + t * (self.y[j] - self.y[i]),
                self.z[i] + t * (self.z[j] - self.z[i]),
                self.seg_yaw[i])

    def project(self, x: float, y: float, heading: Optional[float] = None,
                max_heading_error: float = math.pi / 3,
                near_u: Optional[float] = None, window: Optional[float] = None
                ) -> Optional[Tuple[float, float]]:
        """Ближайшая точка кольца → (u, расстояние до кольца) или None.

        heading — сегменты с курсом, отличающимся больше чем на max_heading_error, пропускаются
        (так не путаются пути встречных направлений); near_u и window — искать только
        в окне ±window вокруг ожидаемого места (непрерывность движения вдоль кольца).
        """
        n = len(self.x)
        if near_u is not None and window is not None and 2 * window < self.length:
            i0 = max(0, bisect.bisect_right(self.u, self.wrap(near_u - window)) - 1)
            count = int(2 * window / max(min(self.seg_len), 0.1)) + 2
            indices = ((i0 + k) % n for k in range(min(count, n)))
        else:
            indices = range(n)
        best = None
        for i in indices:
            if heading is not None and abs(angle_diff(self.seg_yaw[i], heading)) > max_heading_error:
                continue
            j = (i + 1) % n
            ax, ay = self.x[i], self.y[i]
            vx, vy = self.x[j] - ax, self.y[j] - ay
            ll = vx * vx + vy * vy
            t = 0.0 if ll == 0.0 else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / ll))
            d = math.hypot(ax + t * vx - x, ay + t * vy - y)
            if best is None or d < best[1]:
                best = (self.wrap(self.u[i] + t * self.seg_len[i]), d)
        return best

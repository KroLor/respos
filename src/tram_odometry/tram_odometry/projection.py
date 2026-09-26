"""Перевод пройденного пути в координаты x, y, z и курс.

Этап 1: путь откладывается по прямой из начала координат с постоянным курсом
(относительная одометрия от стартовой точки — допускается ТЗ).
Этап 3 заменит это начальной выставкой по GNSS и движением по карте маршрута.
"""
import math


class StraightLineProjector:
    """Путь по прямой из точки origin с курсом yaw (рад, 0 — вдоль оси x)."""

    def __init__(self, origin=(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        self.origin = origin
        self.yaw = yaw

    def project(self, distance: float):
        """Пройденный путь, м → (x, y, z, yaw)."""
        x0, y0, z0 = self.origin
        return (x0 + distance * math.cos(self.yaw),
                y0 + distance * math.sin(self.yaw),
                z0,
                self.yaw)

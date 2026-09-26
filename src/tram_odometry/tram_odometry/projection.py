"""Перевод пройденного пути в координаты x, y, z, курс и их неопределённость.

Этап 1: путь откладывается по прямой из начала координат с постоянным курсом
(относительная одометрия от стартовой точки — допускается ТЗ). Курс при этом неизвестен,
поэтому неопределённость поперёк пути и по курсу честно большая.
Карта маршрута (следующий шаг) заменит это движением вдоль известного пути с малыми
поперечной ошибкой и ошибкой курса — интерфейс project() тот же.
"""
import math
from dataclasses import dataclass


@dataclass
class ProjectedPose:
    """Положение на момент оценки и неопределённость, которую вносит перевод пути в координаты."""

    x: float
    y: float
    z: float
    yaw: float          # курс, рад (0 — вдоль оси x, против часовой стрелки)
    cross_var: float    # дисперсия поперёк пути, м²
    z_var: float        # дисперсия высоты, м²
    yaw_var: float      # дисперсия курса, рад²


# Курс неизвестен: равномерное распределение на окружности, дисперсия π²/3
UNKNOWN_YAW_VARIANCE = math.pi ** 2 / 3.0
# Высота по маршруту неизвестна: перепады в пределах ~10 м
UNKNOWN_Z_VARIANCE = 10.0 ** 2
# Начальная неопределённость положения поперёк пути, м
CROSS_TRACK_SIGMA0 = 1.0


class StraightLineProjector:
    """Путь по прямой из точки origin с курсом yaw (рад, 0 — вдоль оси x).

    distance0 — пройденный путь в момент, когда трамвай был в origin.
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), yaw: float = 0.0, distance0: float = 0.0) -> None:
        self.origin = origin
        self.yaw = yaw
        self.distance0 = distance0

    def project(self, distance: float) -> ProjectedPose:
        x0, y0, z0 = self.origin
        travelled = distance - self.distance0
        # Курс неизвестен и путь может поворачивать: поперёк пути трамвай может оказаться
        # в пределах пройденного расстояния
        cross_sigma = CROSS_TRACK_SIGMA0 + abs(travelled)
        return ProjectedPose(
            x=x0 + travelled * math.cos(self.yaw),
            y=y0 + travelled * math.sin(self.yaw),
            z=z0,
            yaw=self.yaw,
            cross_var=cross_sigma ** 2,
            z_var=UNKNOWN_Z_VARIANCE,
            yaw_var=UNKNOWN_YAW_VARIANCE,
        )

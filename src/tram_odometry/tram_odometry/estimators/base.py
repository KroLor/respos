"""Общий интерфейс оценщиков скорости и положения.

Оценщик — обычный Python-класс без зависимостей от ROS. Нода передаёт ему
уже проверенные входные измерения (методы on_*) и запрашивает оценку на нужный
момент времени (estimate). Все метки времени — секунды по времени bag
(header.stamp), скорости — м/с (перевод из км/ч выполняет предобработка).
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Estimate:
    """Оценка состояния трамвая на момент времени запроса."""

    velocity: float                   # продольная скорость, м/с
    distance: float                   # пройденный от старта путь, м
    acceleration: float = 0.0         # продольное ускорение, м/с²
    velocity_var: float = 0.0         # дисперсия скорости, (м/с)²
    distance_var: float = 0.0         # дисперсия пути, м²
    # Координаты x, y, z (м) и курс (рад, 0 — на восток, против часовой стрелки),
    # если оценщик знает их сам; иначе нода получает их из пройденного пути
    position: Optional[Tuple[float, float, float]] = None
    yaw: Optional[float] = None
    slip_detected: bool = False       # признаки проскальзывания/юза или сбоя датчика колёс
    slip_ratio: Optional[float] = None  # оценка продольного скольжения, если оценщик её даёт


class Estimator:
    """Базовый класс оценщика: входные измерения и оценка на момент времени."""

    name = 'base'
    # True — оценщику нужны GNSS-топики; в основном контуре это запрещено правилами
    requires_gnss = False

    def __init__(self, params: dict) -> None:
        self.params = params

    def on_wheel(self, bogie: str, stamp: float, velocity: float,
                 suspicious: bool = False) -> None:
        """Скорость тележки bogie ('front' или 'rear'), м/с.

        suspicious — предобработка заметила скачок быстрее физически возможного
        или расхождение тележек (признак проскальзывания или сбоя датчика).
        """

    def on_driver_cmd(self, stamp: float, position: int) -> None:
        """Положение ручки контроллера: 0 — нейтраль, +1..+15 — тяга, -1..-15 — торможение."""

    def on_gnss_fix(self, source: str, stamp: float,
                    latitude: float, longitude: float, altitude: float) -> None:
        """Позиция GNSS-приёмника source ('master' или 'rover'), WGS84."""

    def on_gnss_vel(self, source: str, stamp: float,
                    vx: float, vy: float, vz: float) -> None:
        """Скорость GNSS-приёмника source, м/с (twist.linear)."""

    def on_time_reset(self, stamp: float) -> None:
        """Отсчёт времени во входных данных сменился назад (например, bag запущен заново).

        Нужно сбросить всё, что привязано к прежним меткам времени; пройденный путь
        сохраняется.
        """

    def estimate(self, stamp: float) -> Estimate:
        """Оценка состояния на момент stamp с учётом всех полученных измерений."""
        raise NotImplementedError

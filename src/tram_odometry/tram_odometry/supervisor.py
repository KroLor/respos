"""Надзор за оценщиком: запасной оценщик, проверка результата, непрерывность пути.

Основной оценщик (например, модель движения) может выбросить исключение или вернуть
нечисловой либо нефизичный результат. Тогда результат берётся у запасного оценщика
(wheel_baseline), который всё время работает параллельно на тех же данных, поэтому
переключение мгновенное. Когда основной снова выдаёт корректный результат, он
возвращается. Пройденный путь при переключении не скачет: к пути нового источника
добавляется смещение. Если отказали все оценщики, скорость удерживается, а путь
продолжает считаться по ней. Модуль не зависит от ROS.
"""
import math
import numbers
from collections import Counter
from dataclasses import replace
from typing import Optional

from tram_odometry.estimators.base import Estimate, Estimator

HOLD = 'hold'
# Отрицательная скорость до этого значения (м/с) — шум, обнуляется; ниже — сбой
NEGATIVE_SPEED_TOLERANCE = 0.5


class SupervisedEstimator:

    def __init__(self, primary: Estimator, fallback: Optional[Estimator] = None,
                 max_speed: float = 30.0) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_speed = max_speed
        self.requires_gnss = primary.requires_gnss
        self.active = primary.name       # чей результат сейчас публикуется
        self.failures = Counter()        # оценщик → число сбоев
        self.switches = 0
        self.last_error = ''
        self._offsets = {}               # оценщик → смещение пути
        self._last_stamp = None
        self._last = None                # последний выданный результат
        self._previous = {}              # оценщик → путь на прошлом шаге (если был результат)

    @property
    def estimators(self):
        return [e for e in (self.primary, self.fallback) if e is not None]

    # --- Входные данные: передаются всем оценщикам, сбой одного не мешает другим ---

    def on_wheel(self, bogie, stamp, velocity, suspicious=False):
        self._broadcast('on_wheel', bogie, stamp, velocity, suspicious)

    def on_driver_cmd(self, stamp, position):
        self._broadcast('on_driver_cmd', stamp, position)

    def on_gnss_fix(self, source, stamp, latitude, longitude, altitude):
        self._broadcast('on_gnss_fix', source, stamp, latitude, longitude, altitude)

    def on_gnss_vel(self, source, stamp, vx, vy, vz):
        self._broadcast('on_gnss_vel', source, stamp, vx, vy, vz)

    def on_time_reset(self, stamp):
        self._broadcast('on_time_reset', stamp)
        self._last_stamp = None

    def _broadcast(self, method, *args):
        for estimator in self.estimators:
            try:
                getattr(estimator, method)(*args)
            except Exception as error:
                self._fail(estimator, f'{method}: {error!r}')

    # --- Оценка ---

    def estimate(self, stamp: float) -> Estimate:
        # Вызываем все оценщики на каждом шаге, иначе запасной не накапливает путь
        results = {}
        for estimator in self.estimators:
            try:
                result = estimator.estimate(stamp)
            except Exception as error:
                self._fail(estimator, f'estimate: {error!r}')
                continue
            checked = self._validate(result)
            if checked is None:
                self._fail(estimator, f'estimate: некорректный результат {result!r}')
                continue
            results[estimator.name] = checked
        previous, self._previous = self._previous, {n: r.distance for n, r in results.items()}
        for estimator in self.estimators:
            if estimator.name in results:
                return self._emit(estimator.name, stamp, results[estimator.name], previous)
        return self._hold(stamp)

    def _validate(self, result) -> Optional[Estimate]:
        """Корректный результат (отрицательный шум скорости обнулён) или None."""
        if not isinstance(result, Estimate):
            return None
        values = [result.velocity, result.distance, result.acceleration,
                  result.velocity_var, result.distance_var]
        if result.position is not None:
            values.extend(result.position)
        if not all(isinstance(v, numbers.Real) and math.isfinite(v) for v in values):
            return None
        if not -NEGATIVE_SPEED_TOLERANCE <= result.velocity <= self.max_speed:
            return None
        if result.velocity_var < 0.0 or result.distance_var < 0.0:
            return None
        # Некорректная оценка скольжения — не сбой всей оценки: считаем, что её нет
        slip_ratio = result.slip_ratio
        if slip_ratio is not None and not (isinstance(slip_ratio, numbers.Real)
                                           and math.isfinite(slip_ratio)):
            slip_ratio = None
        return replace(result, velocity=max(0.0, float(result.velocity)), slip_ratio=slip_ratio)

    def _emit(self, name: str, stamp: float, result: Estimate, previous: dict) -> Estimate:
        if name != self.active:
            # Переключение источника: путь продолжается от последнего выданного значения
            # на приращение нового источника за шаг (или по последней скорости, если
            # на прошлом шаге новый источник сам не дал результата)
            if self._last is not None:
                if name in previous:
                    step = result.distance - previous[name]
                else:
                    dt = stamp - self._last_stamp if self._last_stamp is not None else 0.0
                    step = self._last.velocity * max(0.0, dt)
                self._offsets[name] = self._last.distance + step - result.distance
            self.active = name
            self.switches += 1
        output = replace(result, distance=result.distance + self._offsets.get(name, 0.0))
        self._last_stamp, self._last = stamp, output
        return output

    def _hold(self, stamp: float) -> Estimate:
        """Все оценщики отказали: удерживаем скорость, путь считаем по ней."""
        if self.active != HOLD:
            self.active = HOLD
            self.switches += 1
        if self._last is None:
            output = Estimate(velocity=0.0, distance=0.0, velocity_var=1.0, distance_var=1.0)
        else:
            dt = stamp - self._last_stamp if self._last_stamp is not None else 0.0
            dt = max(0.0, dt)
            output = replace(self._last,
                             distance=self._last.distance + self._last.velocity * dt,
                             velocity_var=self._last.velocity_var + dt,
                             distance_var=self._last.distance_var + (self._last.velocity * dt) ** 2)
        self._last_stamp, self._last = stamp, output
        return output

    def _fail(self, estimator: Estimator, message: str) -> None:
        self.failures[estimator.name] += 1
        self.last_error = f'{estimator.name}.{message}'[:300]

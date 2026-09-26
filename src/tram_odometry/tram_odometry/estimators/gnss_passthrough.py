"""ТЕСТОВЫЙ оценщик: возвращает данные GNSS как результат.

Нужен только для проверки конвейера (нода, топики, метки времени, запись,
метрики): выход совпадает с эталоном, поэтому ошибка должна быть около нуля.
Использование GNSS в основном контуре запрещено правилами — не для сдачи.
"""
import math

from tram_odometry.estimators.base import Estimate, Estimator
from tram_odometry.geo import LocalEnu

# Rover используется, только если master молчит дольше этого времени, с
MASTER_SILENCE_SEC = 2.0
# Ниже этой скорости курс не определяется и путь не накапливается, м/с
MIN_MOVING_SPEED = 0.5


class GnssPassthroughEstimator(Estimator):

    name = 'gnss_passthrough'
    requires_gnss = True

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._first_stamp = None      # первое сообщение GNSS любого приёмника
        self._master_stamp = None     # последнее сообщение master
        self._enu = None              # локальная система с началом в первой точке
        self._position = None
        self._velocity = 0.0
        self._yaw = None
        self._distance = 0.0

    def _accept(self, source: str, stamp: float) -> bool:
        """Держимся приёмника master; rover — только если master молчит."""
        if self._first_stamp is None:
            self._first_stamp = stamp
        if source == 'master':
            self._master_stamp = stamp
            return True
        silent_since = self._master_stamp if self._master_stamp is not None else self._first_stamp
        return stamp - silent_since > MASTER_SILENCE_SEC

    def on_gnss_fix(self, source, stamp, latitude, longitude, altitude):
        if not self._accept(source, stamp):
            return
        if self._enu is None:
            self._enu = LocalEnu(latitude, longitude, altitude)
        position = self._enu.to_enu(latitude, longitude, altitude)
        if self._position is not None and self._velocity > MIN_MOVING_SPEED:
            self._distance += math.dist(position[:2], self._position[:2])
        self._position = position

    def on_gnss_vel(self, source, stamp, vx, vy, vz):
        if not self._accept(source, stamp):
            return
        self._velocity = math.hypot(vx, vy)
        if self._velocity > MIN_MOVING_SPEED:
            self._yaw = math.atan2(vy, vx)

    def estimate(self, stamp: float) -> Estimate:
        return Estimate(velocity=self._velocity, distance=self._distance,
                        velocity_var=0.01, distance_var=0.25,
                        position=self._position, yaw=self._yaw)

"""Ядро резервной одометрии без ROS: входы → предобработка → оценщик → результат для публикации.

Нода (node.py) только переводит сообщения ROS в вызовы ядра, а результат ядра — в сообщения.
Тот же код используют модульные тесты и офлайн-оценка по bag, поэтому проверяется ровно то,
что работает у жюри.

Правила публикации:
- основной триггер — сообщение контроллера (~20 Гц); колёса (~9,4 Гц) — только если контроллер
  молчит дольше driver_cmd_timeout (по времени bag);
- метка результата — метка входного сообщения-триггера; метки результатов строго растут;
- до первого измерения колёс скорость неизвестна: публикация ждёт его, но не дольше
  wheel_wait_at_start (при неработающих датчиках колёс она всё равно начнётся);
- смена отсчёта времени во входах назад — метки результатов начинаются заново.
"""
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from tram_odometry.estimators import ESTIMATORS, Estimate, create_estimator
from tram_odometry.preprocessing import InputPreprocessor, PreprocessingParams
from tram_odometry.projection import StraightLineProjector
from tram_odometry.supervisor import SupervisedEstimator

DEFAULT_ESTIMATOR = 'wheel_baseline'
VEHICLE_INPUTS = ('front', 'rear', 'driver_cmd')
GNSS_INPUTS = ('gnss_fix_master', 'gnss_vel_master', 'gnss_fix_rover', 'gnss_vel_rover')

# Дисперсии величин, которые пока не оцениваются
UNKNOWN_VARIANCE = 1e3
# Трамвай на рельсах не движется вбок и вертикально относительно корпуса
CONSTRAINED_VARIANCE = 1e-4


@dataclass
class CoreParams:
    """Параметры ядра; имена совпадают с параметрами ROS в config/params.yaml."""

    estimator: str = DEFAULT_ESTIMATOR
    driver_cmd_timeout: float = 0.15        # с: контроллер молчит дольше — триггер по колёсам
    max_output_speed: float = 30.0          # м/с: выше — сбой оценщика
    bogie_mismatch_max_duration: float = 3.0  # с: дольше — отказ датчика
    wheel_hold_max: float = 2.0             # с: удержание скорости без данных колёс
    wheel_decay_tau: float = 5.0            # с: затем снижение с этой постоянной времени
    wheel_wait_at_start: float = 1.0        # с: ожидание первого измерения колёс (не параметр ROS)


@dataclass
class Output:
    """Результат на момент stamp — то, что публикует нода."""

    stamp: float
    estimate: Estimate
    x: float
    y: float
    z: float
    yaw: float
    pose_covariance: List[float]    # 6x6 построчно: x, y, z, крен, тангаж, курс
    twist_covariance: List[float]   # 6x6 построчно: vx, vy, vz, wx, wy, wz


def diagonal_covariance(diagonal) -> List[float]:
    """Ковариационная матрица 6x6 (построчно) с заданной диагональю."""
    covariance = [0.0] * 36
    for i, value in enumerate(diagonal):
        covariance[i * 7] = float(value)
    return covariance


class OdometryCore:

    def __init__(self, pre_params: Optional[PreprocessingParams] = None,
                 params: Optional[CoreParams] = None,
                 log: Optional[Callable[[str, str], None]] = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.p = params or CoreParams()
        self.pre = InputPreprocessor(pre_params or PreprocessingParams())
        self._log = log or (lambda level, text: None)
        self._clock = clock
        self.estimator = self._create_supervisor()
        self.projector = StraightLineProjector()
        self.inputs = list(VEHICLE_INPUTS)
        if self.estimator.requires_gnss:
            self.inputs += GNSS_INPUTS
        self.arrival = {}               # вход → время (clock) последнего принятого сообщения
        self.last_output = None
        self.published = 0
        self.skipped = 0                # результаты, пропущенные из-за немонотонной метки
        self.time_resets = 0
        self._last_cmd_stamp = None
        self._last_pub_stamp = None
        self._wheel_seen = False
        self._first_trigger_stamp = None

    def _create_supervisor(self) -> SupervisedEstimator:
        """Основной оценщик + запасной wheel_baseline; ошибка создания не роняет ядро."""
        pre = self.pre.p
        params = {
            'wheel_timeout': pre.wheel_timeout,
            'bogie_mismatch': pre.bogie_mismatch_kmh * pre.wheel_speed_scale,
            'plausible_accel': pre.suspicious_accel,
            'plausible_margin': pre.suspicious_margin_kmh * pre.wheel_speed_scale,
            'bogie_mismatch_max_duration': self.p.bogie_mismatch_max_duration,
            'wheel_hold_max': self.p.wheel_hold_max,
            'wheel_decay_tau': self.p.wheel_decay_tau,
        }
        name = self.p.estimator
        if name not in ESTIMATORS:
            self._log('error', f'Неизвестный оценщик "{name}", доступны: {", ".join(ESTIMATORS)}. '
                               f'Использую {DEFAULT_ESTIMATOR}.')
            name = DEFAULT_ESTIMATOR
        try:
            primary = create_estimator(name, params)
        except Exception as error:
            self._log('error', f'Не удалось создать оценщик {name}: {error!r}. '
                               f'Использую {DEFAULT_ESTIMATOR}.')
            name = DEFAULT_ESTIMATOR
            primary = create_estimator(name, params)
        fallback = None if name == DEFAULT_ESTIMATOR else create_estimator(DEFAULT_ESTIMATOR, params)
        return SupervisedEstimator(primary, fallback, self.p.max_output_speed)

    # --- Входные данные ---

    def on_wheel(self, bogie: str, stamp: float, speed_kmh: float) -> Optional[Output]:
        sample = self.pre.wheel(bogie, stamp, speed_kmh)
        if sample is None:
            return None
        self.arrival[bogie] = self._clock()
        self._wheel_seen = True
        if sample.time_reset:
            self._reset_time(sample.stamp)
        self.estimator.on_wheel(bogie, sample.stamp, sample.value, sample.suspicious)
        if self._last_cmd_stamp is None or sample.stamp - self._last_cmd_stamp > self.p.driver_cmd_timeout:
            return self._publish(sample.stamp)
        return None

    def on_driver_cmd(self, stamp: float, position: int) -> Optional[Output]:
        sample = self.pre.driver_cmd(stamp, position)
        if sample is None:
            return None
        self.arrival['driver_cmd'] = self._clock()
        if sample.time_reset:
            self._reset_time(sample.stamp)
        self._last_cmd_stamp = sample.stamp
        self.estimator.on_driver_cmd(sample.stamp, sample.value)
        return self._publish(sample.stamp)

    def on_gnss_fix(self, source: str, stamp: float, status: int,
                    latitude: float, longitude: float, altitude: float) -> None:
        sample = self.pre.gnss_fix(source, stamp, status, latitude, longitude, altitude)
        if sample is not None:
            self.arrival[f'gnss_fix_{source}'] = self._clock()
            self.estimator.on_gnss_fix(source, sample.stamp, *sample.value)

    def on_gnss_vel(self, source: str, stamp: float, vx: float, vy: float, vz: float) -> None:
        sample = self.pre.gnss_vel(source, stamp, vx, vy, vz)
        if sample is not None:
            self.arrival[f'gnss_vel_{source}'] = self._clock()
            self.estimator.on_gnss_vel(source, sample.stamp, *sample.value)

    def _reset_time(self, stamp: float) -> None:
        """Отсчёт времени во входных данных сменился назад: метки результатов начинаются заново."""
        self.time_resets += 1
        self._last_pub_stamp = None
        self._last_cmd_stamp = None
        self.estimator.on_time_reset(stamp)
        self._log('warn', f'Отсчёт времени во входных данных сменился назад (метка {stamp:.3f})')

    # --- Результат ---

    def _publish(self, stamp: float) -> Optional[Output]:
        # Метки результатов строго растут: судья сопоставляет их с эталоном по времени
        if self._last_pub_stamp is not None and stamp <= self._last_pub_stamp:
            self.skipped += 1
            return None
        if not self._wheel_seen:
            if self._first_trigger_stamp is None:
                self._first_trigger_stamp = stamp
            if stamp - self._first_trigger_stamp < self.p.wheel_wait_at_start:
                return None
        estimate = self.estimator.estimate(stamp)
        if estimate.position is not None:
            x, y, z = estimate.position
            yaw = estimate.yaw if estimate.yaw is not None else 0.0
        else:
            x, y, z, yaw = self.projector.project(estimate.distance)
        output = Output(
            stamp=stamp, estimate=estimate, x=x, y=y, z=z, yaw=yaw,
            # Ориентация пока не оценивается
            pose_covariance=diagonal_covariance([
                estimate.distance_var, estimate.distance_var, estimate.distance_var,
                UNKNOWN_VARIANCE, UNKNOWN_VARIANCE, UNKNOWN_VARIANCE]),
            # Скорость — продольная, в системе трамвая
            twist_covariance=diagonal_covariance([
                estimate.velocity_var, CONSTRAINED_VARIANCE, CONSTRAINED_VARIANCE,
                UNKNOWN_VARIANCE, UNKNOWN_VARIANCE, UNKNOWN_VARIANCE]),
        )
        self._last_pub_stamp = stamp
        self.last_output = output
        self.published += 1
        return output

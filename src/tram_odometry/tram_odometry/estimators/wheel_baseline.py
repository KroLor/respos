"""Базовый оценщик: скорость — по свежим измерениям тележек, путь — интеграл скорости.

Модели привода нет. Нужен как запасной вариант (работает параллельно с основным
оценщиком) и как точка отсчёта для сравнения. Простые защиты:
- тележки расходятся (буксование, юз, всплеск или отказ датчика), выставляется флаг
  проскальзывания. Правила проверены на всех прогонах с GNSS (11 эпизодов, все < 1,7 с);
  применяются по порядку:
  * одна тележка стоит (< 1 км/ч), а скорость другой правдоподобна — отказ датчика
    или заблокированное колесо: берётся движущаяся (даже при малом расхождении —
    иначе нулевой датчик тянет среднее вниз при трогании);
  * расхождение дольше bogie_mismatch_max_duration — отказ датчика, а не
    проскальзывание: берётся большая скорость;
  * правдоподобна только одна тележка — её значение отличается от прогноза
    «прошлая скорость + ускорение × шаг» не больше, чем позволяет физически возможное
    ускорение: берётся она (отсекает всплески, провалы, юз и буксование);
  * одна тележка показывает почти ноль, а другая нет — отказ датчика или
    заблокированное колесо: берётся большая скорость (в данных 4 из 4);
  * иначе — тележка, чьё значение ближе к прогнозу: буксующее колесо разгоняется,
    а колесо на юзе тормозит быстрее трамвая (в данных при тяге ближе к истине
    меньшая — 48 из 48, при торможении большая — 35 из 51);
- нет свежих данных колёс: скорость удерживается wheel_hold_max секунд, затем плавно
  снижается, чтобы путь не «убегал» при долгом отказе датчиков.
Дополнительно оценивается продольное скольжение тележек относительно итоговой скорости.
"""
import math

from tram_odometry.estimators.base import Estimate, Estimator

# Постоянная времени сглаживания ускорения, с
ACCELERATION_FILTER_TAU = 0.5
# Скорость тележки ниже этой (м/с, 1 км/ч) при движении другой — отказ датчика или блокировка колеса
STOPPED_WHEEL_SPEED = 1.0 / 3.6
# Скольжение считается относительно скорости трамвая, но не меньше этой (м/с) — у остановки
# деление на почти ноль превратило бы шум датчика в огромное «скольжение»
MIN_SLIP_REFERENCE_SPEED = 1.0
# Неопределённости, откалиброванные по 86 прогонам с GNSS: RMSE скорости выше 1 м/с —
# 0,09–0,11 м/с и почти не зависит от скорости; ошибка пути — 0,65 м в первые 100 м,
# дрейф в конце прогона — медиана 0,19 %, p90 0,88 % пути
VELOCITY_SIGMA = 0.10               # м/с
VELOCITY_SIGMA_SLIP = 0.5           # м/с — добавка при расхождении тележек
VELOCITY_SIGMA_PER_SILENCE = 0.5    # м/с за секунду без данных колёс
DISTANCE_SIGMA0 = 0.5               # м
DISTANCE_SIGMA_PER_METER = 0.006    # доля пройденного пути


class WheelBaselineEstimator(Estimator):

    name = 'wheel_baseline'

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._wheel_timeout = float(params.get('wheel_timeout', 0.5))       # с
        self._mismatch = float(params.get('bogie_mismatch', 3.0 / 3.6))     # м/с
        self._mismatch_max = float(params.get('bogie_mismatch_max_duration', 3.0))  # с
        # Физически возможное отклонение от прогноза: ускорение × шаг + запас на шум
        self._plausible_accel = float(params.get('plausible_accel', 3.0))            # м/с²
        self._plausible_margin = float(params.get('plausible_margin', 1.0 / 3.6))    # м/с
        self._hold_max = float(params.get('wheel_hold_max', 2.0))           # с
        self._decay_tau = float(params.get('wheel_decay_tau', 5.0))         # с
        self._wheels = {}           # тележка -> (метка времени, скорость м/с)
        self._mismatch_since = None  # начало текущего расхождения тележек
        self._stamp = None          # время последней оценки
        self._data_stamp = None     # время последней оценки со свежими данными колёс
        self._velocity = 0.0
        self._acceleration = 0.0
        self._distance = 0.0

    def on_wheel(self, bogie, stamp, velocity, suspicious=False):
        self._wheels[bogie] = (stamp, velocity)

    def on_time_reset(self, stamp):
        self._wheels.clear()
        self._stamp = None
        self._data_stamp = None
        self._mismatch_since = None

    def _fresh(self, stamp):
        """Скорости тележек, измеренные не раньше wheel_timeout до stamp."""
        return [v for t, v in self._wheels.values() if stamp - t <= self._wheel_timeout]

    def _measured_velocity(self, stamp, dt, fresh):
        """(скорость по свежим измерениям или None, тележки расходятся)."""
        if not fresh:
            return None, False
        low, high = min(fresh), max(fresh)
        mismatch = high - low > self._mismatch
        if not mismatch:
            self._mismatch_since = None
        elif self._mismatch_since is None:
            self._mismatch_since = stamp
        predicted = self._velocity + self._acceleration * dt
        allowed = self._plausible_accel * dt + self._plausible_margin

        # Одна тележка стоит, другая правдоподобно едет — отказ датчика или блокировка колеса.
        # Проверяется до порога расхождения, иначе нулевой датчик тянет среднее вниз при трогании
        if len(fresh) == 2 and low < STOPPED_WHEEL_SPEED <= high \
                and abs(high - predicted) <= allowed:
            return high, mismatch
        if not mismatch:
            return (low + high) / 2.0, False
        if stamp - self._mismatch_since > self._mismatch_max:
            return high, True
        plausible = [v for v in fresh if abs(v - predicted) <= allowed]
        if len(plausible) == 1:
            return plausible[0], True
        if low < STOPPED_WHEEL_SPEED:
            return high, True
        return min(fresh, key=lambda v: abs(v - predicted)), True

    def estimate(self, stamp):
        dt = stamp - self._stamp if self._stamp is not None and stamp > self._stamp else 0.0
        fresh = self._fresh(stamp)
        measured, slip = self._measured_velocity(stamp, dt, fresh)
        silence = 0.0
        if measured is not None:
            velocity = measured
            self._data_stamp = stamp
        else:
            velocity = self._velocity
            if self._data_stamp is not None:
                silence = stamp - self._data_stamp
                if silence > self._hold_max:
                    velocity *= math.exp(-dt / self._decay_tau)

        if dt > 0.0:
            # Путь — интеграл скорости методом трапеций; ускорение — сглаженная производная
            self._distance += 0.5 * (self._velocity + velocity) * dt
            raw_acceleration = (velocity - self._velocity) / dt
            self._acceleration += dt / (ACCELERATION_FILTER_TAU + dt) * (
                raw_acceleration - self._acceleration)
        if dt > 0.0 or self._stamp is None:
            self._stamp = stamp
        self._velocity = velocity

        # Продольное скольжение s = (v_колеса − v_трамвая) / v_трамвая: > 0 — буксование,
        # < 0 — юз; из тележек берётся наибольшее по модулю
        slip_ratio = None
        if fresh:
            reference = max(velocity, MIN_SLIP_REFERENCE_SPEED)
            slip_ratio = max((v - velocity for v in fresh), key=abs) / reference

        # Неопределённость скорости растёт при проскальзывании и без данных колёс;
        # неопределённость пути — с пройденным путём
        sigma_v = (VELOCITY_SIGMA + (VELOCITY_SIGMA_SLIP if slip else 0.0)
                   + VELOCITY_SIGMA_PER_SILENCE * silence)
        distance_var = DISTANCE_SIGMA0 ** 2 + (DISTANCE_SIGMA_PER_METER * self._distance) ** 2
        return Estimate(velocity=velocity, distance=self._distance,
                        acceleration=self._acceleration,
                        velocity_var=sigma_v ** 2, distance_var=distance_var,
                        slip_detected=slip, slip_ratio=slip_ratio)

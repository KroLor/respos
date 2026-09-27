"""Приём и предобработка входных данных: проверка значений и меток времени, флаги, статистика.

Правила выведены из анализа всех 122 прогонов датасета:
- скорость колёс приходит в км/ч; около нуля бывает шум до -0,4 км/ч — такие значения
  обнуляются, заметно отрицательные — отбрасываются;
- в некоторых прогонах идут два перемешанных потока, метки которых различаются на ~1 с
  (часть сообщений записана с задержкой): для каждого топика берутся только сообщения
  с растущей меткой, то есть самые свежие данные;
- метка, убежавшая вперёд больше чем на stamp_max_lead и от своей предыдущей, и от других
  входов, считается мусорной; несколько согласованных таких меток подряд — новый отсчёт
  времени (например, bag запущен заново), на него выполняется переход;
- в норме скорость колеса меняется не быстрее ~3 м/с², а тележки расходятся не больше
  ~3 км/ч (так ведут себя 99,9 % измерений); остальное помечается подозрительным
  (признак проскальзывания, юза или сбоя датчика), но не отбрасывается — решает оценщик.
Модуль не зависит от ROS.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

WHEELS = ('front', 'rear')
# Входы, по которым определяется текущее время bag
VEHICLE_TOPICS = ('front', 'rear', 'driver_cmd')

# Результат проверки метки времени
STAMP_REJECTED = 'rejected'
STAMP_OK = 'ok'
STAMP_RESYNC_FORWARD = 'resync_forward'
STAMP_RESYNC_BACKWARD = 'resync_backward'


# Масштаб скорости колёс (км/ч → м/с) по трамваям: калибровка по координатам GNSS
# на надёжных участках (скорость > 10 км/ч, тележки согласны, без скачков GNSS).
# Путь колёс / путь GNSS: 30618 — 3,5940 (60 прогонов, 300 км, квартили 3,594–3,598),
# 30639 — 3,6095 (18 прогонов, 85 км; между днями менялся от 3,63 до 3,59).
# Передняя и задняя тележки совпадают в пределах 0,01 %.
VEHICLE_WHEEL_SCALES = {
    '30618': 1.0 / 3.5940,
    '30639': 1.0 / 3.6095,
}
NOMINAL_WHEEL_SCALE = 1.0 / 3.6


@dataclass
class PreprocessingParams:
    vehicle_id: str = '30618'                  # трамвай: откалиброванный масштаб колёс
    wheel_speed_scale: float = 0.0             # км/ч → м/с; 0 — масштаб для vehicle_id
    wheel_speed_max_kmh: float = 120.0         # выше — заведомо неверное значение
    wheel_negative_tolerance_kmh: float = 1.0  # от -этого до 0 — шум, обнуляется
    suspicious_accel: float = 3.0              # м/с²: изменение скорости быстрее — подозрительно
    suspicious_margin_kmh: float = 1.0         # запас на шум к порогу по ускорению
    bogie_mismatch_kmh: float = 3.0            # расхождение тележек больше — подозрительно
    wheel_timeout: float = 0.5                 # с: измерение старше считается устаревшим
    stamp_max_lead: float = 5.0                # с: допустимый скачок метки вперёд
    stamp_max_lag: float = 5.0                 # с: откат назад больше — возможная смена отсчёта
    stamp_resync_count: int = 5                # согласованных меток подряд для смены отсчёта
    gap_threshold: float = 1.0                 # с: пропуск длиннее учитывается в статистике
    driver_position_limit: int = 15            # |положение ручки| не больше этого


@dataclass
class Sample:
    """Принятое измерение."""

    stamp: float
    value: Any                   # м/с для колёс, int для контроллера, кортеж для GNSS
    suspicious: bool = False
    time_reset: bool = False     # отсчёт времени этого входа сменился назад


class TopicState:
    """Состояние и статистика одного входа."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_stamp = None
        self.last_value = None
        self.accepted = 0
        self.rejected = Counter()     # причина → количество
        self.suspicious = 0
        self.clamped = 0              # отрицательный шум, обнулённый
        self.resyncs = 0              # переходы на новый отсчёт времени
        self.gaps = 0                 # пропуски длиннее gap_threshold
        self.max_gap = 0.0
        self.chain = []               # подряд отброшенные, но согласованные метки


class InputPreprocessor:

    def __init__(self, params: PreprocessingParams) -> None:
        self.p = params
        # Масштаб колёс: заданный вручную, иначе откалиброванный для трамвая, иначе номинальный
        self.known_vehicle = params.vehicle_id in VEHICLE_WHEEL_SCALES
        if params.wheel_speed_scale > 0.0:
            self.wheel_scale = params.wheel_speed_scale
        else:
            self.wheel_scale = VEHICLE_WHEEL_SCALES.get(params.vehicle_id, NOMINAL_WHEEL_SCALE)
        self.topics = {}

    def state(self, name: str) -> TopicState:
        if name not in self.topics:
            self.topics[name] = TopicState(name)
        return self.topics[name]

    def current_stamp(self) -> Optional[float]:
        """Текущее время bag — самая поздняя принятая метка входов трамвая."""
        stamps = [self.topics[n].last_stamp for n in VEHICLE_TOPICS
                  if n in self.topics and self.topics[n].last_stamp is not None]
        return max(stamps) if stamps else None

    # --- Входы ---

    def wheel(self, bogie: str, stamp: float, speed_kmh: float) -> Optional[Sample]:
        """Скорость тележки в км/ч → Sample со скоростью в м/с или None, если отброшено."""
        p = self.p
        state = self.state(bogie)
        if not math.isfinite(speed_kmh):
            return self._reject(state, 'value_invalid')
        if speed_kmh > p.wheel_speed_max_kmh or speed_kmh < -p.wheel_negative_tolerance_kmh:
            return self._reject(state, 'value_out_of_range')
        prev_stamp, prev_value = state.last_stamp, state.last_value
        result = self._check_stamp(state, stamp)
        if result == STAMP_REJECTED:
            return None
        if speed_kmh < 0.0:
            speed_kmh = 0.0
            state.clamped += 1

        suspicious = False
        # Скачок быстрее физически возможного (после смены отсчёта сравнивать не с чем)
        if result == STAMP_OK and prev_stamp is not None:
            allowed = p.suspicious_accel * 3.6 * (stamp - prev_stamp) + p.suspicious_margin_kmh
            suspicious = abs(speed_kmh - prev_value) > allowed
        # Расхождение со свежим измерением другой тележки
        other = self.topics.get('rear' if bogie == 'front' else 'front')
        if (other is not None and other.last_stamp is not None
                and abs(stamp - other.last_stamp) <= p.wheel_timeout
                and abs(speed_kmh - other.last_value) > p.bogie_mismatch_kmh):
            suspicious = True

        state.last_value = speed_kmh
        if suspicious:
            state.suspicious += 1
        return Sample(stamp, speed_kmh * self.wheel_scale, suspicious,
                      result == STAMP_RESYNC_BACKWARD)

    def driver_cmd(self, stamp: float, position: int) -> Optional[Sample]:
        state = self.state('driver_cmd')
        if abs(position) > self.p.driver_position_limit:
            return self._reject(state, 'value_out_of_range')
        result = self._check_stamp(state, stamp)
        if result == STAMP_REJECTED:
            return None
        state.last_value = position
        return Sample(stamp, int(position), time_reset=result == STAMP_RESYNC_BACKWARD)

    def gnss_fix(self, source: str, stamp: float, status: int,
                 latitude: float, longitude: float, altitude: float) -> Optional[Sample]:
        state = self.state(f'gnss_fix_{source}')
        if status < 0:
            return self._reject(state, 'no_fix')
        if not all(math.isfinite(v) for v in (latitude, longitude, altitude)):
            return self._reject(state, 'value_invalid')
        if abs(latitude) > 90.0 or abs(longitude) > 180.0 or (latitude == 0.0 and longitude == 0.0):
            return self._reject(state, 'value_out_of_range')
        if self._check_stamp(state, stamp) == STAMP_REJECTED:
            return None
        state.last_value = (latitude, longitude, altitude, int(status))
        return Sample(stamp, state.last_value)

    def gnss_vel(self, source: str, stamp: float, vx: float, vy: float, vz: float) -> Optional[Sample]:
        state = self.state(f'gnss_vel_{source}')
        if not all(math.isfinite(v) for v in (vx, vy, vz)):
            return self._reject(state, 'value_invalid')
        if math.sqrt(vx * vx + vy * vy + vz * vz) > self.p.wheel_speed_max_kmh / 3.6:
            return self._reject(state, 'value_out_of_range')
        if self._check_stamp(state, stamp) == STAMP_REJECTED:
            return None
        state.last_value = (vx, vy, vz)
        return Sample(stamp, state.last_value)

    # --- Метки времени ---

    def _check_stamp(self, state: TopicState, stamp: float) -> str:
        p = self.p
        if not math.isfinite(stamp) or stamp <= 0.0:
            self._reject(state, 'stamp_invalid')
            return STAMP_REJECTED
        last = state.last_stamp
        if last is not None and stamp <= last:
            # Откат назад: мелкий — запоздавшие сообщения, крупный — возможно, новый отсчёт
            reason = 'stamp_duplicate' if stamp == last else 'stamp_backward'
            big_shift = last - stamp > p.stamp_max_lag
        else:
            reference = self._reference_stamp(state.name)
            own_jump = stamp - last if last is not None else math.inf
            if (reference is not None and stamp - reference > p.stamp_max_lead
                    and own_jump > p.stamp_max_lead):
                reason, big_shift = 'stamp_ahead', True
            else:
                self._accept(state, stamp)
                return STAMP_OK

        if big_shift and self._confirms_new_timebase(state, stamp):
            state.resyncs += 1
            backward = last is not None and stamp < last
            state.last_stamp = None     # пропуск между отсчётами не считаем
            state.last_value = None
            self._accept(state, stamp)
            return STAMP_RESYNC_BACKWARD if backward else STAMP_RESYNC_FORWARD
        if not big_shift:
            state.chain.clear()
        self._reject(state, reason)
        return STAMP_REJECTED

    def _confirms_new_timebase(self, state: TopicState, stamp: float) -> bool:
        """Копит подряд идущие согласованные «чужие» метки; True — их достаточно для перехода."""
        chain = state.chain
        if chain and 0.0 < stamp - chain[-1] <= self.p.gap_threshold:
            chain.append(stamp)
        else:
            chain[:] = [stamp]
        if len(chain) >= self.p.stamp_resync_count:
            chain.clear()
            return True
        return False

    def _reference_stamp(self, name: str) -> Optional[float]:
        """Самая поздняя принятая метка других входов трамвая."""
        stamps = [self.topics[n].last_stamp for n in VEHICLE_TOPICS
                  if n != name and n in self.topics and self.topics[n].last_stamp is not None]
        return max(stamps) if stamps else None

    def _accept(self, state: TopicState, stamp: float) -> None:
        if state.last_stamp is not None:
            gap = stamp - state.last_stamp
            if gap > self.p.gap_threshold:
                state.gaps += 1
            state.max_gap = max(state.max_gap, gap)
        state.last_stamp = stamp
        state.accepted += 1
        state.chain.clear()

    @staticmethod
    def _reject(state: TopicState, reason: str) -> None:
        state.rejected[reason] += 1
        return None

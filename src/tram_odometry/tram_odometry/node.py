"""ROS 2 нода резервной одометрии трамвая: сообщения ROS ↔ ядро (core.py).

Нода подписывается на входные топики, передаёт данные ядру и публикует его результат:
/result/velocity и /result/position (контракт скрипта-судьи), а также ускорение, флаг
проскальзывания и диагностику. Логика и математика — в ядре, предобработке и оценщиках.
"""
import functools
import math
import os
import signal
import time
from dataclasses import fields

import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import AccelStamped, TwistStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float64
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from tram_odometry.core import CoreParams, OdometryCore
from tram_odometry.estimators import ESTIMATORS
from tram_odometry.preprocessing import PreprocessingParams
from tram_odometry.supervisor import HOLD

# Входные топики; ключ — имя входа в ядре
INPUT_TOPICS = {
    'front': '/vehicle/front_bogie_velocity',
    'rear': '/vehicle/rear_bogie_velocity',
    'driver_cmd': '/vehicle/driver_position_cmd',
    'gnss_fix_master': '/sensing/gnss/master/fix',
    'gnss_fix_rover': '/sensing/gnss/rover/fix',
    'gnss_vel_master': '/sensing/gnss/master/vel',
    'gnss_vel_rover': '/sensing/gnss/rover/vel',
}
# Выходные топики: первые два — контракт скрипта-судьи, остальные — дополнительные
TOPIC_OUT_VELOCITY = '/result/velocity'
TOPIC_OUT_POSITION = '/result/position'
TOPIC_OUT_ACCELERATION = '/result/acceleration'
TOPIC_OUT_SLIP = '/result/slip_detected'
TOPIC_OUT_SLIP_RATIO = '/result/slip_ratio'
TOPIC_DIAGNOSTICS = '/diagnostics'

# Подписка best-effort совместима и с reliable, и с best-effort издателем
INPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
)
# Публикация reliable (по умолчанию); судья подписывается best-effort — это совместимо
OUTPUT_QOS_DEPTH = 10

# Минимальная частота публикации по требованию ТЗ, Гц
MIN_OUTPUT_RATE = 10.0
# Вход, от которого столько секунд реального времени нет принятых сообщений, считается замолчавшим
# (время bag в этом случае стоит, поэтому возраст по нему тишину не покажет)
INPUT_WALL_TIMEOUT = 2.0

# Параметры ROS для ядра и предобработки: имя → описание.
# Значения по умолчанию — в CoreParams и PreprocessingParams (единственное место)
PARAM_DESCRIPTIONS = {
    'estimator': 'Оценщик: ' + ', '.join(ESTIMATORS) + ' (gnss_passthrough — только для тестов)',
    'vehicle_id': 'Трамвай (30618 или 30639): выбирает откалиброванный по GNSS масштаб колёс',
    'wheel_speed_scale':
        'Множитель скорости колёс км/ч → м/с; 0 — откалиброванный для vehicle_id',
    'wheel_speed_max_kmh': 'Скорость колеса выше (км/ч) — отброс',
    'wheel_negative_tolerance_kmh':
        'Скорость от -этого значения до 0 (км/ч) — шум, обнуляется; ниже — отброс',
    'suspicious_accel': 'Изменение скорости колеса быстрее (м/с²) помечается подозрительным',
    'suspicious_margin_kmh': 'Запас на шум к порогу по ускорению, км/ч',
    'bogie_mismatch_kmh':
        'Расхождение тележек больше (км/ч) — подозрительно (проскальзывание, юз, сбой)',
    'wheel_timeout': 'Измерение старше этого времени (с) считается устаревшим',
    'stamp_max_lead': 'Метка, убежавшая вперёд больше чем на столько секунд, отбрасывается',
    'stamp_max_lag':
        'Откат метки назад больше чем на столько секунд — возможная смена отсчёта времени',
    'stamp_resync_count': 'Столько согласованных меток подряд подтверждают новый отсчёт времени',
    'driver_cmd_timeout':
        'Публикация идёт по сообщениям контроллера; если он молчит дольше (с, по времени bag), '
        'то по сообщениям колёс',
    'max_output_speed':
        'Скорость от оценщика выше (м/с) считается сбоем — переключение на запасной',
    'bogie_mismatch_max_duration':
        'Расхождение тележек дольше (с) — отказ датчика: берётся большая скорость',
    'wheel_hold_max':
        'Без данных колёс скорость удерживается столько секунд, затем плавно снижается',
    'wheel_decay_tau': 'Постоянная времени снижения скорости без данных колёс, с',
    'wheel_time_alignment':
        'Пересчитывать измерения колёс по ускорению на метку результата (синхронизация по времени)',
    'speed_time_offset': 'Скорость результата относится к моменту (метка − это значение), с',
    'route_map_file':
        'Карта маршрута (.csv); пусто — карта из пакета (config/route_map.csv), none — без карты',
    'reference_antenna':
        'Приёмник GNSS для начала координат и привязки к карте: master или rover',
    'gnss_correction': 'Поправлять положение на карте по редким точкам GNSS в середине маршрута',
    'gnss_correction_interval': 'Коррекция по GNSS не чаще, чем раз в столько секунд',
    'route_stops_file':
        'Остановки (.csv) для уточнения пути на стоянках; пусто — файл из пакета, none — без них',
    'stop_correction': 'Уточнять путь на стоянках у известных остановок',
    'stop_min_duration': 'Стоянка дольше (с) — привязка к ближайшей известной остановке',
}


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def guarded(method):
    """Исключение в обработчике не должно ронять ноду: учитываем, пишем в лог, работаем дальше."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self._callback_errors += 1
            self.get_logger().error(
                f'Ошибка в {method.__name__}: {error!r}', throttle_duration_sec=5.0)
    return wrapper


class TramOdometryNode(Node):

    def __init__(self) -> None:
        super().__init__('tram_odometry')
        self._callback_errors = 0

        pre_params = self._declare_dataclass(PreprocessingParams)
        core_params = self._declare_dataclass(CoreParams)
        if not core_params.route_map_file:
            core_params.route_map_file = os.path.join(
                get_package_share_directory('tram_odometry'), 'config', 'route_map.csv')
        elif core_params.route_map_file.lower() == 'none':
            core_params.route_map_file = ''
        if not core_params.route_stops_file:
            core_params.route_stops_file = os.path.join(
                get_package_share_directory('tram_odometry'), 'config', 'route_stops.csv')
        elif core_params.route_stops_file.lower() == 'none':
            core_params.route_stops_file = ''
        self._frame_id = self._declare('frame_id', 'map', 'Система координат положения')
        self._child_frame_id = self._declare(
            'child_frame_id', 'base_link', 'Система координат трамвая')
        diagnostics_period = self._declare(
            'diagnostics_period', 1.0, 'Период публикации /diagnostics, с (0 — выключить)')
        status_period = self._declare(
            'status_period', 5.0, 'Период вывода статуса в лог, с (0 — выключить)')

        self._core = OdometryCore(pre_params, core_params, log=self._log_from_core)

        self._pub_velocity = self.create_publisher(
            VelocitySensor, TOPIC_OUT_VELOCITY, OUTPUT_QOS_DEPTH)
        self._pub_position = self.create_publisher(
            Odometry, TOPIC_OUT_POSITION, OUTPUT_QOS_DEPTH)
        self._pub_acceleration = self.create_publisher(
            AccelStamped, TOPIC_OUT_ACCELERATION, OUTPUT_QOS_DEPTH)
        self._pub_slip = self.create_publisher(Bool, TOPIC_OUT_SLIP, OUTPUT_QOS_DEPTH)
        self._pub_slip_ratio = self.create_publisher(Float64, TOPIC_OUT_SLIP_RATIO, OUTPUT_QOS_DEPTH)
        self._pub_diagnostics = self.create_publisher(
            DiagnosticArray, TOPIC_DIAGNOSTICS, OUTPUT_QOS_DEPTH)

        self.create_subscription(
            VelocitySensor, INPUT_TOPICS['front'],
            lambda msg: self._on_wheel('front', msg), INPUT_QOS)
        self.create_subscription(
            VelocitySensor, INPUT_TOPICS['rear'],
            lambda msg: self._on_wheel('rear', msg), INPUT_QOS)
        self.create_subscription(
            DriverControllerCommand, INPUT_TOPICS['driver_cmd'], self._on_driver_cmd, INPUT_QOS)
        # GNSS: начальная выставка и коррекция по редким точкам (разрешено организаторами)
        for source in ('master', 'rover'):
            self.create_subscription(
                NavSatFix, INPUT_TOPICS[f'gnss_fix_{source}'],
                lambda msg, s=source: self._on_gnss_fix(s, msg), INPUT_QOS)
            self.create_subscription(
                TwistStamped, INPUT_TOPICS[f'gnss_vel_{source}'],
                lambda msg, s=source: self._on_gnss_vel(s, msg), INPUT_QOS)
        if self._core.estimator.requires_gnss:
            self.get_logger().warn(
                'ТЕСТОВЫЙ РЕЖИМ: оценщик берёт скорость и положение из GNSS — для сдачи запрещено!')

        self._proc_times = []           # время обработки «вход → публикация», с
        self._diag_prev = None          # (время bag, принято по входам, опубликовано)
        if diagnostics_period > 0.0:
            self.create_timer(diagnostics_period, self._publish_diagnostics)
        if status_period > 0.0:
            self.create_timer(status_period, self._log_status)

        estimator = self._core.estimator
        self.get_logger().info(
            f'Оценщик: {estimator.primary.name}'
            + (f' (запасной: {estimator.fallback.name})' if estimator.fallback else '')
            + f'; публикация: {TOPIC_OUT_VELOCITY}, {TOPIC_OUT_POSITION}')

    def _declare(self, name, default, description):
        return self.declare_parameter(
            name, default, ParameterDescriptor(description=description)).value

    def _declare_dataclass(self, cls):
        """Объявить параметры ROS для полей dataclass (по умолчанию — его значения)."""
        defaults = cls()
        values = {f.name: self._declare(f.name, getattr(defaults, f.name), PARAM_DESCRIPTIONS[f.name])
                  for f in fields(cls) if f.name in PARAM_DESCRIPTIONS}
        return cls(**values)

    def _log_from_core(self, level: str, text: str) -> None:
        logger = self.get_logger()
        {'error': logger.error, 'warn': logger.warn}.get(level, logger.info)(text)

    # --- Входные данные ---

    @guarded
    def _on_wheel(self, bogie: str, msg: VelocitySensor) -> None:
        received = time.perf_counter()
        output = self._core.on_wheel(bogie, stamp_to_sec(msg.header.stamp), msg.velocity)
        if output is not None:
            self._publish(output, msg.header.stamp, received)

    @guarded
    def _on_driver_cmd(self, msg: DriverControllerCommand) -> None:
        received = time.perf_counter()
        output = self._core.on_driver_cmd(stamp_to_sec(msg.header.stamp), int(msg.position))
        if output is not None:
            self._publish(output, msg.header.stamp, received)

    @guarded
    def _on_gnss_fix(self, source: str, msg: NavSatFix) -> None:
        self._core.on_gnss_fix(source, stamp_to_sec(msg.header.stamp), msg.status.status,
                               msg.latitude, msg.longitude, msg.altitude)

    @guarded
    def _on_gnss_vel(self, source: str, msg: TwistStamped) -> None:
        linear = msg.twist.linear
        self._core.on_gnss_vel(source, stamp_to_sec(msg.header.stamp),
                               linear.x, linear.y, linear.z)

    # --- Публикация результата ---

    def _publish(self, output, stamp_msg, received: float) -> None:
        """Результат ядра → сообщения; метка — исходная метка входного сообщения-триггера."""
        estimate = output.estimate

        velocity_msg = VelocitySensor()
        velocity_msg.header.stamp = stamp_msg
        velocity_msg.header.frame_id = self._child_frame_id
        velocity_msg.velocity = float(estimate.velocity)

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x = float(output.x)
        odom.pose.pose.position.y = float(output.y)
        odom.pose.pose.position.z = float(output.z)
        # Поворот только вокруг вертикали: кватернион (0, 0, sin(yaw/2), cos(yaw/2))
        odom.pose.pose.orientation.z = math.sin(output.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(output.yaw / 2.0)
        odom.pose.covariance = output.pose_covariance
        # Скорость — продольная, в системе трамвая (child_frame_id)
        odom.twist.twist.linear.x = float(estimate.velocity)
        odom.twist.covariance = output.twist_covariance

        acceleration_msg = AccelStamped()
        acceleration_msg.header.stamp = stamp_msg
        acceleration_msg.header.frame_id = self._child_frame_id
        acceleration_msg.accel.linear.x = float(estimate.acceleration)

        self._pub_velocity.publish(velocity_msg)
        self._pub_position.publish(odom)
        self._pub_acceleration.publish(acceleration_msg)
        self._pub_slip.publish(Bool(data=bool(estimate.slip_detected)))
        # Скольжение — только когда оценщик его дал (нужны свежие данные колёс)
        if estimate.slip_ratio is not None:
            self._pub_slip_ratio.publish(Float64(data=float(estimate.slip_ratio)))
        self._proc_times.append(time.perf_counter() - received)

    # --- Диагностика ---

    @guarded
    def _publish_diagnostics(self) -> None:
        core = self._core
        now = core.pre.current_stamp()
        accepted = {name: core.pre.state(name).accepted for name in core.inputs}
        prev = self._diag_prev
        span = now - prev[0] if prev is not None and now is not None and prev[0] is not None else 0.0

        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        for name in core.inputs:
            state = core.pre.state(name)
            rate = (accepted[name] - prev[1].get(name, 0)) / span if span > 0.0 else 0.0
            age = now - state.last_stamp if now is not None and state.last_stamp is not None else None
            level, message = DiagnosticStatus.OK, 'OK'
            silence = time.perf_counter() - core.arrival.get(name, 0.0)
            if state.last_stamp is None:
                level, message = DiagnosticStatus.WARN, 'нет данных'
            elif silence > INPUT_WALL_TIMEOUT:
                level, message = DiagnosticStatus.WARN, f'нет новых данных {silence:.0f} с'
            elif name in ('front', 'rear', 'driver_cmd') and age is not None \
                    and age > core.pre.p.wheel_timeout:
                level, message = DiagnosticStatus.WARN, f'нет данных {age:.1f} с'
            values = {
                'принято': state.accepted, 'частота, Гц': f'{rate:.1f}',
                'возраст, с': '-' if age is None else f'{age:.2f}',
                'отброшено': sum(state.rejected.values()),
                **{f'отброшено: {reason}': count for reason, count in sorted(state.rejected.items())},
                'подозрительных': state.suspicious, 'обнулено': state.clamped,
                'пропусков > 1 с': state.gaps, 'макс. пропуск, с': f'{state.max_gap:.2f}',
                'смен отсчёта времени': state.resyncs,
            }
            array.status.append(self._status(f'вход {INPUT_TOPICS[name]}', level, message, values))

        est = core.estimator
        last = core.last_output.estimate if core.last_output is not None else None
        level, message = DiagnosticStatus.OK, f'активен {est.active}'
        if est.active == HOLD:
            level, message = DiagnosticStatus.ERROR, 'все оценщики отказали, скорость удерживается'
        elif est.active != est.primary.name:
            level = DiagnosticStatus.WARN
            message = f'основной {est.primary.name} отказал, активен запасной {est.active}'
        values = {
            'основной': est.primary.name, 'активен': est.active, 'переключений': est.switches,
            **{f'сбоев: {name}': count for name, count in sorted(est.failures.items())},
            'последняя ошибка': est.last_error or '-',
        }
        if last is not None:
            values.update({
                'скорость, м/с': f'{last.velocity:.2f}', 'путь, м': f'{last.distance:.1f}',
                'ускорение, м/с²': f'{last.acceleration:.2f}',
                'проскальзывание': 'да' if last.slip_detected else 'нет',
                'скольжение': '-' if last.slip_ratio is None else f'{last.slip_ratio:.3f}',
            })
        array.status.append(self._status('оценщик', level, message, values))

        anchor = core.anchor
        level = DiagnosticStatus.OK if anchor.on_route else DiagnosticStatus.WARN
        values = {
            'состояние': anchor.state, 'опорный приёмник': anchor.reference,
            'курс': '-' if anchor.heading is None else f'{math.degrees(anchor.heading):.1f}°',
            'курс по': anchor.heading_source,
            'до карты при выставке, м': '-' if anchor.init_distance is None else f'{anchor.init_distance:.2f}',
            'коррекций GNSS': anchor.corrections, 'отклонено точек GNSS': anchor.rejected,
            'привязок к остановкам': anchor.stop_updates,
        }
        if anchor.on_route and last is not None:
            values['σ вдоль пути, м'] = f'{math.sqrt(anchor.along_var(last.distance)):.2f}'
        array.status.append(self._status('выставка по GNSS', level, anchor.state, values))

        out_rate = (core.published - prev[2]) / span if span > 0.0 else 0.0
        times = self._proc_times
        self._proc_times = []
        level, message = DiagnosticStatus.OK, 'OK'
        if span > 0.0 and out_rate < MIN_OUTPUT_RATE:
            level, message = DiagnosticStatus.WARN, f'частота {out_rate:.1f} Гц < {MIN_OUTPUT_RATE:.0f}'
        values = {
            'опубликовано': core.published, 'частота, Гц': f'{out_rate:.1f}',
            'пропущено (метка не растёт)': core.skipped,
            'смен отсчёта времени': core.time_resets,
            'ошибок в обработчиках': self._callback_errors,
            'обработка, мс (сред.)': f'{1e3 * sum(times) / len(times):.2f}' if times else '-',
            'обработка, мс (макс.)': f'{1e3 * max(times):.2f}' if times else '-',
        }
        array.status.append(self._status('выход', level, message, values))
        self._pub_diagnostics.publish(array)
        self._diag_prev = (now, accepted, core.published)

    @staticmethod
    def _status(name, level, message, values) -> DiagnosticStatus:
        status = DiagnosticStatus()
        status.level = level
        status.name = f'tram_odometry: {name}'
        status.message = message
        status.hardware_id = 'tram'
        status.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
        return status

    @guarded
    def _log_status(self) -> None:
        core = self._core
        estimate = core.last_output.estimate if core.last_output is not None else None
        state = (f'v={estimate.velocity:.2f} м/с, путь={estimate.distance:.1f} м'
                 + (', ПРОСКАЛЬЗЫВАНИЕ' if estimate.slip_detected else '')
                 if estimate is not None else 'данных ещё нет')
        inputs = ', '.join(
            f'{name}={core.pre.state(name).accepted}'
            + (f'(-{sum(core.pre.state(name).rejected.values())})'
               if core.pre.state(name).rejected else '')
            for name in core.inputs[:3])
        self.get_logger().info(
            f'{state}; оценщик: {core.estimator.active}; принято: {inputs}; '
            f'опубликовано={core.published}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TramOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl+C в терминале приходит и ноде, и launch, который пересылает его ещё раз:
        # повторный сигнал не должен прерывать корректное завершение
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

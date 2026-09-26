"""ROS 2 нода резервной одометрии трамвая.

Подписывается на датчики скорости тележек и контроллер водителя, проверяет входные
данные (preprocessing), передаёт их оценщику под надзором (supervisor) и публикует
/result/velocity и /result/position, а также ускорение, флаг проскальзывания и
диагностику. Сама нода не содержит математики модели.
"""
import functools
import math
import signal
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import AccelStamped, TwistStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from tram_odometry.estimators import ESTIMATORS, create_estimator
from tram_odometry.preprocessing import InputPreprocessor, PreprocessingParams
from tram_odometry.projection import StraightLineProjector
from tram_odometry.supervisor import HOLD, SupervisedEstimator

DEFAULT_ESTIMATOR = 'wheel_baseline'

# Входные топики (разрешены в основном контуре); ключ — имя входа в предобработке
INPUT_TOPICS = {
    'front': '/vehicle/front_bogie_velocity',
    'rear': '/vehicle/rear_bogie_velocity',
    'driver_cmd': '/vehicle/driver_position_cmd',
    # GNSS — только для оценщиков, которые явно его требуют (тестовый gnss_passthrough)
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

# Дисперсии величин, которые пока не оцениваются
UNKNOWN_VARIANCE = 1e3
# Трамвай на рельсах не движется вбок и вертикально относительно корпуса
CONSTRAINED_VARIANCE = 1e-4
# Минимальная частота публикации по требованию ТЗ, Гц
MIN_OUTPUT_RATE = 10.0
# Вход, от которого столько секунд реального времени нет принятых сообщений, считается замолчавшим
# (время bag в этом случае стоит, поэтому возраст по нему тишину не покажет)
INPUT_WALL_TIMEOUT = 2.0
# До первого измерения колёс скорость неизвестна: публикация ждёт его, но не дольше этого
# времени (с, по времени bag) — при неработающих датчиках колёс она всё равно начнётся
WHEEL_WAIT_AT_START = 1.0


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def diagonal_covariance(diagonal) -> list:
    """Ковариационная матрица 6x6 (построчно) с заданной диагональю."""
    covariance = [0.0] * 36
    for i, value in enumerate(diagonal):
        covariance[i * 7] = float(value)
    return covariance


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

        estimator_name = self._declare(
            'estimator', DEFAULT_ESTIMATOR,
            'Оценщик: ' + ', '.join(ESTIMATORS) + ' (gnss_passthrough — только для тестов)')
        pre = PreprocessingParams(
            wheel_speed_scale=self._declare(
                'wheel_speed_scale', 1.0 / 3.6,
                'Множитель скорости колёс: датчики пишут км/ч, оценщики работают в м/с'),
            wheel_speed_max_kmh=self._declare(
                'wheel_speed_max_kmh', 120.0, 'Скорость колеса выше (км/ч) — отброс'),
            wheel_negative_tolerance_kmh=self._declare(
                'wheel_negative_tolerance_kmh', 1.0,
                'Скорость от -этого значения до 0 (км/ч) — шум, обнуляется; ниже — отброс'),
            suspicious_accel=self._declare(
                'suspicious_accel', 3.0,
                'Изменение скорости колеса быстрее (м/с²) помечается подозрительным'),
            suspicious_margin_kmh=self._declare(
                'suspicious_margin_kmh', 1.0, 'Запас на шум к порогу по ускорению, км/ч'),
            bogie_mismatch_kmh=self._declare(
                'bogie_mismatch_kmh', 3.0,
                'Расхождение тележек больше (км/ч) — подозрительно (проскальзывание, юз, сбой)'),
            wheel_timeout=self._declare(
                'wheel_timeout', 0.5, 'Измерение старше этого времени (с) считается устаревшим'),
            stamp_max_lead=self._declare(
                'stamp_max_lead', 5.0,
                'Метка, убежавшая вперёд больше чем на столько секунд, отбрасывается'),
            stamp_max_lag=self._declare(
                'stamp_max_lag', 5.0,
                'Откат метки назад больше чем на столько секунд — возможная смена отсчёта времени'),
            stamp_resync_count=self._declare(
                'stamp_resync_count', 5,
                'Столько согласованных меток подряд подтверждают новый отсчёт времени'),
        )
        self._cmd_timeout = self._declare(
            'driver_cmd_timeout', 0.15,
            'Публикация идёт по сообщениям контроллера; если он молчит дольше (с, по времени bag), '
            'то по сообщениям колёс')
        estimator_params = {
            'wheel_timeout': pre.wheel_timeout,
            'bogie_mismatch': pre.bogie_mismatch_kmh * pre.wheel_speed_scale,
            'plausible_accel': pre.suspicious_accel,
            'plausible_margin': pre.suspicious_margin_kmh * pre.wheel_speed_scale,
            'bogie_mismatch_max_duration': self._declare(
                'bogie_mismatch_max_duration', 3.0,
                'Расхождение тележек дольше (с) — отказ датчика: берётся большая скорость'),
            'wheel_hold_max': self._declare(
                'wheel_hold_max', 2.0,
                'Без данных колёс скорость удерживается столько секунд, затем плавно снижается'),
            'wheel_decay_tau': self._declare(
                'wheel_decay_tau', 5.0, 'Постоянная времени снижения скорости без данных колёс, с'),
        }
        max_output_speed = self._declare(
            'max_output_speed', 30.0,
            'Скорость от оценщика выше (м/с) считается сбоем — переключение на запасной')
        self._frame_id = self._declare('frame_id', 'map', 'Система координат положения')
        self._child_frame_id = self._declare(
            'child_frame_id', 'base_link', 'Система координат трамвая')
        diagnostics_period = self._declare(
            'diagnostics_period', 1.0, 'Период публикации /diagnostics, с (0 — выключить)')
        status_period = self._declare(
            'status_period', 5.0, 'Период вывода статуса в лог, с (0 — выключить)')

        self._pre = InputPreprocessor(pre)
        self._estimator = self._create_supervisor(
            estimator_name, estimator_params, max_output_speed)
        self._projector = StraightLineProjector()

        self._pub_velocity = self.create_publisher(
            VelocitySensor, TOPIC_OUT_VELOCITY, OUTPUT_QOS_DEPTH)
        self._pub_position = self.create_publisher(
            Odometry, TOPIC_OUT_POSITION, OUTPUT_QOS_DEPTH)
        self._pub_acceleration = self.create_publisher(
            AccelStamped, TOPIC_OUT_ACCELERATION, OUTPUT_QOS_DEPTH)
        self._pub_slip = self.create_publisher(Bool, TOPIC_OUT_SLIP, OUTPUT_QOS_DEPTH)
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
        self._inputs = ['front', 'rear', 'driver_cmd']
        if self._estimator.requires_gnss:
            self.get_logger().warn(
                'ТЕСТОВЫЙ РЕЖИМ: оценщик берёт данные из GNSS — для сдачи решения запрещено!')
            for source in ('master', 'rover'):
                self.create_subscription(
                    NavSatFix, INPUT_TOPICS[f'gnss_fix_{source}'],
                    lambda msg, s=source: self._on_gnss_fix(s, msg), INPUT_QOS)
                self.create_subscription(
                    TwistStamped, INPUT_TOPICS[f'gnss_vel_{source}'],
                    lambda msg, s=source: self._on_gnss_vel(s, msg), INPUT_QOS)
                self._inputs += [f'gnss_fix_{source}', f'gnss_vel_{source}']

        self._last_cmd_stamp = None     # время последнего принятого сообщения контроллера
        self._last_pub_stamp = None     # время последней публикации (bag)
        self._last_estimate = None
        self._published = 0
        self._skipped = 0               # публикации, пропущенные из-за немонотонной метки
        self._time_resets = 0
        self._proc_times = []           # время обработки «вход → публикация», с
        self._diag_prev = None          # (время bag, принято по входам, опубликовано)
        self._arrival = {}              # вход → реальное время последнего принятого сообщения
        self._wheel_seen = False        # было ли хоть одно принятое измерение колёс
        self._first_trigger_stamp = None
        if diagnostics_period > 0.0:
            self.create_timer(diagnostics_period, self._publish_diagnostics)
        if status_period > 0.0:
            self.create_timer(status_period, self._log_status)

        self.get_logger().info(
            f'Оценщик: {self._estimator.primary.name}'
            + (f' (запасной: {self._estimator.fallback.name})' if self._estimator.fallback else '')
            + f'; публикация: {TOPIC_OUT_VELOCITY}, {TOPIC_OUT_POSITION}')

    def _declare(self, name, default, description):
        return self.declare_parameter(
            name, default, ParameterDescriptor(description=description)).value

    def _create_supervisor(self, name, params, max_speed):
        """Основной оценщик + запасной wheel_baseline; ошибка создания не роняет ноду."""
        if name not in ESTIMATORS:
            self.get_logger().error(
                f'Неизвестный оценщик "{name}", доступны: {", ".join(ESTIMATORS)}. '
                f'Использую {DEFAULT_ESTIMATOR}.')
            name = DEFAULT_ESTIMATOR
        try:
            primary = create_estimator(name, params)
        except Exception as error:
            self.get_logger().error(
                f'Не удалось создать оценщик {name}: {error!r}. Использую {DEFAULT_ESTIMATOR}.')
            name = DEFAULT_ESTIMATOR
            primary = create_estimator(name, params)
        fallback = None if name == DEFAULT_ESTIMATOR else create_estimator(DEFAULT_ESTIMATOR, params)
        return SupervisedEstimator(primary, fallback, max_speed)

    # --- Входные данные ---

    @guarded
    def _on_wheel(self, bogie: str, msg: VelocitySensor) -> None:
        received = time.perf_counter()
        sample = self._pre.wheel(bogie, stamp_to_sec(msg.header.stamp), msg.velocity)
        if sample is None:
            return
        self._arrival[bogie] = received
        self._wheel_seen = True
        if sample.time_reset:
            self._reset_time(sample.stamp)
        self._estimator.on_wheel(bogie, sample.stamp, sample.value, sample.suspicious)
        # Основной триггер публикации — контроллер (~20 Гц); колёса (~9,4 Гц) — если он молчит
        if self._last_cmd_stamp is None or sample.stamp - self._last_cmd_stamp > self._cmd_timeout:
            self._publish(msg.header.stamp, received)

    @guarded
    def _on_driver_cmd(self, msg: DriverControllerCommand) -> None:
        received = time.perf_counter()
        sample = self._pre.driver_cmd(stamp_to_sec(msg.header.stamp), int(msg.position))
        if sample is None:
            return
        self._arrival['driver_cmd'] = received
        if sample.time_reset:
            self._reset_time(sample.stamp)
        self._last_cmd_stamp = sample.stamp
        self._estimator.on_driver_cmd(sample.stamp, sample.value)
        self._publish(msg.header.stamp, received)

    @guarded
    def _on_gnss_fix(self, source: str, msg: NavSatFix) -> None:
        sample = self._pre.gnss_fix(source, stamp_to_sec(msg.header.stamp), msg.status.status,
                                    msg.latitude, msg.longitude, msg.altitude)
        if sample is not None:
            self._arrival[f'gnss_fix_{source}'] = time.perf_counter()
            self._estimator.on_gnss_fix(source, sample.stamp, *sample.value)

    @guarded
    def _on_gnss_vel(self, source: str, msg: TwistStamped) -> None:
        linear = msg.twist.linear
        sample = self._pre.gnss_vel(source, stamp_to_sec(msg.header.stamp),
                                    linear.x, linear.y, linear.z)
        if sample is not None:
            self._arrival[f'gnss_vel_{source}'] = time.perf_counter()
            self._estimator.on_gnss_vel(source, sample.stamp, *sample.value)

    def _reset_time(self, stamp: float) -> None:
        """Отсчёт времени во входных данных сменился назад: начинаем метки выхода заново."""
        self._time_resets += 1
        self._last_pub_stamp = None
        self._last_cmd_stamp = None
        self._estimator.on_time_reset(stamp)
        self.get_logger().warn(f'Отсчёт времени во входных данных сменился назад (метка {stamp:.3f})')

    # --- Публикация результата ---

    def _publish(self, stamp_msg, received: float) -> None:
        stamp = stamp_to_sec(stamp_msg)
        # Метки на выходе строго возрастают: судья сопоставляет их с эталоном по времени
        if self._last_pub_stamp is not None and stamp <= self._last_pub_stamp:
            self._skipped += 1
            return
        if not self._wheel_seen:
            if self._first_trigger_stamp is None:
                self._first_trigger_stamp = stamp
            if stamp - self._first_trigger_stamp < WHEEL_WAIT_AT_START:
                return
        estimate = self._estimator.estimate(stamp)
        if estimate.position is not None:
            x, y, z = estimate.position
            yaw = estimate.yaw if estimate.yaw is not None else 0.0
        else:
            x, y, z, yaw = self._projector.project(estimate.distance)

        velocity_msg = VelocitySensor()
        velocity_msg.header.stamp = stamp_msg
        velocity_msg.header.frame_id = self._child_frame_id
        velocity_msg.velocity = float(estimate.velocity)

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        # Поворот только вокруг вертикали: кватернион (0, 0, sin(yaw/2), cos(yaw/2))
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # Порядок диагонали: x, y, z, крен, тангаж, курс; ориентация пока не оценивается
        odom.pose.covariance = diagonal_covariance([
            estimate.distance_var, estimate.distance_var, estimate.distance_var,
            UNKNOWN_VARIANCE, UNKNOWN_VARIANCE, UNKNOWN_VARIANCE])
        # Скорость — продольная, в системе трамвая (child_frame_id)
        odom.twist.twist.linear.x = float(estimate.velocity)
        odom.twist.covariance = diagonal_covariance([
            estimate.velocity_var, CONSTRAINED_VARIANCE, CONSTRAINED_VARIANCE,
            UNKNOWN_VARIANCE, UNKNOWN_VARIANCE, UNKNOWN_VARIANCE])

        acceleration_msg = AccelStamped()
        acceleration_msg.header.stamp = stamp_msg
        acceleration_msg.header.frame_id = self._child_frame_id
        acceleration_msg.accel.linear.x = float(estimate.acceleration)

        self._pub_velocity.publish(velocity_msg)
        self._pub_position.publish(odom)
        self._pub_acceleration.publish(acceleration_msg)
        self._pub_slip.publish(Bool(data=bool(estimate.slip_detected)))
        self._last_pub_stamp = stamp
        self._last_estimate = estimate
        self._published += 1
        self._proc_times.append(time.perf_counter() - received)

    # --- Диагностика ---

    @guarded
    def _publish_diagnostics(self) -> None:
        now = self._pre.current_stamp()
        accepted = {name: self._pre.state(name).accepted for name in self._inputs}
        prev = self._diag_prev
        span = now - prev[0] if prev is not None and now is not None and prev[0] is not None else 0.0

        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        for name in self._inputs:
            state = self._pre.state(name)
            rate = (accepted[name] - prev[1].get(name, 0)) / span if span > 0.0 else 0.0
            age = now - state.last_stamp if now is not None and state.last_stamp is not None else None
            rejected_now = sum(state.rejected.values())
            level, message = DiagnosticStatus.OK, 'OK'
            if state.last_stamp is None:
                level, message = DiagnosticStatus.WARN, 'нет данных'
            elif time.perf_counter() - self._arrival.get(name, 0.0) > INPUT_WALL_TIMEOUT:
                silence = time.perf_counter() - self._arrival.get(name, 0.0)
                level, message = DiagnosticStatus.WARN, f'нет новых данных {silence:.0f} с'
            elif name in ('front', 'rear', 'driver_cmd') and age is not None \
                    and age > self._pre.p.wheel_timeout:
                level, message = DiagnosticStatus.WARN, f'нет данных {age:.1f} с'
            values = {
                'принято': state.accepted, 'частота, Гц': f'{rate:.1f}',
                'возраст, с': '-' if age is None else f'{age:.2f}',
                'отброшено': rejected_now,
                **{f'отброшено: {reason}': count for reason, count in sorted(state.rejected.items())},
                'подозрительных': state.suspicious, 'обнулено': state.clamped,
                'пропусков > 1 с': state.gaps, 'макс. пропуск, с': f'{state.max_gap:.2f}',
                'смен отсчёта времени': state.resyncs,
            }
            array.status.append(self._status(f'вход {INPUT_TOPICS[name]}', level, message, values))

        est = self._estimator
        last = self._last_estimate
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

        out_rate = (self._published - prev[2]) / span if span > 0.0 else 0.0
        times = self._proc_times
        self._proc_times = []
        level, message = DiagnosticStatus.OK, 'OK'
        if span > 0.0 and out_rate < MIN_OUTPUT_RATE:
            level, message = DiagnosticStatus.WARN, f'частота {out_rate:.1f} Гц < {MIN_OUTPUT_RATE:.0f}'
        values = {
            'опубликовано': self._published, 'частота, Гц': f'{out_rate:.1f}',
            'пропущено (метка не растёт)': self._skipped,
            'смен отсчёта времени': self._time_resets,
            'ошибок в обработчиках': self._callback_errors,
            'обработка, мс (сред.)': f'{1e3 * sum(times) / len(times):.2f}' if times else '-',
            'обработка, мс (макс.)': f'{1e3 * max(times):.2f}' if times else '-',
        }
        array.status.append(self._status('выход', level, message, values))
        self._pub_diagnostics.publish(array)
        self._diag_prev = (now, accepted, self._published)

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
        estimate = self._last_estimate
        state = (f'v={estimate.velocity:.2f} м/с, путь={estimate.distance:.1f} м'
                 + (', ПРОСКАЛЬЗЫВАНИЕ' if estimate.slip_detected else '')
                 if estimate is not None else 'данных ещё нет')
        inputs = ', '.join(
            f'{name}={self._pre.state(name).accepted}'
            + (f'(-{sum(self._pre.state(name).rejected.values())})'
               if self._pre.state(name).rejected else '')
            for name in self._inputs[:3])
        self.get_logger().info(
            f'{state}; оценщик: {self._estimator.active}; принято: {inputs}; '
            f'опубликовано={self._published}')


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

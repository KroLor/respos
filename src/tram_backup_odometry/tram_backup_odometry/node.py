"""ROS 2 нода резервной одометрии трамвая.

Вход:  /vehicle/front_bogie_velocity, /vehicle/rear_bogie_velocity (VelocitySensor),
       /vehicle/driver_position_cmd (DriverControllerCommand),
       /sensing/gnss/{master,rover}/fix — только до завершения начальной выставки, затем подписки уничтожаются.
Выход: /result/velocity (VelocitySensor, м/с), /result/position (nav_msgs/Odometry),
       /result/geo (NavSatFix — та же оценка в WGS-84), /result/slip (std_msgs/Bool),
       /result/diagnostics (diagnostic_msgs/DiagnosticArray, 1 Гц — режим, сцепление, задержка).
Публикация — на каждое входное сообщение (≈20–30 Гц), header.stamp = stamp входа.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import os
import time
from collections import deque

import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor

from .core import build_estimator
from .estimator import Output, Params


def _stamp_sec(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _to_stamp(msg_stamp, t: float):
    ns = int(round(t * 1e9))
    msg_stamp.sec = ns // 1_000_000_000
    msg_stamp.nanosec = ns % 1_000_000_000


class BackupOdometryNode(Node):
    def __init__(self):
        super().__init__('tram_backup_odometry')
        share = get_package_share_directory('tram_backup_odometry')
        self.declare_parameter('model_file', os.path.join(share, 'config', 'model.json'))
        self.declare_parameter('map_file', os.path.join(share, 'data', 'pathgraph.json'))
        self.declare_parameter('esn_file', os.path.join(share, 'config', 'esn.json'))
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('use_gnss_init', True)
        self.declare_parameter('fallback_rate_hz', 20.0)
        defaults = Params()
        for k, v in vars(defaults).items():
            self.declare_parameter(f'estimator.{k}', v)

        model_file = self.get_parameter('model_file').value
        # явно заданные (отличные от умолчания) параметры перекрывают калибровку модели
        overrides = {k: self.get_parameter(f'estimator.{k}').value for k, v in vars(defaults).items()
                     if self.get_parameter(f'estimator.{k}').value != v}
        self.est = build_estimator(model_file, self.get_parameter('map_file').value,
                                   self.get_parameter('esn_file').value, overrides)
        p, pm, esn = self.est.p, self.est.map, self.est.esn
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value

        qos_in = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        qos_out = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub_v = self.create_publisher(VelocitySensor, '/result/velocity', qos_out)
        self.pub_p = self.create_publisher(Odometry, '/result/position', qos_out)
        self.pub_geo = self.create_publisher(NavSatFix, '/result/geo', qos_out)
        self.pub_slip = self.create_publisher(Bool, '/result/slip', qos_out)
        self.pub_acc = self.create_publisher(AccelStamped, '/result/acceleration', qos_out)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/result/diagnostics', 10)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self._on_wheel('front', m), qos_in)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity',
                                 lambda m: self._on_wheel('rear', m), qos_in)
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self._on_cmd, qos_in)
        self.gnss_subs = []
        if self.get_parameter('use_gnss_init').value:
            self._subscribe_gnss()
        else:
            self.est.init.t_first_fix = None
            self.est.p.init_timeout = 0.0

        self.last_out: Output | None = None
        self.last_pub_stamp = -math.inf
        self.recent_stamps: deque = deque(maxlen=64)
        self.last_input_wall = None
        self.lat_ms: list[float] = []
        self.n_pub = 0
        self.create_timer(1.0 / max(self.get_parameter('fallback_rate_hz').value, 1.0), self._on_timer)
        self.create_timer(1.0, self._on_diag)
        self.get_logger().info(f'модель: {model_file}; карта: {"да" if pm else "нет"}; '
                               f'масштаб колёс {p.wheel_scale:.6f}; ESN: {"да" if esn else "нет"}')

    # ------------------------------------------------------------------ GNSS (только выставка)
    def _subscribe_gnss(self):
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        for ant in ('master', 'rover'):
            self.gnss_subs.append(self.create_subscription(
                NavSatFix, f'/sensing/gnss/{ant}/fix', lambda m, a=ant: self._on_fix(a, m), qos))

    def _on_fix(self, ant, m: NavSatFix):
        if self.est.gnss_done:
            return
        self._safe(self.est.on_fix, ant, _stamp_sec(m.header), m.latitude, m.longitude, m.altitude, int(m.status.status))

    def _check_gnss_release(self):
        if self.est.gnss_done and self.gnss_subs:
            for s in self.gnss_subs:
                self.destroy_subscription(s)
            self.gnss_subs = []
            self.get_logger().info(f'выставка завершена: режим {self.est.mode}; GNSS-подписки закрыты')
        elif not self.est.gnss_done and not self.gnss_subs and self.get_parameter('use_gnss_init').value:
            self._subscribe_gnss()        # сброс оценщика (новый прогон) — снова нужна выставка

    # ------------------------------------------------------------------ входы
    def _safe(self, fn, *a):
        """Нода не должна падать на некорректных данных: ошибка → лог и сброс оценщика."""
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'ошибка оценщика: {e!r}; сброс состояния')
            self.est.reset()
            return None

    def _on_wheel(self, side, m: VelocitySensor):
        t0 = time.perf_counter()
        o = self._safe(self.est.on_wheel, side, _stamp_sec(m.header), float(m.velocity))
        self._publish(o, t0, m.header.stamp)

    def _on_cmd(self, m: DriverControllerCommand):
        t0 = time.perf_counter()
        o = self._safe(self.est.on_cmd, _stamp_sec(m.header), int(m.position))
        self._publish(o, t0, m.header.stamp)

    def _on_timer(self):
        """Входы пропали (>0.1 с по стенным часам): до 2 с публикуем экстраполяцию с постоянной скоростью.
        Состояние оценщика не меняется — после возобновления входов расчёт продолжается по их stamp."""
        if self.last_out is None or self.last_input_wall is None:
            return
        dt = time.monotonic() - self.last_input_wall
        if 0.1 < dt < 2.0:
            o = self.last_out
            ds = o.v * dt
            self._publish(dataclasses.replace(o, stamp=o.stamp + dt, e=o.e + ds * math.cos(o.yaw),
                                              n=o.n + ds * math.sin(o.yaw), s=o.s + ds), None, keep_last=True)

    # ------------------------------------------------------------------ выход
    def _publish(self, o: Output | None, t0, hdr_stamp=None, keep_last=False):
        if t0 is not None:
            self.last_input_wall = time.monotonic()
        self._check_gnss_release()
        if o is None or o.stamp in self.recent_stamps:
            return                                   # один выход на stamp (тележки часто с одинаковым stamp)
        self.recent_stamps.append(o.stamp)
        if keep_last and o.stamp <= self.last_pub_stamp:
            return
        self.last_pub_stamp = max(self.last_pub_stamp, o.stamp)
        if not keep_last and o.stamp >= self.last_pub_stamp:
            self.last_out = o
        v = VelocitySensor()
        if hdr_stamp is not None:
            v.header.stamp = hdr_stamp          # ровно stamp входного сообщения
        else:
            _to_stamp(v.header.stamp, o.stamp)
        v.header.frame_id = self.child_frame_id
        v.velocity = float(o.v)
        od = Odometry()
        od.header.stamp = copy.copy(v.header.stamp)      # без общих ссылок между сообщениями
        od.header.frame_id = self.frame_id
        od.child_frame_id = self.child_frame_id
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = o.e, o.n, o.u
        od.pose.pose.orientation.z = math.sin(0.5 * o.yaw)
        od.pose.pose.orientation.w = math.cos(0.5 * o.yaw)
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        sa2, sc2 = o.sigma_along ** 2, o.sigma_cross ** 2
        cov = [0.0] * 36
        cov[0] = c * c * sa2 + s * s * sc2
        cov[1] = cov[6] = c * s * (sa2 - sc2)
        cov[7] = s * s * sa2 + c * c * sc2
        cov[14] = 1.0
        cov[21] = cov[28] = 1e3
        cov[35] = 0.05 ** 2
        od.pose.covariance = cov
        od.twist.twist.linear.x = float(o.v)
        tc = [0.0] * 36
        tc[0] = max(o.var_v, 1e-4)
        tc[7] = tc[14] = 1e-4
        tc[21] = tc[28] = tc[35] = 1e3
        od.twist.covariance = tc
        self.pub_v.publish(v)
        if o.pos_valid:                      # до первого фикса GNSS положение неизвестно — не публикуем
            self.pub_p.publish(od)
        if o.pos_valid and math.isfinite(o.lat):
            g = NavSatFix()
            g.header.stamp = copy.copy(v.header.stamp)
            g.header.frame_id = self.child_frame_id
            g.status.status = NavSatStatus.STATUS_NO_FIX     # это не измерение GNSS, а оценка одометрии
            g.status.service = 0
            g.latitude, g.longitude, g.altitude = o.lat, o.lon, o.h
            g.position_covariance = [cov[0], cov[1], 0.0, cov[1], cov[7], 0.0, 0.0, 0.0, 1.0]
            g.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
            self.pub_geo.publish(g)
        self.pub_slip.publish(Bool(data=bool(o.slip)))
        ac = AccelStamped()
        ac.header.stamp = copy.copy(v.header.stamp)
        ac.header.frame_id = self.child_frame_id
        ac.accel.linear.x = float(o.a)               # продольное ускорение (оценка)
        self.pub_acc.publish(ac)
        self.n_pub += 1
        if t0 is not None:
            self.lat_ms.append((time.perf_counter() - t0) * 1e3)

    def _on_diag(self):
        o = self.last_out
        if o is None:
            return
        lat = sorted(self.lat_ms)
        p50 = lat[len(lat) // 2] if lat else float('nan')
        p99 = lat[min(len(lat) - 1, int(0.99 * len(lat)))] if lat else float('nan')
        self.lat_ms.clear()
        st = DiagnosticStatus()
        st.name = 'tram_backup_odometry'
        st.hardware_id = 'tram'
        st.level = DiagnosticStatus.OK if not (o.slip or o.sensor_fault) and o.wheels_ok else DiagnosticStatus.WARN
        st.message = ('проскальзывание: ' + o.slip_kind) if o.slip else (
            ('сбой датчика: ' + o.sensor_fault) if o.sensor_fault else ('норма' if o.wheels_ok else 'нет колёс — модель'))
        kv = {'mode': o.mode, 'velocity_mps': f'{o.v:.3f}', 'accel_mps2': f'{o.a:.3f}', 'distance_m': f'{o.s:.1f}',
              'slip': str(o.slip), 'slip_kind': o.slip_kind, 'slip_ratio': f'{o.slip_ratio:.3f}',
              'adhesion_used': f'{abs(o.a) / 9.81:.3f}', 'wheels_ok': str(o.wheels_ok),
              'sensor_fault': o.sensor_fault, 'wheel_noise_mps': f'{o.wheel_noise:.3f}',
              'wheel_scale_adj': f'{self.est.scale_adj:.5f}', 'stop_corrections': str(self.est.n_stop_corr),
              'branch_switches': str(self.est.n_fork_switch),
              'sigma_along_m': f'{o.sigma_along:.2f}', 'proc_ms_p50': f'{p50:.3f}', 'proc_ms_p99': f'{p99:.3f}',
              'drive_specific_force_mps2': f'{o.a_drive:.3f}', 'motor_torque_nm': f'{o.torque_nm:.0f}',
              'motor_shaft_rpm': f'{o.shaft_rpm:.0f}', 'map_edge': str(o.map_edge), 'map_s_m': f'{o.map_s:.1f}',
              'published': str(self.n_pub)}
        st.values = [KeyValue(key=k, value=v) for k, v in kv.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = BackupOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

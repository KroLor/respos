"""Замер задержки «вход → результат», частоты публикации и ресурсов ноды оценки.

Задержка: для каждого /result/velocity ищется входное сообщение с тем же header.stamp;
latency = время приёма результата − время приёма входа (стенные часы этого процесса).
Ресурсы (Linux): %CPU и RSS процесса ноды по /proc.

ros2 run tram_backup_odometry latency_monitor [--ros-args -p report_period:=5.0 -p csv:=/tmp/latency.csv]
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tram_vehicle_msgs.msg import DriverControllerCommand, VelocitySensor


def _key(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def _find_pid(pattern='odometry_node'):
    me = os.getpid()
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != me:
            try:
                with open(f'/proc/{d}/cmdline', 'rb') as f:
                    if pattern.encode() in f.read():
                        return int(d)
            except OSError:
                pass
    return None


def _proc_stat(pid):
    with open(f'/proc/{pid}/stat') as f:
        parts = f.read().rsplit(')', 1)[1].split()
    ticks = int(parts[11]) + int(parts[12])
    with open(f'/proc/{pid}/status') as f:
        rss = next((int(line.split()[1]) for line in f if line.startswith('VmRSS')), 0)
    return ticks, rss / 1024.0


class LatencyMonitor(Node):
    def __init__(self):
        super().__init__('latency_monitor')
        self.declare_parameter('report_period', 5.0)
        self.declare_parameter('csv', '')
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.inputs: OrderedDict[int, float] = OrderedDict()
        for t in ('/vehicle/front_bogie_velocity', '/vehicle/rear_bogie_velocity'):
            self.create_subscription(VelocitySensor, t, self._on_in, qos)
        self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self._on_in, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', self._on_out, qos)
        self.create_subscription(Odometry, '/result/position', self._on_pos, qos)
        self.lat, self.all_lat = [], []
        self.n_v = self.n_p = 0
        self.t_start = time.monotonic()
        self.t_rep = self.t_start
        self.pid = None
        self.cpu0 = None
        self.rss_max = 0.0
        self.cpu_hist = []
        self.create_timer(self.get_parameter('report_period').value, self._report)

    def _on_in(self, m):
        k = _key(m.header)
        if k not in self.inputs:
            self.inputs[k] = time.monotonic()
            while len(self.inputs) > 5000:
                self.inputs.popitem(last=False)

    def _on_out(self, m):
        now = time.monotonic()
        self.n_v += 1
        t_in = self.inputs.get(_key(m.header))
        if t_in is not None:
            self.lat.append((now - t_in) * 1e3)

    def _on_pos(self, m):
        self.n_p += 1

    def _report(self):
        now = time.monotonic()
        per = now - self.t_rep
        self.t_rep = now
        lat = sorted(self.lat)
        self.all_lat += self.lat
        self.lat = []
        if self.pid is None:
            self.pid = _find_pid()
        cpu = rss = float('nan')
        if self.pid is not None:
            try:
                ticks, rss = _proc_stat(self.pid)
                if self.cpu0 is not None:
                    cpu = 100.0 * (ticks - self.cpu0[0]) / os.sysconf('SC_CLK_TCK') / (now - self.cpu0[1])
                    self.cpu_hist.append(cpu)
                self.cpu0 = (ticks, now)
                self.rss_max = max(self.rss_max, rss)
            except OSError:
                self.pid = None
        if lat:
            q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
            self.get_logger().info(
                f'velocity {self.n_v / per:5.1f} Гц, position {self.n_p / per:5.1f} Гц | задержка, мс: '
                f'p50 {q(0.5):.2f} p95 {q(0.95):.2f} p99 {q(0.99):.2f} max {lat[-1]:.2f} (n={len(lat)}) | '
                f'CPU {cpu:.1f}% RSS {rss:.1f} МБ')
        else:
            self.get_logger().info(f'нет сопоставленных результатов за {per:.1f} с (ожидание данных)')
        self.n_v = self.n_p = 0

    def summary(self):
        lat = sorted(self.all_lat)
        if not lat:
            return
        q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
        cpu = sum(self.cpu_hist) / len(self.cpu_hist) if self.cpu_hist else float('nan')
        msg = (f'ИТОГ: n={len(lat)}, задержка p50 {q(0.5):.2f} мс, p99 {q(0.99):.2f} мс, max {lat[-1]:.2f} мс, '
               f'>100 мс: {sum(x > 100 for x in lat)}; CPU ср. {cpu:.1f}%; RSS max {self.rss_max:.1f} МБ')
        print(msg, flush=True)
        path = self.get_parameter('csv').value
        if path:
            with open(path, 'w') as f:
                f.write('latency_ms\n' + '\n'.join(f'{x:.4f}' for x in self.all_lat) + '\n')


def main(args=None):
    rclpy.init(args=args)
    n = LatencyMonitor()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.summary()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

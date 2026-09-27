"""Онлайн-оценка точности по GNSS-эталону (только для проверки; в контуре оценки GNSS не используется).

Эталон: base_link по паре антенн status=2 (tf организаторов) в плоских координатах MGRS 37U CB, z — высота рельса;
скорость = |twist| master/vel.
Сопоставление с /result/* по ближайшему header.stamp (±0.05 с), как у жюри.
Печатает RMSE/MAE/bias скорости, ошибку положения (горизонт., 3D, along/cross), итоговый дрейф в % пути.

ros2 run tram_backup_odometry online_eval [--ros-args -p report_period:=10.0 -p csv:=/tmp/eval.csv]
"""
from __future__ import annotations

import bisect
import math

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import VelocitySensor

from .geo import MgrsFrame

TOL = 0.05


def _t(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class _Series:
    """Упорядоченный по времени буфер (t, value) с поиском ближайшего."""

    def __init__(self, maxlen=20000):
        self.t, self.v, self.maxlen = [], [], maxlen

    def add(self, t, v):
        if self.t and t <= self.t[-1]:
            return
        self.t.append(t)
        self.v.append(v)
        if len(self.t) > self.maxlen:
            del self.t[:1000], self.v[:1000]

    def nearest(self, t):
        i = bisect.bisect_left(self.t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= TOL:
                if best is None or abs(self.t[j] - t) < abs(self.t[best] - t):
                    best = j
        return None if best is None else self.v[best]


class OnlineEval(Node):
    def __init__(self):
        super().__init__('online_eval')
        self.declare_parameter('report_period', 10.0)
        self.declare_parameter('csv', '')
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(NavSatFix, '/sensing/gnss/master/fix', lambda m: self._on_fix('master', m), qos)
        self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix', lambda m: self._on_fix('rover', m), qos)
        self.create_subscription(TwistStamped, '/sensing/gnss/master/vel', self._on_vel, qos)
        self.create_subscription(VelocitySensor, '/result/velocity', self._on_rv, qos)
        self.create_subscription(Odometry, '/result/position', self._on_rp, qos)
        self.frame = MgrsFrame()
        self.last_ant = {}
        self.last_pair = None
        self.rv, self.rp = _Series(), _Series()
        self.pend_v, self.pend_p = [], []          # эталон, ждущий выхода оценщика
        self.ev, self.ep = [], []                  # (t, err) / (t, dx, dy, dz, hdg)
        self.last_ref = None
        self.dist = 0.0
        self.hdg = 0.0
        self.create_timer(self.get_parameter('report_period').value, self._report)

    def _on_fix(self, ant, m: NavSatFix):
        if m.status.status != 2:
            return
        t = _t(m.header)
        self.last_ant[ant] = (t, *(float(v) for v in self.frame.to_enu(m.latitude, m.longitude, m.altitude)))
        a, b = self.last_ant.get('master'), self.last_ant.get('rover')
        if a is None or b is None or abs(a[0] - b[0]) > 0.06 or a[0] == self.last_pair:
            return                              # пара антенн с близкими метками, каждая пара — один раз
        self.last_pair = a[0]
        if abs(math.hypot(b[1] - a[1], b[2] - a[2]) - 12.436) > 0.5:
            return
        k = 9.873 / 12.436                      # base_link = master + k·(rover − master), высота − 3.0 м
        e, n, u = a[1] + k * (b[1] - a[1]), a[2] + k * (b[2] - a[2]), a[3] + k * (b[3] - a[3]) - 3.0
        t = a[0]
        if self.last_ref is not None:
            d = math.hypot(e - self.last_ref[1], n - self.last_ref[2])
            dt = t - self.last_ref[0]
            if d > 5.0 * max(dt, 0.1) + 3.0:        # скачок эталона
                return
            if d > 0.3:
                self.hdg = math.atan2(n - self.last_ref[2], e - self.last_ref[1])
            self.dist += d
        self.last_ref = (t, e, n, u)
        self.pend_p.append((t, e, n, u, self.hdg))

    def _on_vel(self, m: TwistStamped):
        self.pend_v.append((_t(m.header), math.hypot(m.twist.linear.x, m.twist.linear.y)))

    def _on_rv(self, m: VelocitySensor):
        self.rv.add(_t(m.header), m.velocity)
        self._flush()

    def _on_rp(self, m: Odometry):
        p = m.pose.pose.position
        self.rp.add(_t(m.header), (p.x, p.y, p.z))
        self._flush()

    def _flush(self):
        now = self.rv.t[-1] if self.rv.t else None
        if now is None:
            return
        keep = []
        for t, v in self.pend_v:
            if t > now - TOL:
                keep.append((t, v))
                continue
            est = self.rv.nearest(t)
            if est is not None:
                self.ev.append((t, est - v))
        self.pend_v = keep
        keep = []
        nowp = self.rp.t[-1] if self.rp.t else now
        for r in self.pend_p:
            if r[0] > nowp - TOL:
                keep.append(r)
                continue
            est = self.rp.nearest(r[0])
            if est is not None:
                self.ep.append((r[0], est[0] - r[1], est[1] - r[2], est[2] - r[3], r[4]))
        self.pend_p = keep

    def _report(self, final=False):
        if not self.ev and not self.ep:
            self.get_logger().info('ожидание данных…')
            return
        vv = [e for _, e in self.ev if abs(e) < 1.0]      # |ошибка|>1 м/с — сбои скорости GNSS
        out = []
        if vv:
            out.append(f'v: RMSE {math.sqrt(sum(e * e for e in vv) / len(vv)):.3f} MAE '
                       f'{sum(abs(e) for e in vv) / len(vv):.3f} bias {sum(vv) / len(vv):+.3f} м/с (n={len(vv)})')
        if self.ep:
            h = [math.hypot(a, b) for _, a, b, _, _ in self.ep]
            d3 = [math.sqrt(a * a + b * b + c * c) for _, a, b, c, _ in self.ep]
            al = [a * math.cos(g) + b * math.sin(g) for _, a, b, _, g in self.ep]
            cr = [-a * math.sin(g) + b * math.cos(g) for _, a, b, _, g in self.ep]
            rm = lambda x: math.sqrt(sum(y * y for y in x) / len(x))  # noqa: E731
            out.append(f'pos: гориз. RMSE {rm(h):.2f} max {max(h):.2f} | 3D RMSE {rm(d3):.2f} | along RMSE {rm(al):.2f} '
                       f'MAE {sum(abs(y) for y in al) / len(al):.2f} max {max(abs(y) for y in al):.2f} | cross RMSE '
                       f'{rm(cr):.2f} | текущая ошибка {h[-1]:.2f} м = {100 * h[-1] / max(self.dist, 1):.3f}% '
                       f'от {self.dist:.0f} м')
        msg = ('ИТОГ ' if final else '') + ' || '.join(out)
        if final:
            print(msg, flush=True)
        else:
            self.get_logger().info(msg)

    def save(self):
        path = self.get_parameter('csv').value
        if path and self.ep:
            with open(path, 'w') as f:
                f.write('t,dx,dy,dz,ref_heading\n')
                for r in self.ep:
                    f.write(','.join(f'{x:.4f}' for x in r) + '\n')


def main(args=None):
    rclpy.init(args=args)
    n = OnlineEval()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n._flush()
        n._report(final=True)
        n.save()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

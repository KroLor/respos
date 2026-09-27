"""Чтение rosbag2 (sqlite3, CDR) без ROS: входы ноды и эталон стенда организаторов.

Разбираются только нужные поля: header.stamp; VelocitySensor.velocity (км/ч, как в топике);
DriverControllerCommand.position; NavSatFix status/latitude/longitude/altitude;
TwistStamped GNSS: горизонтальная скорость;
Odometry (эталон /localization/kinematic_state): положение, курс, продольная скорость.
"""
import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

TOPICS = {
    '/vehicle/front_bogie_velocity': 'front',
    '/vehicle/rear_bogie_velocity': 'rear',
    '/vehicle/driver_position_cmd': 'driver_cmd',
    '/sensing/gnss/master/fix': 'master',
    '/sensing/gnss/rover/fix': 'rover',
    '/sensing/gnss/master/vel': 'master_vel',
    '/sensing/gnss/rover/vel': 'rover_vel',
    '/localization/kinematic_state': 'reference',
}


@dataclass
class Message:
    kind: str           # front, rear, driver_cmd, master, rover
    stamp: float        # header.stamp, с
    value: object       # км/ч | позиция | (status, lat, lon, alt)
    record: float       # время записи в bag, с (порядок воспроизведения)


@dataclass
class Bag:
    messages: List[Message] = field(default_factory=list)
    # эталон: (stamp, x, y, z, продольная скорость, курс)
    reference: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    # скорость GNSS-приёмников: приёмник → [(stamp, горизонтальная скорость, м/с)]
    gnss_speed: dict = field(default_factory=lambda: {'master': [], 'rover': []})


def _header(raw):
    sec, nsec, flen = struct.unpack_from('<iII', raw, 4)
    return sec + nsec * 1e-9, 16 + flen


def _align(off, n):
    return 4 + ((off - 4 + n - 1) // n) * n


def _odometry(raw):
    stamp, off = _header(raw)
    off = _align(off, 4)
    (child_len,) = struct.unpack_from('<I', raw, off)
    off = _align(off + 4 + child_len, 8)
    x, y, z, qx, qy, qz, qw = struct.unpack_from('<7d', raw, off)
    (vx,) = struct.unpack_from('<d', raw, off + 43 * 8)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return (stamp, x, y, z, vx, yaw)


def read_bag(path: Path) -> Bag:
    db = path if path.suffix == '.db3' else next(path.glob('*.db3'))
    con = sqlite3.connect(str(db))
    ids = {i: TOPICS[name] for i, name in con.execute('select id, name from topics') if name in TOPICS}
    bag = Bag()
    query = (f'select topic_id, timestamp, data from messages where topic_id in '
             f'({",".join(map(str, ids))}) order by timestamp')
    for topic_id, record, raw in con.execute(query):
        kind = ids[topic_id]
        if kind == 'reference':
            bag.reference.append(_odometry(raw))
            continue
        stamp, off = _header(raw)
        if kind.endswith('_vel'):
            vx, vy = struct.unpack_from('<2d', raw, _align(off, 8))
            bag.gnss_speed[kind[:-4]].append((stamp, math.hypot(vx, vy)))
            continue
        if kind == 'driver_cmd':
            value = struct.unpack_from('<b', raw, off)[0]
        elif kind in ('master', 'rover'):
            status = struct.unpack_from('<b', raw, off)[0]
            lat, lon, alt = struct.unpack_from('<3d', raw, _align(off + 4, 8))
            value = (status, lat, lon, alt)
        else:
            value = struct.unpack_from('<d', raw, _align(off, 8))[0]
        bag.messages.append(Message(kind, stamp, value, record * 1e-9))
    bag.reference.sort()
    for speeds in bag.gnss_speed.values():
        speeds.sort()
    return bag

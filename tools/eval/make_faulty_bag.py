"""Копия bag с внесёнными сбоями входных данных — для проверки устойчивости ноды.

Запуск в ROS 2 Humble (после source install/setup.bash workspace с tram_vehicle_msgs):
  python3 tools/eval/make_faulty_bag.py <исходный_bag> <новый_bag>
Окна сбоев — секунды от начала ЗАПИСИ bag (время записи, не header.stamp); рассчитаны на
прогон длиннее 215 с (например, 30618_af7496f0). Сравнивать выход с прогоном по исходному bag.
"""
import math
import sys
from collections import Counter

FRONT = '/vehicle/front_bogie_velocity'
REAR = '/vehicle/rear_bogie_velocity'
CMD = '/vehicle/driver_position_cmd'

FAULTS = [  # (начало, конец, описание)
    (40, 45, 'front NaN'),
    (60, 80, 'rear всплески +40 км/ч'),
    (100, 103, 'колёса молчат'),
    (120, 122, 'контроллер молчит'),
    (140, 145, 'контроллер: дубли'),
    (150, 152, 'front: нулевые метки'),
    (160, 161.5, 'контроллер: метки ±1 ч'),
    (170, 172.5, 'вне диапазона'),
    (180, 200, 'rear залип на 0'),
    (210, 215, 'front inf'),
]


def main(src, dst):
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=src, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=dst, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    metas = reader.get_all_topics_and_types()
    types = {m.name: m.type for m in metas}
    for meta in metas:
        writer.create_topic(meta)

    t0 = None
    done = Counter()
    rear_count = 0
    once = set()
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t0 = t0 if t0 is not None else t_ns
        t = (t_ns - t0) * 1e-9
        if topic not in (FRONT, REAR, CMD):
            writer.write(topic, data, t_ns)
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        copies = 1
        if topic == FRONT and 40 <= t < 45:
            msg.velocity = math.nan
            done['front NaN'] += 1
        if topic == REAR and 60 <= t < 80:
            rear_count += 1
            if rear_count % 10 == 0:
                msg.velocity += 40.0
                done['rear всплеск'] += 1
        if topic in (FRONT, REAR) and 100 <= t < 103:
            done['колёса удалены'] += 1
            continue
        if topic == CMD and 120 <= t < 122:
            done['контроллер удалён'] += 1
            continue
        if topic == CMD and 140 <= t < 145:
            copies = 2
            done['контроллер дубль'] += 1
        if topic == FRONT and 150 <= t < 152:
            msg.header.stamp.sec, msg.header.stamp.nanosec = 0, 0
            done['front нулевая метка'] += 1
        if topic == CMD and t >= 160 and 'future' not in once:
            msg.header.stamp.sec += 3600
            once.add('future')
            done['метка +1 ч'] += 1
        if topic == CMD and t >= 161 and 'past' not in once:
            msg.header.stamp.sec -= 3600
            once.add('past')
            done['метка -1 ч'] += 1
        if topic == CMD and t >= 170 and 'pos' not in once:
            msg.position = 99
            once.add('pos')
            done['ручка 99'] += 1
        if topic == FRONT and t >= 171 and 'fast' not in once:
            msg.velocity = 500.0
            once.add('fast')
            done['front 500 км/ч'] += 1
        if topic == REAR and t >= 172 and 'neg' not in once:
            msg.velocity = -50.0
            once.add('neg')
            done['rear -50 км/ч'] += 1
        if topic == REAR and 180 <= t < 200:
            msg.velocity = 0.0
            done['rear залип'] += 1
        if topic == FRONT and 210 <= t < 215:
            msg.velocity = math.inf
            done['front inf'] += 1
        for _ in range(copies):
            writer.write(topic, serialize_message(msg), t_ns)

    del writer
    print('Внесено:', dict(done))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

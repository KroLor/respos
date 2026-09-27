"""Копия bag «как на проверке жюри»: GNSS только в первые секунды записи.

Запуск в WSL (после source ~/respos_ws/install/setup.bash):
  python3 make_testlike_bag.py <исходный_bag> <новый_bag> [секунд GNSS, по умолчанию 5]
Организаторы: в проверочных bag GNSS гарантирован только в первые несколько секунд.
"""
import sys

import rosbag2_py

GNSS_PREFIX = '/sensing/gnss/'


def main(src, dst, seconds=5.0):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=src, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=dst, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for meta in reader.get_all_topics_and_types():
        writer.create_topic(meta)
    t0, kept, dropped = None, 0, 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t0 = t0 if t0 is not None else t_ns
        if topic.startswith(GNSS_PREFIX) and (t_ns - t0) * 1e-9 > seconds:
            dropped += 1
            continue
        writer.write(topic, data, t_ns)
        kept += 1
    del writer
    print(f'Записано {kept} сообщений, GNSS после {seconds:.0f} с убрано: {dropped}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 5.0)

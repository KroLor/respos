"""Разбор прогона ноды: выход, ошибки относительно GNSS, окна сбоев, диагностика.

Запуск в WSL (после source ~/respos_ws/install/setup.bash):
  python3 eval_run.py <исходный_bag> <bag_результата> [bag_результата_чистого_прогона] [--json файл]
Третий аргумент — для прогона на испорченной копии (make_faulty_bag.py): отличие выхода
от чистого прогона по окнам сбоев и через 5 с после них. --json — сохранить метрики
(их проверяет global_test.py).

Сопоставление с эталоном — как у судьи: по ближайшей метке с допуском 0,05 с. Эталон —
GNSS master, если есть, иначе rover. Положение сравнивается в локальной ENU с началом
в первой точке эталона (так же, как предполагаем у судьи). Это временный инструмент;
полноценный оценщик метрик — этап 4 плана.
"""
import bisect
import json
import math
import os
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tram_odometry.geo import LocalEnu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))
from make_faulty_bag import FAULTS  # noqa: E402

LEVELS = {0: 'OK', 1: 'WARN', 2: 'ERROR', 3: 'STALE'}


def read(uri, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, recv = reader.read_next()
        if topic in types:
            out[topic].append((recv * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def nearest(stamps, t, tol=0.05):
    i = bisect.bisect_left(stamps, t)
    cands = [j for j in (i - 1, i) if 0 <= j < len(stamps) and abs(stamps[j] - t) <= tol]
    return min(cands, key=lambda j: abs(stamps[j] - t)) if cands else None


def rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else None


def stats(errors, unit='м/с'):
    if not errors:
        return 'нет пар'
    return (f'RMSE {rmse(errors):.3f}, bias {sum(errors) / len(errors):+.3f}, '
            f'макс |e| {max(map(abs, errors)):.2f} {unit} (пар {len(errors)})')


def level_value(level):
    return int.from_bytes(level, 'little') if isinstance(level, bytes) else int(level)


def trapezoid(series):
    return sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(series, series[1:]))


def main(src_bag, res_bag, clean_bag=None):
    m = {}
    res = read(res_bag, ['/result/velocity', '/result/position',
                         '/result/slip_detected', '/result/slip_ratio', '/diagnostics'])
    vel = [(sec(msg.header.stamp), msg.velocity) for _, msg in res['/result/velocity']]
    pos = [(sec(msg.header.stamp), msg.pose.pose.position) for _, msg in res['/result/position']]
    m['n_out'] = len(vel)
    if not vel:
        print('В результате нет /result/velocity — нода ничего не опубликовала')
        return m
    src = read(src_bag, ['/vehicle/driver_position_cmd',
                         '/sensing/gnss/master/vel', '/sensing/gnss/rover/vel',
                         '/sensing/gnss/master/fix', '/sensing/gnss/rover/fix'])
    cmd = src['/vehicle/driver_position_cmd']
    # Время отсчитываем от начала записи bag (как окна сбоев в make_faulty_bag.py)
    t_start = cmd[0][0]
    m['n_cmd'] = len(cmd)
    m['out_per_cmd'] = len(vel) / len(cmd)

    stamps = [s for s, _ in vel]
    gaps = sorted(((b - a, a - t_start) for a, b in zip(stamps, stamps[1:])), reverse=True)[:3]
    m['back'] = sum(1 for a, b in zip(stamps, stamps[1:]) if b <= a)
    m['rate'] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    m['max_gap'] = gaps[0][0] if gaps else 0.0
    print(f'Выход: {len(vel)} сообщ. ({100 * m["out_per_cmd"]:.1f} % от сообщений контроллера), '
          f'первая метка t+{stamps[0] - t_start:.2f} c, последняя t+{stamps[-1] - t_start:.1f} c; '
          f'{m["rate"]:.2f} Гц по меткам; невозрастающих: {m["back"]}')
    print('   самые большие разрывы меток:',
          ', '.join(f'{g:.2f} c (на t+{t:.1f})' for g, t in gaps))
    slips = [msg.data for _, msg in res['/result/slip_detected']]
    m['slip_count'] = sum(slips)
    if slips:
        print(f'   флаг проскальзывания: {sum(slips)} из {len(slips)} сообщений')
    ratios = sorted(abs(msg.data) for _, msg in res['/result/slip_ratio'])
    m['slip_ratio_count'] = len(ratios)
    if ratios:
        m['slip_ratio_p99'] = ratios[min(len(ratios) - 1, int(0.99 * len(ratios)))]
        m['slip_ratio_max'] = ratios[-1]
        print(f'   скольжение: {len(ratios)} сообщ., |s| p99 {m["slip_ratio_p99"]:.3f}, макс {m["slip_ratio_max"]:.2f}')

    source = 'master' if src['/sensing/gnss/master/vel'] else 'rover'
    gnss_vel = src[f'/sensing/gnss/{source}/vel']
    gnss_fix = [(sec(msg.header.stamp), msg) for _, msg in src[f'/sensing/gnss/{source}/fix']]
    if not gnss_vel:
        print('GNSS в этом bag нет — сравнение с эталоном невозможно')
    else:
        ref = [(sec(msg.header.stamp), math.hypot(msg.twist.linear.x, msg.twist.linear.y))
               for _, msg in gnss_vel]
        errors = []
        for t, v_ref in ref:
            j = nearest(stamps, t)
            if j is not None:
                errors.append(vel[j][1] - v_ref)
        m.update(speed_rmse=rmse(errors), speed_bias=sum(errors) / len(errors),
                 speed_mae=sum(map(abs, errors)) / len(errors))
        print(f'Скорость относительно GNSS ({source}):', stats(errors))
        ref_dist, out_dist = trapezoid(ref), trapezoid(vel)
        m['path_pct'] = 100 * (out_dist - ref_dist) / ref_dist if ref_dist > 0 else None
        print(f'Путь (интеграл скорости): эталон {ref_dist:.1f} м, выход {out_dist:.1f} м'
              + (f' ({m["path_pct"]:+.2f} %)' if m['path_pct'] is not None else ''))
        if pos and abs(pos[0][1].x) > 1000.0:
            # Положение в плоских MGRS (backup_model): эталон — base_link по паре антенн status=2
            from tram_backup_odometry.geo import MgrsFrame
            frame = MgrsFrame()
            master = [(sec(msg.header.stamp), msg) for _, msg in src['/sensing/gnss/master/fix']
                      if msg.status.status >= 2]
            rover = [(sec(msg.header.stamp), msg) for _, msg in src['/sensing/gnss/rover/fix']
                     if msg.status.status >= 2]
            rover.sort(key=lambda item: item[0])
            rover_stamps = [t for t, _ in rover]
            pos_stamps = [s for s, _ in pos]
            dists = []
            for t, fm in master:
                k = nearest(rover_stamps, t, tol=0.06)
                j = nearest(pos_stamps, t)
                if k is None or j is None:
                    continue
                fr = rover[k][1]
                me, mn, mu = frame.to_enu(fm.latitude, fm.longitude, fm.altitude)
                re_, rn, ru = frame.to_enu(fr.latitude, fr.longitude, fr.altitude)
                w = 9.873 / 12.436
                ref_p = (me + w * (re_ - me), mn + w * (rn - mn), mu + w * (ru - mu) - 3.0)
                p = pos[j][1]
                dists.append(math.dist((p.x, p.y, p.z), ref_p))
            if dists:
                m.update(pos_mean=sum(dists) / len(dists), pos_max=max(dists), pos_final=dists[-1])
                print(f'Положение (3D, MGRS base_link по паре антенн): среднее {m["pos_mean"]:.2f} м, '
                      f'макс {m["pos_max"]:.2f} м, в конце {m["pos_final"]:.2f} м (пар {len(dists)})')
        elif gnss_fix and pos:
            first = gnss_fix[0][1]
            enu = LocalEnu(first.latitude, first.longitude, first.altitude)
            pos_stamps = [s for s, _ in pos]
            dists = []
            for t, fix in gnss_fix:
                j = nearest(pos_stamps, t)
                if j is not None:
                    e, n, u = enu.to_enu(fix.latitude, fix.longitude, fix.altitude)
                    p = pos[j][1]
                    dists.append(math.dist((p.x, p.y, p.z), (e, n, u)))
            if dists:
                m.update(pos_mean=sum(dists) / len(dists), pos_max=max(dists), pos_final=dists[-1])
                print(f'Положение (3D, ENU от первой точки эталона): среднее {m["pos_mean"]:.2f} м, '
                      f'макс {m["pos_max"]:.2f} м, в конце {m["pos_final"]:.2f} м (пар {len(dists)})')

    if clean_bag:
        clean = read(clean_bag, ['/result/velocity'])['/result/velocity']
        clean_vel = [(sec(msg.header.stamp), msg.velocity) for _, msg in clean]
        clean_stamps = [s for s, _ in clean_vel]
        diffs = []   # (время, выход − выход чистого прогона) по совпадающим меткам
        for s, v in vel:
            j = nearest(clean_stamps, s, tol=0.001)
            if j is not None:
                diffs.append((s - t_start, v - clean_vel[j][1]))
        m['faults'] = []
        print('Отличие от чистого прогона по окнам сбоев:')
        for a, b, name in FAULTS:
            inside = [e for t, e in diffs if a <= t < b + 0.5]
            after = [e for t, e in diffs if b + 0.5 <= t < b + 5.5]
            m['faults'].append({'name': name,
                                'inside_max': max(map(abs, inside)) if inside else None,
                                'after_max': max(map(abs, after)) if after else None})
            print(f'   {a:>3}–{b:<5} {name:24s} внутри: {stats(inside)}')
            print(f'   {"":9} {"":24s} 5 с после: {stats(after)}')
        d_clean, d_this = trapezoid(clean_vel), trapezoid(vel)
        m['path_diff_pct'] = 100 * (d_this - d_clean) / d_clean
        m['out_vs_clean'] = len(vel) / len(clean_vel)
        print(f'Путь: чистый прогон {d_clean:.1f} м, этот {d_this:.1f} м, '
              f'разница {d_this - d_clean:+.1f} м ({m["path_diff_pct"]:+.2f} %)')

    diags = res['/diagnostics']
    m['diag_errors'] = sum(1 for _, d in diags for s in d.status if level_value(s.level) == 2)
    if diags:
        print(f'Диагностика: {len(diags)} сообщений (статусов ERROR за прогон: {m["diag_errors"]}); последнее:')
        for status in diags[-1][1].status:
            keep = [f'{kv.key}={kv.value}' for kv in status.values
                    if kv.value not in ('0', '-', '0.00') and not kv.key.startswith(('возраст', 'скольжение'))]
            print(f'   [{LEVELS.get(level_value(status.level), "?")}] {status.name}: {status.message}\n      '
                  + '; '.join(keep))
    return m


if __name__ == '__main__':
    args = sys.argv[1:]
    json_path = None
    if '--json' in args:
        i = args.index('--json')
        json_path = args[i + 1]
        del args[i:i + 2]
    metrics = main(*args[:3])
    if json_path:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1)

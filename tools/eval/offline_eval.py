"""Офлайн-оценка решения по bag датасета (без ROS): скорость и положение в системе судьи.

Сообщения каждого bag в порядке записи проходят через ядро ноды (tram_odometry.core.OdometryCore,
тот же код, что в ROS). GNSS ядру подаётся «как на проверке» (--gnss):
  check (по умолчанию) — первые 30 с и всплески по 2 с раз в 150 с, как в bag стенда организаторов;
  first — только первые 5 с; all — весь GNSS.
Эталон: /localization/kinematic_state, если он есть в bag; иначе положение base_link по паре антенн
со status ≥ 2 (base_link = master + 0,794·(rover − master), высота −3 м; плоские MGRS, как у судьи)
и горизонтальная скорость GNSS master (иначе rover). Сопоставление по метке ±0,05 с.

Нужны Python 3.10+ и numpy. Запуск из корня репозитория:
  python tools/eval/offline_eval.py [bag ...] [--data каталог] [--gnss check|first|all]
         [--estimator backup_model] [--json файл] [-j процессов]
"""
import argparse
import bisect
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Резервное позиционирование' / 'data'
sys.path.insert(0, str(ROOT / 'tools' / 'eval'))
for package in ('tram_odometry', 'tram_backup_odometry'):
    sys.path.insert(0, str(ROOT / 'src' / package))
from bagread import read_bag  # noqa: E402

MASTER_X, ROVER_X, ANTENNA_Z = -9.873, 2.563, 3.0     # tf антенн в base_link (организаторы)
PAIR_TOLERANCE = 0.06            # с: точки master и rover — одна пара
SYNC_TOLERANCE = 0.05            # с: как у судьи
GNSS_POLICIES = {
    'first': lambda t: t <= 5.0,
    'check': lambda t: t <= 30.0 or (t - 30.0) % 150.0 < 2.0,
    'all': lambda t: True,
}


def gnss_reference(bag):
    """Эталон положения по парам антенн: [(stamp, x, y, z, курс)] в плоских MGRS."""
    from tram_backup_odometry.geo import MgrsFrame
    frame = MgrsFrame()
    fixes = {'master': [], 'rover': []}
    for m in bag.messages:
        if m.kind in fixes and m.value[0] >= 2:
            fixes[m.kind].append((m.stamp, *frame.to_enu(*m.value[1:])))
    rover = sorted(fixes['rover'])
    rover_t = [r[0] for r in rover]
    weight = -MASTER_X / (ROVER_X - MASTER_X)
    out = []
    for t, me, mn, mu in sorted(fixes['master']):
        i = bisect.bisect_left(rover_t, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(rover) and abs(rover_t[j] - t) <= PAIR_TOLERANCE]
        if not near:
            continue
        _, re_, rn, ru = rover[min(near, key=lambda j: abs(rover_t[j] - t))]
        if abs(math.hypot(re_ - me, rn - mn) - (ROVER_X - MASTER_X)) > 1.5:
            continue                        # пара антенн не согласована (многолучёвость, скачок)
        out.append((t, me + weight * (re_ - me), mn + weight * (rn - mn),
                    mu + weight * (ru - mu) - ANTENNA_Z, math.atan2(rn - mn, re_ - me)))
    return out


def nearest(stamps, t):
    i = bisect.bisect_left(stamps, t)
    near = [j for j in (i - 1, i) if 0 <= j < len(stamps) and abs(stamps[j] - t) <= SYNC_TOLERANCE]
    return min(near, key=lambda j: abs(stamps[j] - t)) if near else None


def evaluate(args):
    """Один bag → метрики (или None, если эталона нет)."""
    path, gnss, estimator = args
    from tram_odometry.core import CoreParams, OdometryCore
    from tram_odometry.preprocessing import PreprocessingParams
    bag = read_bag(Path(path))
    if bag.reference:
        position_ref = [(r[0], r[1], r[2], r[3], r[5]) for r in bag.reference]
        speed_ref = [(r[0], r[4]) for r in bag.reference]
    else:
        position_ref = gnss_reference(bag)
        speed_ref = bag.gnss_speed['master'] or bag.gnss_speed['rover']
    if len(position_ref) < 10 or not speed_ref:
        return None
    core = OdometryCore(PreprocessingParams(), CoreParams(estimator=estimator), clock=lambda: 0.0)
    allowed = GNSS_POLICIES[gnss]
    start = bag.messages[0].record
    out = []
    for m in bag.messages:
        o = None
        if m.kind == 'driver_cmd':
            o = core.on_driver_cmd(m.stamp, m.value)
        elif m.kind in ('front', 'rear'):
            o = core.on_wheel(m.kind, m.stamp, m.value)
        elif allowed(m.record - start):
            core.on_gnss_fix(m.kind, m.stamp, *m.value)
        if o is not None:
            out.append((o.stamp, o.estimate.velocity, o.x, o.y, o.z, o.position_valid))
    stamps = [o[0] for o in out]
    speed_err = []
    for t, v in speed_ref:
        j = nearest(stamps, t)
        if j is not None:
            speed_err.append(out[j][1] - v)
    valid = [o for o in out if o[5]]
    valid_t = [o[0] for o in valid]
    d3, along, travelled = [], [], 0.0
    prev = None
    for t, x, y, z, yaw in position_ref:
        if prev is not None:
            travelled += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        j = nearest(valid_t, t)
        if j is None:
            continue
        dx, dy, dz = valid[j][2] - x, valid[j][3] - y, valid[j][4] - z
        d3.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        along.append(dx * math.cos(yaw) + dy * math.sin(yaw))
    if not d3 or not speed_err:
        return None
    rms = lambda values: math.sqrt(sum(v * v for v in values) / len(values))  # noqa: E731
    return {'bag': Path(path).name, 'duration': stamps[-1] - stamps[0] if stamps else 0.0,
            'path': travelled, 'n_speed': len(speed_err), 'speed_sq': sum(e * e for e in speed_err),
            'speed_sum': sum(speed_err), 'speed_rmse': rms(speed_err), 'rmse_3d': rms(d3),
            'max_3d': max(d3), 'end_3d': d3[-1], 'along_rmse': rms(along),
            'end_pct': 100.0 * d3[-1] / travelled if travelled > 100.0 else None,
            'rate': (len(out) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0,
            'model_failures': sum(core.estimator.failures.values())}


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('bags', nargs='*')
    parser.add_argument('--data', default=str(DATA))
    parser.add_argument('--gnss', choices=sorted(GNSS_POLICIES), default='check')
    parser.add_argument('--estimator', default='backup_model')
    parser.add_argument('--json')
    parser.add_argument('-j', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    bags = args.bags or sorted(str(p) for p in Path(args.data).iterdir() if p.is_dir())
    with Pool(args.j) as pool:
        results = [r for r in pool.map(evaluate, [(b, args.gnss, args.estimator) for b in bags]) if r]
    if not results:
        sys.exit('Нет bag с эталоном')
    n = sum(r['n_speed'] for r in results)
    summary = {
        'bags': len(results), 'hours': sum(r['duration'] for r in results) / 3600.0,
        'km': sum(r['path'] for r in results) / 1000.0,
        'speed_rmse': math.sqrt(sum(r['speed_sq'] for r in results) / n),
        'speed_bias': sum(r['speed_sum'] for r in results) / n,
        'rmse_3d_median': quantile([r['rmse_3d'] for r in results], 0.5),
        'rmse_3d_p90': quantile([r['rmse_3d'] for r in results], 0.9),
        'max_3d_median': quantile([r['max_3d'] for r in results], 0.5),
        'end_pct_median': quantile([r['end_pct'] for r in results if r['end_pct'] is not None], 0.5),
        'rate_min': min(r['rate'] for r in results),
        'model_failures': sum(r['model_failures'] for r in results),
    }
    print(f"Оценщик {args.estimator}, GNSS: {args.gnss}; прогонов {summary['bags']}, "
          f"{summary['hours']:.1f} ч, {summary['km']:.0f} км")
    print(f"Скорость: RMSE {summary['speed_rmse']:.3f} м/с, bias {summary['speed_bias']:+.3f}")
    print(f"Положение 3D RMSE по прогонам: медиана {summary['rmse_3d_median']:.2f} м, "
          f"p90 {summary['rmse_3d_p90']:.2f}; max — медиана {summary['max_3d_median']:.2f} м; "
          f"ошибка в конце / путь — медиана {summary['end_pct_median']:.3f} %")
    print(f"Частота выхода: мин {summary['rate_min']:.1f} Гц; сбоев модели: {summary['model_failures']}")
    worst = sorted(results, key=lambda r: -r['rmse_3d'])[:5]
    print('Худшие по 3D RMSE: ' + ', '.join(f"{r['bag']} {r['rmse_3d']:.1f} м" for r in worst))
    if args.json:
        Path(args.json).write_text(json.dumps({'summary': summary, 'bags': results}, ensure_ascii=False,
                                              indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

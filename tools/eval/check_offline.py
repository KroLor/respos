"""Офлайн-копия стенда организаторов (check-code) — за секунды вместо 22 минут воспроизведения.

Прогоняет сообщения bag в порядке записи (как `ros2 bag play`) через ядро ноды
tram_odometry.core.OdometryCore — тот же код, что работает в ROS, — и считает те же метрики,
что hackathon_solution_checker: /result/velocity против twist.twist.linear.x и
/result/position против pose.pose.position эталона /localization/kinematic_state,
сопоставление по header.stamp с допуском 0,05 с. Дополнительно — ошибки вдоль и поперёк пути
и максимум по минутам (где копится ошибка).

Нужны только Python 3.10+ и numpy (без ROS). Запуск из корня репозитория:
  python tools/eval/check_offline.py [bag] [--estimator backup_model] [--no-gnss-correction]
         [--gnss all|first] [--minutes] [--json файл]
bag по умолчанию — check-code/bags/30618_88aea4d9 (эталон есть только в bag стенда).
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'eval'))
from bagread import read_bag  # noqa: E402

for package in ('tram_odometry', 'tram_backup_odometry'):
    sys.path.insert(0, str(ROOT / 'src' / package))
from tram_odometry.core import CoreParams, OdometryCore  # noqa: E402
from tram_odometry.preprocessing import PreprocessingParams  # noqa: E402

SYNC_TOLERANCE = 0.05            # с, как у hackathon_solution_checker
GNSS_FIRST_SECONDS = 30.0        # --gnss first: GNSS только в начале записи


def run_core(messages, params: CoreParams, gnss: str):
    """Сообщения bag → выходы ядра: (метка, скорость, x, y, z, положение известно)."""
    core = OdometryCore(PreprocessingParams(), params, clock=lambda: 0.0)
    start = messages[0].record if messages else 0.0
    out = []
    for m in messages:
        o = None
        if m.kind == 'driver_cmd':
            o = core.on_driver_cmd(m.stamp, m.value)
        elif m.kind in ('front', 'rear'):
            o = core.on_wheel(m.kind, m.stamp, m.value)
        elif m.kind in ('master', 'rover'):
            if gnss == 'all' or m.record - start <= GNSS_FIRST_SECONDS:
                status, lat, lon, alt = m.value
                core.on_gnss_fix(m.kind, m.stamp, status, lat, lon, alt)
        if o is not None:
            out.append((o.stamp, o.estimate.velocity, o.x, o.y, o.z, o.position_valid))
    return out, core


def match(results, reference):
    """Пары (результат, эталон) по ближайшей метке в пределах допуска; эталон — не больше одного раза."""
    ref_t = [r[0] for r in reference]
    used = set()
    pairs = []
    for res in results:
        i = bisect.bisect_left(ref_t, res[0])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ref_t) and j not in used and abs(ref_t[j] - res[0]) <= SYNC_TOLERANCE:
                if best is None or abs(ref_t[j] - res[0]) < abs(ref_t[best] - res[0]):
                    best = j
        if best is not None:
            used.add(best)
            pairs.append((res, reference[best]))
    return pairs


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float('nan')


def report(results, reference, minutes=False):
    velocity_pairs = match(results, reference)
    position_pairs = match([r for r in results if r[5]], reference)
    ev = [res[1] - ref[4] for res, ref in velocity_pairs]
    dx = [res[2] - ref[1] for res, ref in position_pairs]
    dy = [res[3] - ref[2] for res, ref in position_pairs]
    dz = [res[4] - ref[3] for res, ref in position_pairs]
    d3 = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(dx, dy, dz)]
    along, cross = [], []
    for (res, ref), ex, ey in zip(position_pairs, dx, dy):
        c, s = math.cos(ref[5]), math.sin(ref[5])
        along.append(ex * c + ey * s)
        cross.append(-ex * s + ey * c)
    print(f'Скорость, м/с: RMSE {rms(ev):.3f}, max {max(map(abs, ev)):.3f}, n={len(ev)}')
    for name, values in (('x', dx), ('y', dy), ('z', dz), ('3D', d3)):
        print(f'Положение {name:>2}, м: RMSE {rms(values):.3f}, max {max(map(abs, values)):.3f}')
    print(f'Вдоль пути, м: RMSE {rms(along):.3f}, max {max(map(abs, along)):.3f}; '
          f'поперёк: RMSE {rms(cross):.3f}, max {max(map(abs, cross)):.3f}; '
          f'n={len(d3)}, ошибка в конце {d3[-1]:.2f} м')
    if minutes:
        t0 = position_pairs[0][1][0]
        buckets = {}
        for (res, ref), err, a, c in zip(position_pairs, d3, along, cross):
            b = buckets.setdefault(int((ref[0] - t0) // 60), [0.0, 0.0, 0.0])
            b[0], b[1], b[2] = max(b[0], err), max(b[1], abs(a)), max(b[2], abs(c))
        for k in sorted(buckets):
            b = buckets[k]
            print(f'  {60 * k:5d} с: max 3D {b[0]:6.2f}, |вдоль| {b[1]:6.2f}, |поперёк| {b[2]:5.2f}')
    return {'velocity_rmse': rms(ev), 'velocity_max': max(map(abs, ev)), 'n_velocity': len(ev),
            'rmse_3d': rms(d3), 'max_3d': max(d3), 'rmse_x': rms(dx), 'rmse_y': rms(dy),
            'rmse_z': rms(dz), 'along_rmse': rms(along), 'cross_rmse': rms(cross),
            'n_position': len(d3), 'end_3d': d3[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('bag', nargs='?', default=str(ROOT / 'check-code' / 'bags' / '30618_88aea4d9'))
    parser.add_argument('--estimator', default='backup_model')
    parser.add_argument('--no-gnss-correction', action='store_true')
    parser.add_argument('--gnss', choices=('all', 'first'), default='all',
                        help='all — все точки GNSS bag (как у жюри); first — только первые 30 с')
    parser.add_argument('--minutes', action='store_true', help='максимум ошибки по минутам')
    parser.add_argument('--json', help='записать метрики в файл JSON')
    args = parser.parse_args()

    bag = read_bag(Path(args.bag))
    if not bag.reference:
        sys.exit('В bag нет эталона /localization/kinematic_state')
    params = CoreParams(estimator=args.estimator, gnss_correction=not args.no_gnss_correction,
                        route_map_file=str(ROOT / 'src' / 'tram_odometry' / 'config' / 'route_map.csv'),
                        route_stops_file=str(ROOT / 'src' / 'tram_odometry' / 'config' / 'route_stops.csv'))
    results, core = run_core(bag.messages, params, args.gnss)
    est = core.estimator
    print(f'{Path(args.bag).name}: оценщик {est.primary.name}, активен в конце {est.active}, '
          f'переключений {est.switches}, сбоев {dict(est.failures)}')
    for key, value in est.primary.diagnostics().items():
        print(f'  {key}: {value}')
    metrics = report(results, bag.reference, args.minutes)
    if args.json:
        metrics.update(estimator=est.primary.name, active=est.active,
                       failures=sum(est.failures.values()))
        Path(args.json).write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

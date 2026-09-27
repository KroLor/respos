"""Устойчивость к сбоям входных данных: bag и его испорченная копия (make_faulty_bag.py) через ядро ноды.

Офлайн и в порядке записи — детерминированно (в ROS порядок доставки разных топиков от прогона
к прогону немного разный, и сравнение двух ROS-прогонов зашумлено само по себе). Для каждого окна
сбоя (FAULTS) — наибольшее отличие скорости от прогона по исходному bag внутри окна и через 5–10 с
после него; путь в конце.

  python3 tools/eval/fault_eval.py <исходный bag> <испорченный bag> [--estimator backup_model] [--json файл]
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'eval'))
for package in ('tram_odometry', 'tram_backup_odometry'):
    sys.path.insert(0, str(ROOT / 'src' / package))
from bagread import read_bag  # noqa: E402
from make_faulty_bag import FAULTS  # noqa: E402


def run(path, estimator):
    """Выходы ядра: (метка, скорость, путь, время записи от начала bag)."""
    from tram_odometry.core import CoreParams, OdometryCore
    from tram_odometry.preprocessing import PreprocessingParams
    bag = read_bag(Path(path))
    core = OdometryCore(PreprocessingParams(), CoreParams(estimator=estimator), clock=lambda: 0.0)
    start = bag.messages[0].record
    out = []
    for m in bag.messages:
        o = None
        if m.kind == 'driver_cmd':
            o = core.on_driver_cmd(m.stamp, m.value)
        elif m.kind in ('front', 'rear'):
            o = core.on_wheel(m.kind, m.stamp, m.value)
        else:
            core.on_gnss_fix(m.kind, m.stamp, *m.value)
        if o is not None:
            out.append((o.stamp, o.estimate.velocity, o.estimate.distance, m.record - start))
    return out, core


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('clean')
    parser.add_argument('faulty')
    parser.add_argument('--estimator', default='backup_model')
    parser.add_argument('--json')
    args = parser.parse_args()
    clean, _ = run(args.clean, args.estimator)
    faulty, core = run(args.faulty, args.estimator)
    stamps = [c[0] for c in clean]
    windows = []
    for begin, end, name in FAULTS:
        inside = after = 0.0
        for stamp, v, _, record in faulty:
            i = bisect.bisect_left(stamps, stamp)
            if i < len(stamps) and abs(stamps[i] - stamp) < 1e-3:
                diff = abs(v - clean[i][1])
                if begin <= record <= end:
                    inside = max(inside, diff)
                elif end + 5.0 <= record <= end + 10.0:
                    after = max(after, diff)
        windows.append({'name': name, 'inside_max': inside, 'after_max': after})
        print(f'{begin:5.0f}–{end:<5g} {name:28s} внутри {inside:6.3f} м/с, через 5–10 с {after:6.3f} м/с')
    path_pct = 100.0 * (faulty[-1][2] - clean[-1][2]) / clean[-1][2]
    failures = sum(core.estimator.failures.values())
    print(f'Путь: {clean[-1][2]:.1f} → {faulty[-1][2]:.1f} м ({path_pct:+.2f} %); '
          f'выходов {len(clean)} → {len(faulty)}; сбоев оценщика {failures}')
    if args.json:
        Path(args.json).write_text(json.dumps({'faults': windows, 'path_diff_pct': path_pct,
                                               'failures': failures}, ensure_ascii=False, indent=1),
                                   encoding='utf-8')


if __name__ == '__main__':
    main()

"""Golden-данные для регрессионного теста ядра: входы bag стенда организаторов (30618_88aea4d9, 1309 с)
в порядке воспроизведения + выходы текущей версии ядра на каждое сообщение.

test/test_golden.py воспроизводит входы и сверяет выходы (скорость, положение, режим) — так проверяется,
что ядро, перенесённое в другое окружение или рантайм, считает то же самое, без датасета и без ROS.
Пересоздавать только при осознанном изменении алгоритма (порядок — docs/INTEGRATION.md, §5).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/make_golden.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
import common  # noqa: E402
from common import events, make_estimator  # noqa: E402

OUT = ROOT / 'src' / 'tram_backup_odometry' / 'test' / 'data' / 'golden_88aea4d9.npz'
KINDS = ['cmd', 'front', 'rear', 'master', 'rover']


def main():
    common.EX = ROOT / 'extracted_check'
    ev = events(common.load_run('30618_88aea4d9'))
    kind = np.array([KINDS.index(k) for _, k, _, _ in ev], np.int8)
    stamp = np.array([s for _, _, s, _ in ev], np.float64)
    val = np.full((len(ev), 4), np.nan)
    for i, (_, k, _, v) in enumerate(ev):
        val[i] = v if k in ('master', 'rover') else (v, np.nan, np.nan, np.nan)
    est = make_estimator()
    out = np.full((len(ev), 6), np.nan)          # stamp, v, e, n, u, mode(0 нет выхода)
    modes = ['pre_init', 'map', 'dead_reckoning', 'relative', 'approach']
    for i in range(len(ev)):
        k, s = KINDS[kind[i]], stamp[i]
        if k == 'cmd':
            o = est.on_cmd(s, int(val[i, 0]))
        elif k in ('front', 'rear'):
            o = est.on_wheel(k, s, float(val[i, 0]))
        else:
            est.on_fix(k, s, *val[i, :3], int(val[i, 3]))
            o = None
        if o is not None:
            out[i] = (o.stamp, o.v, o.e if o.pos_valid else np.nan, o.n if o.pos_valid else np.nan,
                      o.u if o.pos_valid else np.nan, modes.index(o.mode) + 1 if o.mode in modes else -1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, kind=kind, stamp=stamp, val=val, out=out)
    print(f'{OUT}: {len(ev)} входов, {np.isfinite(out[:, 0]).sum()} выходов, {OUT.stat().st_size / 1e6:.1f} МБ; '
          f'переустановок ветки {est.n_fork_switch}, коррекций по остановкам {est.n_stop_corr}')


if __name__ == '__main__':
    main()

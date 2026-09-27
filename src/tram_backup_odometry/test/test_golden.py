"""Регрессионный golden-тест ядра: bag стенда организаторов (30618_88aea4d9, 1309 с, 50 848 входов).

Входы воспроизводятся в порядке записи, выходы сверяются с зафиксированными (tools/make_golden.py).
Проверяет, что ядро в новом окружении/рантайме считает то же самое: данные пакета (модель, ESN, карта,
профили стрелок) на месте и подхвачены, порядок и единицы входов прежние. Датасет и ROS не нужны.

Данные ядра ищутся так же, как в core.default_files(): TRAM_ODOM_SHARE или исходники пакета.
"""
import os

import numpy as np
import pytest

from tram_backup_odometry.core import build_estimator, default_files

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'golden_88aea4d9.npz')
KINDS = ['cmd', 'front', 'rear', 'master', 'rover']
MODES = ['pre_init', 'map', 'dead_reckoning', 'relative', 'approach']
TOL_V = 1e-6        # м/с
TOL_POS = 1e-4      # м (накопление по 5.5 км пути: допуск на другую сборку numpy/libm)


@pytest.fixture(scope='module')
def replay():
    g = np.load(GOLDEN)
    est = build_estimator(**default_files())
    kind, stamp, val, ref = g['kind'], g['stamp'], g['val'], g['out']
    out = np.full_like(ref, np.nan)
    for i in range(len(kind)):
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
                      o.u if o.pos_valid else np.nan, MODES.index(o.mode) + 1 if o.mode in MODES else -1)
    return est, ref, out


def test_golden_data_loaded(replay):
    est, _, _ = replay
    assert est.map is not None, 'карта pathgraph.json не подхвачена'
    assert est.map.fork_of, 'нет data/branch_profiles.json рядом с pathgraph.json'
    assert est.map.approach, 'нет data/approach_tracks.json рядом с pathgraph.json'
    assert est.esn is not None, 'нет config/esn.json'
    assert est.p.wheel_scale == pytest.approx(0.2778683639906087), 'масштаб колёс не взят из model.json'


def test_golden_outputs_same(replay):
    _, ref, out = replay
    has_ref, has_out = np.isfinite(ref[:, 0]), np.isfinite(out[:, 0])
    assert np.array_equal(has_ref, has_out), 'выход публикуется не на тех же входах'
    r, o = ref[has_ref], out[has_ref]
    assert np.array_equal(r[:, 0], o[:, 0]), 'stamp выхода отличается от stamp входа'
    assert np.array_equal(r[:, 5], o[:, 5]), 'режим положения отличается'
    assert np.max(np.abs(r[:, 1] - o[:, 1])) < TOL_V
    pos_r, pos_o = np.isfinite(r[:, 2]), np.isfinite(o[:, 2])
    assert np.array_equal(pos_r, pos_o), 'положение валидно не на тех же выходах'
    d = np.abs(r[pos_r, 2:5] - o[pos_r, 2:5])
    assert d.max() < TOL_POS, f'положение расходится на {d.max():.2e} м'


def test_golden_branch_and_stops(replay):
    est, _, _ = replay
    assert est.n_fork_switch == 2          # заезд в депо: стрелки 6/47 и 26/48
    assert est.n_stop_corr == 15

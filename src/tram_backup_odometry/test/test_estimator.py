"""Модульные тесты ядра оценщика на синтетических данных (без ROS)."""
import json
import math
import os
import random

import pytest

from tram_backup_odometry.estimator import MODE_MAP, MODE_REL, Estimator, Params
from tram_backup_odometry.geo import EnuFrame, MgrsFrame
from tram_backup_odometry.model import DriveModel
from tram_backup_odometry.pathmap import PathMap

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(PKG, 'config', 'model.json')
MAP = os.path.join(PKG, 'data', 'pathgraph.json')


def make(use_map=False, **kw):
    with open(MODEL, encoding='utf-8') as f:
        mj = json.load(f)
    p = Params(wheel_scale=1 / 3.6, distance_scale=1.0, use_map=use_map, init_timeout=1.0, **kw)
    return Estimator(p, DriveModel(mj), PathMap(MAP) if use_map else None)


def drive(est, profile, t_end, dt=0.05, u=5, slip=None):
    """Колёса 10 Гц, контроллер 20 Гц; profile(t) — истинная скорость, м/с; slip(t) — множитель показаний.
    Шум показаний σ = 0.07 км/ч (как расхождение тележек в данных)."""
    rng = random.Random(1)
    outs, t, k = [], 0.0, 0
    while t <= t_end:
        est.on_cmd(t, u)
        if k % 2 == 0:
            m = slip(t) if slip else 1.0
            for side in ('front', 'rear'):
                o = est.on_wheel(side, t, max(profile(t) * 3.6 * m + rng.gauss(0.0, 0.07), 0.0))
                outs.append((t, o, profile(t)))
        t += dt
        k += 1
    return outs


def test_constant_speed_integrates_distance():
    est = make()
    outs = drive(est, lambda _t: 10.0, 60.0, u=0)
    _, o, _ = outs[-1]
    assert abs(o.v - 10.0) < 0.05
    assert abs(o.s - 600.0) < 2.0
    assert est.mode == MODE_REL        # GNSS нет — относительная одометрия


def test_slip_rejected():
    est = make()
    prof = lambda t: min(0.8 * t, 8.0)                      # noqa: E731
    slip = lambda t: 1.0 + (0.35 * math.sin(math.pi * (t - 5.0) / 3.0) if 5.0 <= t <= 8.0 else 0.0)  # noqa: E731
    outs = drive(est, prof, 12.0, u=10, slip=slip)
    err = max(abs(o.v - v) for t, o, v in outs if 5.0 <= t <= 9.0)
    assert err < 0.8                                        # показания завышены до +35 % (≈2.5 м/с)
    assert any(o.slip for t, o, _ in outs if 5.0 <= t <= 8.5)


def test_dropout_uses_model():
    est = make()
    outs = []
    for k in range(0, 400):
        t = k * 0.05
        est.on_cmd(t, 0)
        if k % 2 == 0 and not (100 <= k < 300):             # колёса пропали на 10 с
            for side in ('front', 'rear'):
                est.on_wheel(side, t, 36.0 + 0.07 * math.sin(7.3 * k + (side == 'rear')))
        outs.append(est.output(t))
    assert abs(outs[299].v - 10.0) < 1.5                    # на выбеге модель держит скорость


def test_frozen_sensor_detected():
    est = make()
    prof = lambda t: 2.0 + 0.5 * t                          # noqa: E731
    outs, t = [], 0.0
    for k in range(200):
        t = k * 0.1
        est.on_cmd(t, 8)
        z = prof(min(t, 5.0)) if t < 10.0 else prof(t)      # 5–10 с показания «залипли»
        for side in ('front', 'rear'):
            o = est.on_wheel(side, t, z * 3.6)
        outs.append((t, o))
    assert any(o.sensor_fault == 'frozen' for t, o in outs if 5.0 < t < 10.0)


def test_enu_roundtrip():
    fr = EnuFrame(55.8, 37.42, 150.0)
    e, n, u = fr.to_enu(55.81, 37.43, 152.0)
    lat, lon, h = fr.to_geodetic(e, n, u)
    assert lat == pytest.approx(55.81, abs=1e-8)
    assert lon == pytest.approx(37.43, abs=1e-8)
    assert h == pytest.approx(152.0, abs=1e-3)


def test_map_init_from_gnss():
    """Выставка по двум антеннам: выход — base_link (9.873 м впереди master), координаты MGRS."""
    est = make(use_map=True)
    pm = est.map
    ed = pm.edges[0]
    i = 500
    fr = MgrsFrame()
    pm.set_frame(fr)
    psi = float(ed.hdg[i])
    xm, ym = float(ed.e[i]), float(ed.n[i])
    ml, mo, _ = fr.to_geodetic(xm, ym, 150.0)
    rl, ro, _ = fr.to_geodetic(xm + 12.436 * math.cos(psi), ym + 12.436 * math.sin(psi), 150.0)
    for k in range(30):
        t = k * 0.1
        est.on_cmd(t, 0)
        est.on_fix('master', t, float(ml), float(mo), 153.0, 2)
        est.on_fix('rover', t, float(rl), float(ro), 153.0, 2)
        est.on_wheel('front', t, 0.0)
    assert est.mode == MODE_MAP
    assert est.anchor[0] == 0 and abs(est.anchor[1] - (i + 9.873)) < 1.0
    o = est.output(3.0)
    assert 90000 < o.e < 110000 and 80000 < o.n < 90000        # плоские координаты судьи


def test_mgrs_matches_organizers_example():
    """Пример из чата организаторов: MGRS 37UCB, x непрерывен через границу квадратов (103501 ≠ DB 3501)."""
    x, y, _ = MgrsFrame().to_enu(55.8088325462547, 37.4602768500852, 0.0)
    assert float(x) == pytest.approx(103501.6309, abs=1e-3)
    assert float(y) == pytest.approx(85876.1201, abs=1e-3)


def test_branch_fit_robust():
    """Выбор ветки: профиль с разгоном узнаётся; остановка (не похожа ни на одну ветку) не даёт перевеса."""
    from tram_backup_odometry.pathmap import branch_fit
    main = ([2.8] * 31, [0.5] * 31)
    depot = ([2.8] * 5 + [2.8 + 0.2 * k for k in range(21)] + [7.0] * 5, [0.5] * 31)
    v_depot = depot[0][:20]
    assert branch_fit(v_depot, *depot)[0] - branch_fit(v_depot, *main)[0] > 6.0
    assert branch_fit(v_depot, *depot)[1] < 3.0
    v_stop = [2.8] * 5 + [0.0] * 15
    assert branch_fit(v_stop, *depot)[1] > 3.0 and branch_fit(v_stop, *main)[1] > 3.0

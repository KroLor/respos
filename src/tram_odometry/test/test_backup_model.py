"""Оценщик backup_model: модель пакета tram_backup_odometry внутри ядра ноды.

Главная проверка — входы golden-теста пакета (bag стенда организаторов, 50 848 сообщений)
через ядро ноды: скорость и положение на каждой метке выхода должны совпасть с эталонными
выходами самой модели, т. е. встраивание не меняет расчёт (контракт docs/INTEGRATION.md).
Без пакета модели тесты пропускаются.
"""
from pathlib import Path

import pytest

from tram_odometry.core import CoreParams, OdometryCore
from tram_odometry.preprocessing import PreprocessingParams

np = pytest.importorskip('numpy')
pytest.importorskip('tram_backup_odometry')

GOLDEN = (Path(__file__).resolve().parents[2] / 'tram_backup_odometry' / 'test' / 'data'
          / 'golden_88aea4d9.npz')
KINDS = ['cmd', 'front', 'rear', 'master', 'rover']


def make(**kwargs):
    kwargs.setdefault('estimator', 'backup_model')
    return OdometryCore(PreprocessingParams(), CoreParams(**kwargs), clock=lambda: 0.0)


def feed(core, kind, stamp, value):
    if kind == 'cmd':
        return core.on_driver_cmd(stamp, int(value[0]))
    if kind in ('front', 'rear'):
        return core.on_wheel(kind, stamp, float(value[0]))
    core.on_gnss_fix(kind, stamp, int(value[3]), *value[:3])
    return None


@pytest.fixture(scope='module')
def golden():
    if not GOLDEN.exists():
        pytest.skip('нет golden-данных пакета модели')
    return np.load(GOLDEN)


def test_model_inside_node_core_gives_model_outputs(golden):
    core = make(gnss_correction=False)
    assert core.estimator.primary.name == 'backup_model'
    compared = positions = 0
    # Выход модели — на каждое входное сообщение; нода публикует на часть из них с той же меткой
    for kind, stamp, value, row in zip(golden['kind'], golden['stamp'], golden['val'], golden['out']):
        output = feed(core, KINDS[kind], float(stamp), value)
        if output is None or not np.isfinite(row[0]):
            continue
        assert output.stamp == row[0]
        assert abs(output.estimate.velocity - row[1]) < 1e-6
        assert output.position_valid == bool(np.isfinite(row[2]))
        if output.position_valid:
            assert abs(output.x - row[2]) < 1e-4 and abs(output.y - row[3]) < 1e-4
            assert abs(output.z - row[4]) < 1e-4
            positions += 1
        compared += 1
    assert core.estimator.active == 'backup_model' and not core.estimator.failures
    assert compared > 25000 and positions > compared - 100


def test_no_position_before_gnss_alignment():
    core = make()
    output = None
    for i in range(40):
        core.on_wheel('front', 0.1 * i, 0.0)
        output = core.on_driver_cmd(0.1 * i + 0.05, 0) or output
    assert output is not None and not output.position_valid
    assert output.estimate.velocity == 0.0


def test_model_failure_switches_to_fallback_and_keeps_position(golden):
    core = make(gnss_correction=False)
    model = core.estimator.primary
    last = None
    for kind, stamp, value in list(zip(golden['kind'], golden['stamp'], golden['val']))[:20000]:
        last = feed(core, KINDS[kind], float(stamp), value) or last
    assert last.position_valid and core.estimator.active == 'backup_model'

    def broken(*args):
        raise RuntimeError('сбой модели')

    model._core.on_cmd = model._core.on_wheel = broken
    stamp = last.stamp
    for i in range(1, 11):
        core.on_wheel('front', stamp + 0.1 * i, 36.0)
        output = core.on_driver_cmd(stamp + 0.1 * i + 0.01, 1)
        assert output is not None and output.position_valid
    assert core.estimator.active == 'wheel_baseline'
    assert model.resets > 0
    # Положение продолжается от последней точки модели по пути запасного оценщика
    travelled = output.estimate.distance - last.estimate.distance
    shift = ((output.x - last.x) ** 2 + (output.y - last.y) ** 2) ** 0.5
    assert 0.0 < travelled < 15.0 and abs(shift - travelled) < 0.5

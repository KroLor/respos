import math

from tram_odometry.core import CoreParams, OdometryCore


def make(**kwargs):
    return OdometryCore(params=CoreParams(**kwargs), clock=lambda: 0.0)


def wheels(core, stamp, speed_kmh):
    """Обе тележки; возвращает результат, если он был выдан."""
    first = core.on_wheel('front', stamp, speed_kmh)
    second = core.on_wheel('rear', stamp, speed_kmh)
    return second or first


def test_output_on_driver_command_with_its_stamp():
    core = make()
    wheels(core, 10.00, 36.0)
    output = core.on_driver_cmd(10.02, 3)
    assert output is not None and output.stamp == 10.02
    assert abs(output.estimate.velocity - 10.0) < 1e-9       # 36 км/ч → 10 м/с
    assert wheels(core, 10.05, 36.0) is None                 # контроллер активен — колёса не триггер


def test_wheels_trigger_when_controller_is_silent():
    core = make(driver_cmd_timeout=0.15)
    wheels(core, 0.9, 18.0)
    core.on_driver_cmd(1.0, 0)
    assert wheels(core, 1.1, 18.0) is None                   # 0,1 с тишины — ещё нет
    output = wheels(core, 1.2, 18.0)                         # 0,2 с > 0,15 с — триггер по колёсам
    assert output is not None and output.stamp == 1.2


def test_waits_for_first_wheel_measurement_but_not_forever():
    core = make(wheel_wait_at_start=1.0)
    outputs = [core.on_driver_cmd(1.0 + i * 0.05, 0) for i in range(20)]   # 0,95 с без колёс
    assert all(o is None for o in outputs)
    output = core.on_driver_cmd(2.0, 0)
    assert output is not None and output.estimate.velocity == 0.0


def test_output_stamps_strictly_increase():
    core = make()
    assert wheels(core, 5.0, 10.0) is not None
    skipped = core.skipped
    assert core.on_driver_cmd(4.9, 0) is None                # метка меньше уже выданной
    assert core.skipped == skipped + 1


def test_time_reset_restarts_output_stamps():
    core = make()
    wheels(core, 1000.0, 10.0)
    for i in range(10):
        core.on_driver_cmd(1000.0 + 0.05 * i, 0)
    results = [core.on_driver_cmd(10.0 + 0.05 * i, 0) for i in range(6)]   # bag запущен заново
    assert all(r is None for r in results[:4])
    assert results[4] is not None and results[4].stamp == 10.0 + 0.05 * 4
    assert core.time_resets == 1


def test_invalid_inputs_are_ignored_without_output():
    core = make()
    assert core.on_wheel('front', 1.0, math.nan) is None
    assert core.on_wheel('front', 1.0, 500.0) is None
    assert core.on_driver_cmd(0.0, 1) is None                 # нулевая метка
    assert core.on_driver_cmd(1.0, 99) is None                # положение ручки вне диапазона
    assert core.published == 0


def test_output_fields():
    core = make()
    for i in range(11):                                       # 1 с при 10 м/с
        wheels(core, i * 0.1, 36.0)
        output = core.on_driver_cmd(i * 0.1 + 0.01, 0)
    assert len(output.pose_covariance) == 36 and len(output.twist_covariance) == 36
    assert abs(output.x - output.estimate.distance) < 1e-9   # путь по прямой вдоль x
    assert output.y == 0.0 and output.yaw == 0.0

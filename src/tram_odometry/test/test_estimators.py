import math

from tram_odometry.estimators import create_estimator


def test_wheel_baseline_integrates_constant_speed():
    estimator = create_estimator('wheel_baseline', {'wheel_timeout': 0.5})
    for i in range(101):  # 10 с с шагом 0,1 с
        stamp = i * 0.1
        estimator.on_wheel('front', stamp, 10.0)
        estimator.on_wheel('rear', stamp, 10.0)
        result = estimator.estimate(stamp)
    assert abs(result.velocity - 10.0) < 1e-9
    assert abs(result.distance - 100.0) < 1e-6


def test_wheel_baseline_ignores_stale_bogie():
    estimator = create_estimator('wheel_baseline', {'wheel_timeout': 0.5})
    estimator.on_wheel('front', 0.0, 20.0)   # передняя тележка замолчала
    estimator.on_wheel('rear', 1.0, 10.0)
    assert abs(estimator.estimate(1.0).velocity - 10.0) < 1e-9


def test_wheel_baseline_holds_speed_without_data():
    estimator = create_estimator('wheel_baseline', {'wheel_timeout': 0.5})
    estimator.on_wheel('front', 0.0, 5.0)
    estimator.on_wheel('rear', 0.0, 5.0)
    estimator.estimate(0.0)
    result = estimator.estimate(2.0)          # 2 с без измерений
    assert abs(result.velocity - 5.0) < 1e-9
    assert abs(result.distance - 10.0) < 1e-6


def cruise(estimator, until, speed=10.0):
    """Обе тележки едут с одной скоростью до момента until (шаг 0,1 с)."""
    for i in range(int(until * 10) + 1):
        estimator.on_wheel('front', i * 0.1, speed)
        estimator.on_wheel('rear', i * 0.1, speed)
        estimator.estimate(i * 0.1)


def test_wheel_baseline_rejects_spike_and_dropout_by_prediction():
    estimator = create_estimator('wheel_baseline', {})
    cruise(estimator, 2.0)
    estimator.on_wheel('front', 2.1, 10.0)
    estimator.on_wheel('rear', 2.1, 21.0)      # всплеск вверх на задней тележке
    result = estimator.estimate(2.1)
    assert abs(result.velocity - 10.0) < 1e-6 and result.slip_detected
    estimator.on_wheel('front', 2.2, 10.0)
    estimator.on_wheel('rear', 2.2, 0.0)       # провал задней тележки в ноль
    assert abs(estimator.estimate(2.2).velocity - 10.0) < 1e-6


def test_wheel_baseline_zero_reading_bogie_is_ignored_even_from_rest():
    estimator = create_estimator('wheel_baseline', {})
    for i in range(51):                        # задний датчик «залип» на 0, разгон 1 м/с²
        estimator.on_wheel('front', i * 0.1, 0.1 * i)
        estimator.on_wheel('rear', i * 0.1, 0.0)
        result = estimator.estimate(i * 0.1)
    assert abs(result.velocity - 5.0) < 1e-6


def test_wheel_baseline_slip_ratio_sign_and_scale():
    estimator = create_estimator('wheel_baseline', {})
    cruise(estimator, 2.0)                     # 10 м/с
    estimator.on_wheel('front', 2.1, 10.0)
    estimator.on_wheel('rear', 2.1, 12.0)      # задняя тележка буксует
    result = estimator.estimate(2.1)
    assert abs(result.velocity - 10.0) < 1e-6
    assert abs(result.slip_ratio - 0.2) < 1e-6
    estimator.on_wheel('front', 2.2, 10.0)
    estimator.on_wheel('rear', 2.2, 7.0)       # задняя тележка идёт юзом
    assert abs(estimator.estimate(2.2).slip_ratio + 0.3) < 1e-6
    standing = create_estimator('wheel_baseline', {})
    cruise(standing, 1.0, speed=0.0)
    assert standing.estimate(1.0).slip_ratio == 0.0


def test_wheel_baseline_time_alignment_removes_lag():
    """Разгон 1 м/с², колёса каждые 0,1 с, оценка — через 0,05 с после измерения (как метка контроллера)."""
    results = {}
    for aligned in (False, True):
        estimator = create_estimator('wheel_baseline', {'wheel_time_alignment': aligned})
        for i in range(60):
            t = 1.0 + 0.1 * i
            estimator.on_wheel('front', t, t)
            estimator.on_wheel('rear', t, t)
            result = estimator.estimate(t + 0.05)
        results[aligned] = result.velocity - (t + 0.05)      # ошибка относительно истинной скорости
    assert abs(results[False] + 0.05) < 0.01                 # без пересчёта запаздывание 0,05 с × 1 м/с²
    assert abs(results[True]) < 0.01                         # с пересчётом — без смещения


def test_wheel_baseline_recovers_after_wheel_silence_without_overshoot():
    estimator = create_estimator('wheel_baseline', {})
    cruise(estimator, 2.0)                     # 10 м/с
    for i in range(1, 31):                     # 3 с без данных колёс (удержание, затем снижение)
        estimator.estimate(2.0 + 0.1 * i)
    for i in range(20):                        # данные вернулись: трамвай уже едет 12 м/с
        t = 5.1 + 0.1 * i
        estimator.on_wheel('front', t, 12.0)
        estimator.on_wheel('rear', t, 12.0)
        result = estimator.estimate(t + 0.05)
        assert abs(result.velocity - 12.0) < 0.05, f't={t:.1f}: {result.velocity:.3f}'


def test_wheel_baseline_rejects_spike_while_standing():
    estimator = create_estimator('wheel_baseline', {})
    cruise(estimator, 2.0, speed=0.0)          # трамвай стоит
    estimator.on_wheel('front', 2.1, 0.0)
    estimator.on_wheel('rear', 2.1, 11.0)      # всплеск задней тележки
    assert estimator.estimate(2.1).velocity == 0.0


def test_wheel_baseline_long_mismatch_takes_higher_speed():
    estimator = create_estimator('wheel_baseline', {'bogie_mismatch_max_duration': 3.0})
    for i in range(41):                        # задний датчик занижает вдвое с самого старта
        estimator.on_wheel('front', i * 0.1, 10.0)
        estimator.on_wheel('rear', i * 0.1, 5.0)
        result = estimator.estimate(i * 0.1)
        if i == 10:
            assert result.velocity == 5.0      # вначале прогноз (0) ближе к заниженному
    assert result.velocity == 10.0             # через 3 с — отказ датчика, берём большую


def test_wheel_baseline_decays_after_long_silence():
    estimator = create_estimator('wheel_baseline', {'wheel_hold_max': 2.0, 'wheel_decay_tau': 5.0})
    estimator.on_wheel('front', 0.0, 10.0)
    estimator.on_wheel('rear', 0.0, 10.0)
    for i in range(101):                       # 10 с без новых данных колёс
        result = estimator.estimate(i * 0.1)
    assert result.velocity < 10.0 * math.exp(-7.0 / 5.0) * 1.05
    assert result.velocity > 0.0


def test_wheel_baseline_time_reset():
    estimator = create_estimator('wheel_baseline', {})
    estimator.on_wheel('front', 100.0, 10.0)
    estimator.estimate(100.0)
    estimator.on_time_reset(5.0)
    estimator.on_wheel('front', 5.0, 8.0)
    result = estimator.estimate(5.0)
    assert result.velocity == 8.0              # старые метки не мешают новым
    estimator.on_wheel('front', 6.0, 8.0)
    assert estimator.estimate(6.0).distance > result.distance


def test_gnss_passthrough_prefers_master():
    estimator = create_estimator('gnss_passthrough', {})
    estimator.on_gnss_fix('master', 0.0, 55.75, 37.62, 150.0)
    estimator.on_gnss_fix('rover', 0.1, 55.76, 37.62, 150.0)   # master жив — rover игнорируется
    east, north, _ = estimator.estimate(0.1).position
    assert abs(east) < 1e-6 and abs(north) < 1e-6
    estimator.on_gnss_fix('rover', 3.0, 55.751, 37.62, 150.0)  # master молчит > 2 с
    _, north, _ = estimator.estimate(3.0).position
    assert abs(north - 111.34) < 0.1

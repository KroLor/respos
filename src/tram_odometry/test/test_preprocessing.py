import math

from tram_odometry.preprocessing import (NOMINAL_WHEEL_SCALE, VEHICLE_WHEEL_SCALES,
                                         InputPreprocessor, PreprocessingParams)


def make():
    # Номинальный масштаб 1/3,6 — чтобы проверять логику на «круглых» числах
    return InputPreprocessor(PreprocessingParams(wheel_speed_scale=NOMINAL_WHEEL_SCALE))


def test_invalid_and_out_of_range_values_are_rejected():
    pre = make()
    assert pre.wheel('front', 1.0, math.nan) is None
    assert pre.wheel('front', 1.1, math.inf) is None
    assert pre.wheel('front', 1.2, 500.0) is None
    assert pre.wheel('front', 1.3, -5.0) is None
    assert pre.driver_cmd(1.0, 99) is None
    rejected = pre.state('front').rejected
    assert rejected['value_invalid'] == 2 and rejected['value_out_of_range'] == 2
    assert pre.state('driver_cmd').rejected['value_out_of_range'] == 1


def test_small_negative_speed_is_clamped_and_converted():
    pre = make()
    sample = pre.wheel('front', 1.0, -0.4)
    assert sample.value == 0.0 and pre.state('front').clamped == 1
    assert abs(pre.wheel('front', 1.1, 36.0).value - 10.0) < 1e-9   # км/ч → м/с


def test_wheel_scale_by_vehicle():
    assert InputPreprocessor(PreprocessingParams()).wheel_scale == VEHICLE_WHEEL_SCALES['30618']
    pre = InputPreprocessor(PreprocessingParams(vehicle_id='30639'))
    assert pre.wheel_scale == VEHICLE_WHEEL_SCALES['30639'] and pre.known_vehicle
    unknown = InputPreprocessor(PreprocessingParams(vehicle_id='99999'))
    assert unknown.wheel_scale == NOMINAL_WHEEL_SCALE and not unknown.known_vehicle
    manual = InputPreprocessor(PreprocessingParams(vehicle_id='30639', wheel_speed_scale=0.25))
    assert manual.wheel_scale == 0.25
    # 30618: путь колёс / путь GNSS = 3,594 — 35,94 км/ч по колёсам это 10 м/с
    assert abs(InputPreprocessor(PreprocessingParams()).wheel('front', 1.0, 35.94).value - 10.0) < 1e-9


def test_bad_stamps_are_rejected():
    pre = make()
    assert pre.driver_cmd(0.0, 1) is None                 # нулевая метка
    assert pre.driver_cmd(10.0, 1) is not None
    assert pre.driver_cmd(10.0, 1) is None                # повтор
    assert pre.driver_cmd(9.5, 1) is None                 # откат назад
    rejected = pre.state('driver_cmd').rejected
    assert rejected['stamp_invalid'] == 1
    assert rejected['stamp_duplicate'] == 1 and rejected['stamp_backward'] == 1


def test_garbage_future_stamp_is_rejected_real_gap_is_accepted():
    pre = make()
    for i in range(10):
        pre.wheel('front', 100.0 + 0.1 * i, 20.0)
        pre.driver_cmd(100.0 + 0.1 * i, 1)
    assert pre.driver_cmd(100.0 + 3600.0, 1) is None      # на час вперёд — мусор
    assert pre.state('driver_cmd').rejected['stamp_ahead'] == 1
    assert pre.driver_cmd(101.0, 1) is not None           # поток не сломан
    # Настоящий пропуск задней тележки на 73 с при живых других входах — принимается
    pre.wheel('rear', 30.0, 0.0)
    assert pre.wheel('rear', 101.0, 39.0) is not None
    assert pre.state('rear').gaps == 1


def test_interleaved_delayed_stream_keeps_freshest_data():
    # Как в 30618_2255aade: поверх основного потока идут сообщения с метками на ~1 с позже
    pre = make()
    accepted = []
    for i in range(20):
        t = 50.0 + 0.05 * i
        for stamp in (t, t + 1.0):
            if pre.driver_cmd(stamp, 1) is not None:
                accepted.append(stamp)
    assert accepted == sorted(accepted)                   # метки на выходе только растут
    assert accepted[-1] == 50.0 + 0.05 * 19 + 1.0          # свежие данные не потеряны


def test_new_timebase_is_adopted_after_consistent_messages():
    pre = make()
    for i in range(10):
        pre.driver_cmd(1000.0 + 0.05 * i, 1)
    results = [pre.driver_cmd(10.0 + 0.05 * i, 1) for i in range(6)]   # bag запущен заново
    assert all(r is None for r in results[:4])
    assert results[4] is not None and results[4].time_reset
    assert results[5] is not None and not results[5].time_reset
    assert pre.state('driver_cmd').resyncs == 1


def test_suspicious_jump_and_bogie_mismatch():
    pre = make()
    assert not pre.wheel('front', 1.0, 30.0).suspicious
    assert not pre.wheel('rear', 1.0, 30.2).suspicious
    assert not pre.wheel('front', 1.1, 30.5).suspicious   # обычное изменение
    sample = pre.wheel('rear', 1.1, 18.0)                 # юз: -12 км/ч за 0,1 с
    assert sample.suspicious and abs(sample.value - 5.0) < 1e-9   # помечено, но не отброшено
    assert pre.wheel('front', 1.2, 30.4).suspicious       # расходится с задней тележкой
    assert pre.state('rear').suspicious == 1 and pre.state('front').suspicious == 1


def test_gnss_validation():
    pre = make()
    assert pre.gnss_fix('master', 1.0, -1, 55.7, 37.6, 150.0) is None       # нет решения
    assert pre.gnss_fix('master', 1.1, 2, math.nan, 37.6, 150.0) is None
    assert pre.gnss_fix('master', 1.2, 2, 0.0, 0.0, 0.0) is None
    assert pre.gnss_fix('master', 1.3, 2, 55.7, 37.6, 150.0).value == (55.7, 37.6, 150.0, 2)
    assert pre.gnss_vel('master', 1.3, 1.0, 2.0, 0.0) is not None
    assert pre.gnss_vel('master', 1.4, 500.0, 0.0, 0.0) is None

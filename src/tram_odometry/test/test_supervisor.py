import math

from tram_odometry.estimators import Estimate, Estimator, create_estimator
from tram_odometry.supervisor import HOLD, SupervisedEstimator


class FlakyEstimator(Estimator):
    """Тестовый оценщик: 10 м/с, по команде ломается (исключение или NaN)."""

    name = 'flaky'

    def __init__(self):
        super().__init__({})
        self.mode = 'ok'
        self.distance = 0.0
        self.stamp = None

    def estimate(self, stamp):
        if self.stamp is not None:
            self.distance += 10.0 * (stamp - self.stamp)
        self.stamp = stamp
        if self.mode == 'raise':
            raise RuntimeError('сбой модели')
        if self.mode == 'nan':
            return Estimate(velocity=math.nan, distance=self.distance)
        return Estimate(velocity=10.0, distance=self.distance)


def feed_wheels(supervisor, stamp, speed):
    supervisor.on_wheel('front', stamp, speed)
    supervisor.on_wheel('rear', stamp, speed)


def test_fallback_on_failure_and_recovery_without_distance_jump():
    primary = FlakyEstimator()
    supervisor = SupervisedEstimator(primary, create_estimator('wheel_baseline', {}))
    distances = []
    for i in range(60):
        stamp = i * 0.1
        primary.mode = {20: 'raise', 30: 'nan', 40: 'ok'}.get(i, primary.mode)
        feed_wheels(supervisor, stamp, 10.0)
        result = supervisor.estimate(stamp)
        distances.append(result.distance)
        expected = 'flaky' if i < 20 or i >= 40 else 'wheel_baseline'
        assert supervisor.active == expected
    steps = [b - a for a, b in zip(distances, distances[1:])]
    assert all(abs(step - 1.0) < 1e-6 for step in steps)   # путь без скачков при переключениях
    assert supervisor.failures['flaky'] == 20 and supervisor.switches == 2


def test_hold_when_all_estimators_fail():
    primary = FlakyEstimator()
    supervisor = SupervisedEstimator(primary, None)
    supervisor.estimate(0.0)
    supervisor.estimate(1.0)
    primary.mode = 'raise'
    result = supervisor.estimate(2.0)
    assert supervisor.active == HOLD
    assert result.velocity == 10.0 and abs(result.distance - 20.0) < 1e-6


def test_invalid_results_are_rejected():
    supervisor = SupervisedEstimator(FlakyEstimator(), None, max_speed=30.0)
    assert supervisor._validate(Estimate(velocity=50.0, distance=0.0)) is None
    assert supervisor._validate(Estimate(velocity=-3.0, distance=0.0)) is None
    assert supervisor._validate(Estimate(velocity=-0.2, distance=0.0)).velocity == 0.0
    assert supervisor._validate(Estimate(velocity=1.0, distance=math.inf)) is None
    assert supervisor._validate('не Estimate') is None


def test_invalid_slip_ratio_is_dropped_not_failed():
    supervisor = SupervisedEstimator(FlakyEstimator(), None)
    checked = supervisor._validate(Estimate(velocity=1.0, distance=0.0, slip_ratio=math.nan))
    assert checked is not None and checked.slip_ratio is None


def test_input_errors_do_not_break_other_estimators():
    class BrokenInput(FlakyEstimator):
        def on_wheel(self, *args):
            raise ValueError('сбой приёма')

    baseline = create_estimator('wheel_baseline', {})
    supervisor = SupervisedEstimator(BrokenInput(), baseline)
    feed_wheels(supervisor, 0.0, 5.0)
    assert supervisor.failures['flaky'] == 2
    assert baseline.estimate(0.0).velocity == 5.0

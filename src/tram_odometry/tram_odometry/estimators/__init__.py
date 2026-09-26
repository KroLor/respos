"""Реестр оценщиков: имя (значение параметра estimator) → класс."""
from tram_odometry.estimators.base import Estimate, Estimator
from tram_odometry.estimators.gnss_passthrough import GnssPassthroughEstimator
from tram_odometry.estimators.wheel_baseline import WheelBaselineEstimator

ESTIMATORS = {
    WheelBaselineEstimator.name: WheelBaselineEstimator,
    GnssPassthroughEstimator.name: GnssPassthroughEstimator,
}


def create_estimator(name: str, params: dict) -> Estimator:
    """Создать оценщик по имени; KeyError, если такого нет."""
    return ESTIMATORS[name](params)


__all__ = ['ESTIMATORS', 'Estimate', 'Estimator', 'create_estimator']

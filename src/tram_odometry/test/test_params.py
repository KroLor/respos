"""config/params.yaml должен совпадать со значениями по умолчанию в коде.

Нода берёт параметры из params.yaml, а модульные тесты и офлайн-оценка — из dataclass-ов;
при расхождении проверялось бы не то, что работает у жюри.
"""
import math
from pathlib import Path

import pytest
import yaml

from tram_odometry.core import CoreParams
from tram_odometry.preprocessing import PreprocessingParams

PARAMS_FILE = Path(__file__).resolve().parents[1] / 'config' / 'params.yaml'


def test_params_yaml_matches_code_defaults():
    data = yaml.safe_load(PARAMS_FILE.read_text(encoding='utf-8'))
    ros_params = data['tram_odometry']['ros__parameters']
    checked = 0
    for cls in (PreprocessingParams, CoreParams):
        for name, default in vars(cls()).items():
            if name not in ros_params:
                continue
            value = ros_params[name]
            assert type(value) is type(default), f'{name}: тип {type(value)} вместо {type(default)}'
            if isinstance(default, float):
                assert math.isclose(value, default), f'{name}: {value} != {default}'
            else:
                assert value == default, f'{name}: {value} != {default}'
            checked += 1
    assert checked >= 15


def test_model_section_holds_model_defaults():
    """Раздел model: в params.yaml показывает умолчания модели и сам ничего не меняет."""
    estimator = pytest.importorskip('tram_backup_odometry.estimator')
    data = yaml.safe_load(PARAMS_FILE.read_text(encoding='utf-8'))
    model = data['tram_odometry']['ros__parameters']['model']
    defaults = vars(estimator.Params())
    for name, value in model.items():
        assert name in defaults, f'model.{name}: нет такого параметра модели'
        assert type(value) is type(defaults[name]), f'model.{name}: тип {type(value)}'
        assert value == defaults[name], f'model.{name}: {value} != {defaults[name]}'
    assert len(model) >= 15

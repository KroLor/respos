"""Фасад расчётного ядра (без ROS): сборка оценщика из файлов пакета.

Единая точка для ноды ROS 2, тестов и встраивания в другой рантайм — все собирают ядро одинаково:
масштаб колёс и пути берутся из откалиброванной модели, явные параметры перекрывают калибровку,
карта (вместе с лежащими рядом approach_tracks.json и branch_profiles.json) и ESN подключаются по Params.
"""
from __future__ import annotations

import json
import os

from .esn import ResidualESN
from .estimator import Estimator, Output, Params  # noqa: F401  (реэкспорт для внешнего кода)
from .model import DriveModel

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_files(share: str | None = None) -> dict:
    """Пути к данным ядра: каталог share установленного пакета, переменная окружения TRAM_ODOM_SHARE
    или исходники пакета (рядом с этим модулем)."""
    root = share or os.environ.get('TRAM_ODOM_SHARE') or PKG_DIR
    return {'model_file': os.path.join(root, 'config', 'model.json'),
            'map_file': os.path.join(root, 'data', 'pathgraph.json'),
            'esn_file': os.path.join(root, 'config', 'esn.json')}


def build_estimator(model_file: str, map_file: str | None = None, esn_file: str | None = None,
                    overrides: dict | None = None) -> Estimator:
    """Оценщик с откалиброванной моделью. overrides — поля Params, перекрывающие значения по умолчанию
    и калибровку (например {'use_esn': False})."""
    from .pathmap import PathMap
    with open(model_file, encoding='utf-8') as f:
        mj = json.load(f)
    p = Params()
    p.wheel_scale = mj.get('wheel_scale', p.wheel_scale)
    p.distance_scale = mj.get('distance_scale', p.distance_scale)
    for k, v in (overrides or {}).items():
        if not hasattr(p, k):
            raise KeyError(f'неизвестный параметр оценщика: {k}')
        setattr(p, k, type(getattr(p, k))(v))
    pm = PathMap(map_file) if (p.use_map and map_file) else None
    esn = ResidualESN.from_file(esn_file) if (p.use_esn and esn_file and os.path.exists(esn_file)) else None
    return Estimator(p, DriveModel(mj), pm, esn)

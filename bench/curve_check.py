"""Проверка: есть ли в остатке физической модели сопротивление на кривых (зависимость от 1/R).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/curve_check.py
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import esn_feasibility as F
from common import ROOT, load_run, splits


def curv_grid(b):
    S = pd.read_parquet(ROOT / 'map' / 'pathgraph_samples.parquet')
    g = F.run_grid(b)
    if g is None:
        return None
    fx = load_run(b)['master_fix']
    fx = fx[fx.status == 2].sort_values('t_hdr')
    tr = cKDTree(S[['x', 'y']].values)
    x = np.radians(fx.lon.values - F.LON0) * F.R_E * math.cos(math.radians(F.LAT0))
    y = np.radians(fx.lat.values - F.LAT0) * F.R_E
    d, j = tr.query(np.c_[x, y])
    c = np.where(d < 3, S.curv.values[j], 0.0)
    g['curv'] = np.interp(g.t, fx.t_hdr.values / 1e9, c)
    g['a_phys'] = F.physics(g.u.to_numpy(), g.v.to_numpy(), g.grade.to_numpy())
    g['res'] = g.a - g.a_phys
    return g


if __name__ == '__main__':
    with ProcessPoolExecutor(16) as ex:
        G = [g for g in ex.map(curv_grid, splits()['train']) if g is not None]
    D = pd.concat(G)
    D = D[(D.v > 2) & D.grade.notna()]
    D['R'] = 1 / np.maximum(np.abs(D.curv), 1e-6)
    D['rb'] = pd.cut(D.R, [0, 30, 60, 100, 200, 500, 1e9])
    print(D.groupby('rb', observed=True).agg(n=('res', 'size'), res_med=('res', 'median'), res_mean=('res', 'mean')).round(3))
    X = np.c_[np.abs(D.curv), np.ones(len(D))]
    k = np.linalg.lstsq(X, D.res.to_numpy(), rcond=None)[0]
    print(f'остаток = {k[0]:.2f}·(1/R) {k[1]:+.4f};  в удельных силах w_r = {-k[0] / 9.81 * 1000:.0f} / R  Н/кН')

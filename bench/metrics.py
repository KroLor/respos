"""Метрики, повторяющие критерии жюри (PDF кейса, раздел «Детализация критериев»)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from common import ROOT

TOL = 0.05     # допуск сопоставления по stamp, с
_map = pd.read_parquet(ROOT / 'map' / 'pathgraph_samples.parquet')
_R, _LAT0, _LON0 = 6378137.0, 55.804767, 37.419565
_tree = cKDTree(_map[['x', 'y']].to_numpy())


def match(est: pd.DataFrame, ref_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой точки эталона — ближайший по stamp выход оценщика в пределах TOL. -> (idx_ref, idx_est)."""
    te = est.t.to_numpy()
    j = np.clip(np.searchsorted(te, ref_t), 1, len(te) - 1)
    jj = np.where(np.abs(te[j - 1] - ref_t) <= np.abs(te[j] - ref_t), j - 1, j)
    ok = np.abs(te[jj] - ref_t) <= TOL
    return np.nonzero(ok)[0], jj[ok]


def _stats(x: np.ndarray, pre: str) -> dict:
    if len(x) == 0:
        return {f'{pre}_rmse': np.nan, f'{pre}_mae': np.nan, f'{pre}_bias': np.nan, f'{pre}_max': np.nan}
    return {f'{pre}_rmse': float(np.sqrt(np.mean(x ** 2))), f'{pre}_mae': float(np.mean(np.abs(x))),
            f'{pre}_bias': float(np.mean(x)), f'{pre}_max': float(np.max(np.abs(x)))}


def speed_metrics(est: pd.DataFrame, ref: dict, windows: list | None = None) -> dict:
    vr = ref['vel']
    ir, ie = match(est, vr.t.to_numpy())
    r = vr.iloc[ir].reset_index(drop=True)
    ve = est.v.to_numpy()[ie]
    err = ve - r.v.to_numpy()
    clean = ~r.glitch.to_numpy()
    out = {'n_vel_matched': len(ir), 'vel_match_rate': len(ir) / max(len(vr), 1)}
    out.update(_stats(err, 'v_raw'))
    out.update(_stats(err[clean], 'v'))
    # режимы по эталону (ускорение — центрированная производная сглаженной скорости; только для разметки)
    vs = gaussian_filter1d(r.v_wheel.to_numpy(), 5)
    a = np.gradient(vs, r.t.to_numpy()) if len(r) > 2 else np.zeros(len(r))
    reg = np.where(vs < 0.2, 'stop', np.where(a > 0.15, 'accel', np.where(a < -0.15, 'brake', 'cruise')))
    for name in ('stop', 'accel', 'brake', 'cruise'):
        m = clean & (reg == name)
        out.update({k: v for k, v in _stats(err[m], f'v_{name}').items() if k.endswith(('rmse', 'bias'))})
    m = clean & np.isin(reg, ['accel', 'brake'])
    out.update({k: v for k, v in _stats(err[m], 'v_trans').items() if k.endswith(('rmse', 'bias', 'max'))})
    if windows:
        tt = r.t.to_numpy()
        inw = np.zeros(len(tt), bool)
        for a0, a1 in windows:
            inw |= (tt >= a0) & (tt <= a1 + 2.0)
        out.update({k: v for k, v in _stats(err[clean & inw], 'v_fault').items() if k.endswith(('rmse', 'max'))})
    return out


def position_metrics(est: pd.DataFrame, ref: dict) -> tuple[dict, pd.DataFrame]:
    pr = ref['pos']
    if len(pr) < 20:
        return {}, pd.DataFrame()
    if 'pos_valid' in est.columns:
        est = est[est.pos_valid].reset_index(drop=True)     # положение не публикуется до первого фикса
    ir, ie = match(est, pr.t.to_numpy())
    r = pr.iloc[ir].reset_index(drop=True)
    E = est.iloc[ie].reset_index(drop=True)
    de, dn, du = E.e - r.e, E.n - r.n, E.u - r.u
    # касательная эталонной траектории: центрированная разность ±1 с; на стоянке — последняя известная
    t, x, y = pr.t.to_numpy(), pr.e.to_numpy(), pr.n.to_numpy()
    i1 = np.clip(np.searchsorted(t, r.t + 1.0), 0, len(t) - 1)
    i0 = np.clip(np.searchsorted(t, r.t - 1.0), 0, len(t) - 1)
    tx, ty = x[i1] - x[i0], y[i1] - y[i0]
    ln = np.hypot(tx, ty)
    mv = ln > 1.0
    hd = pd.Series(np.where(mv, np.arctan2(ty, tx), np.nan)).ffill().bfill().fillna(0.0).to_numpy()
    along = de * np.cos(hd) + dn * np.sin(hd)
    cross = -de * np.sin(hd) + dn * np.cos(hd)
    h = np.hypot(de, dn)
    d3 = np.sqrt(de ** 2 + dn ** 2 + du ** 2)
    step = np.hypot(np.diff(x), np.diff(y))
    dist = float(np.sum(step[(step < 5.0)]))
    # поперечное отклонение оценки от pathgraph
    lat, lon = E.lat.to_numpy(), E.lon.to_numpy()
    okg = np.isfinite(lat)
    xm = np.radians(lon[okg] - _LON0) * _R * math.cos(math.radians(_LAT0))
    ym = np.radians(lat[okg] - _LAT0) * _R
    dmap = _tree.query(np.c_[xm, ym])[0] if okg.any() else np.array([])
    dmap_ref = _tree.query(np.c_[np.radians(np.array(ref_lon(ref, r)) - _LON0) * _R * math.cos(math.radians(_LAT0)),
                                 np.radians(np.array(ref_lat(ref, r)) - _LAT0) * _R])[0]
    out = {'n_pos_matched': len(ir), 'dist_m': dist,
           'final_err_m': float(h.iloc[-1]), 'final_drift_pct': float(100 * h.iloc[-1] / max(dist, 1.0)),
           'final_err3d_m': float(d3.iloc[-1]),
           'along_mean': float(np.mean(along)), 'along_mae': float(np.mean(np.abs(along))),
           'along_max': float(np.max(np.abs(along))), 'along_rmse': float(np.sqrt(np.mean(along ** 2))),
           'cross_rmse': float(np.sqrt(np.mean(cross ** 2))), 'cross_max': float(np.max(np.abs(cross))),
           'h_rmse': float(np.sqrt(np.mean(h ** 2))), 'h_max': float(np.max(h)),
           'err3d_rmse': float(np.sqrt(np.mean(d3 ** 2))), 'err3d_max': float(np.max(d3)),
           'up_rmse': float(np.sqrt(np.mean(du ** 2))),
           'xtrack_map_rmse': float(np.sqrt(np.mean(dmap ** 2))) if len(dmap) else np.nan,
           'xtrack_map_max': float(np.max(dmap)) if len(dmap) else np.nan,
           'ref_offmap_p95': float(np.percentile(dmap_ref, 95))}
    series = pd.DataFrame({'t': r.t, 'along': along, 'cross': cross, 'h': h, 'du': du})
    return out, series


def ref_lat(ref, r):
    return ref['frame'].to_geodetic(r.e.to_numpy(), r.n.to_numpy(), r.u.to_numpy())[0]


def ref_lon(ref, r):
    return ref['frame'].to_geodetic(r.e.to_numpy(), r.n.to_numpy(), r.u.to_numpy())[1]


def timing_metrics(est: pd.DataFrame, proc: np.ndarray) -> dict:
    t = est.t.to_numpy()
    dt = np.diff(t)
    return {'rate_hz': float(len(t) / max(t[-1] - t[0], 1e-9)) if len(t) > 1 else np.nan,
            'max_gap_s': float(dt.max()) if len(dt) else np.nan,
            'gap_gt_100ms': int((dt > 0.1).sum()) if len(dt) else 0,
            'proc_p50_ms': float(np.percentile(proc, 50) * 1e3), 'proc_p99_ms': float(np.percentile(proc, 99) * 1e3),
            'proc_max_ms': float(proc.max() * 1e3)}


def slip_metrics(est: pd.DataFrame, windows: list | None) -> dict:
    """Качество флага проскальзывания против размеченных окон инъекции."""
    if not windows:
        return {'slip_flag_frac': float(est.slip.mean())}
    t = est.t.to_numpy()
    inw = np.zeros(len(t), bool)
    for a0, a1 in windows:
        inw |= (t >= a0) & (t <= a1)
    hit = sum(bool(est.slip.to_numpy()[(t >= a0) & (t <= a1 + 0.5)].any()) for a0, a1 in windows)
    return {'slip_detect_rate': hit / len(windows),
            'slip_false_frac': float(est.slip.to_numpy()[~inw].mean()) if (~inw).any() else np.nan}

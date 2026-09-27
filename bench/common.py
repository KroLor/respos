"""Общие функции офлайн-стенда: прогоны, разбиение, поток событий как при `ros2 bag play`, эталон GNSS."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / 'extracted'
PKG = ROOT / 'src' / 'tram_backup_odometry'
sys.path.insert(0, str(PKG))

from tram_backup_odometry.esn import ResidualESN  # noqa: E402
from tram_backup_odometry.estimator import Estimator, Params  # noqa: E402
from tram_backup_odometry.geo import EnuFrame, MgrsFrame  # noqa: E402
from tram_backup_odometry.model import DriveModel  # noqa: E402
from tram_backup_odometry.pathmap import PathMap  # noqa: E402

SPLITS = ROOT / 'bench' / 'splits.json'
MODEL_JSON = PKG / 'config' / 'model.json'
MAP_JSON = PKG / 'data' / 'pathgraph.json'
NAMES = ['front', 'rear', 'cmd', 'master_fix', 'rover_fix', 'master_vel', 'rover_vel']


COLS = {'front': ['velocity'], 'rear': ['velocity'], 'cmd': ['position'],
        'master_fix': ['lat', 'lon', 'alt', 'status'], 'rover_fix': ['lat', 'lon', 'alt', 'status'],
        'master_vel': ['vx', 'vy', 'vz', 'wz'], 'rover_vel': ['vx', 'vy', 'vz', 'wz']}


def load_run(bag: str) -> dict[str, pd.DataFrame]:
    out = {}
    for n in NAMES:
        d = pd.read_parquet(EX / bag / f'{n}.parquet')
        if 't_bag' not in d.columns:
            d = pd.DataFrame({c: pd.Series(dtype='int64' if c.startswith('t_') else 'float64')
                              for c in ['t_bag', 't_hdr'] + COLS[n]})
        out[n] = d.sort_values('t_bag', kind='stable').reset_index(drop=True)
    return out


# ============================================================================ разбиение
def splits(rebuild: bool = False) -> dict:
    """Уникальные прогоны (дубли по md5 скорости front) → train/val. val — каждый 4-й прогон с GNSS."""
    if SPLITS.exists() and not rebuild:
        return json.loads(SPLITS.read_text(encoding='utf-8'))
    summary = pd.read_csv(EX / 'summary.csv')
    groups: dict[str, list[str]] = {}
    for b in summary.bag:
        f = pd.read_parquet(EX / b / 'front.parquet')
        groups.setdefault(hashlib.md5(f.velocity.round(4).to_numpy().tobytes()).hexdigest(), []).append(b)
    uniq = sorted(v[0] for v in groups.values())
    dups = {v[0]: v[1:] for v in groups.values() if len(v) > 1}
    info = summary.set_index('bag')
    with_gnss = [b for b in uniq if info.loc[b, 'n_master_fix'] > 0 and info.loc[b, 'duration_s'] > 60]
    val = sorted(with_gnss[i] for i in range(0, len(with_gnss), 4))
    train = sorted(b for b in uniq if b not in val)
    out = {'val': val, 'train': train, 'duplicates': dups,
           'note': 'уникальные прогоны; val — каждый 4-й уникальный прогон с GNSS (>60 с) в алфавитном порядке'}
    SPLITS.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


# ============================================================================ оценщик
ESN_JSON = PKG / 'config' / 'esn.json'


def make_estimator(overrides: dict | None = None, use_map: bool = True) -> Estimator:
    model = DriveModel.from_file(str(MODEL_JSON))
    mj = json.loads(MODEL_JSON.read_text(encoding='utf-8'))
    p = Params(wheel_scale=mj.get('wheel_scale', 1 / 3.6), distance_scale=mj.get('distance_scale', 1.0), use_map=use_map)
    for k, v in (overrides or {}).items():
        setattr(p, k, v)
    esn = ResidualESN.from_file(str(ESN_JSON)) if (p.use_esn and ESN_JSON.exists()) else None
    return Estimator(p, model, PathMap(str(MAP_JSON)) if use_map else None, esn)


def events(run: dict[str, pd.DataFrame], gnss_window_s: float = 5.0) -> list[tuple]:
    """Сообщения в порядке записи (t_bag), как их публикует `ros2 bag play`.
    GNSS обрезается до первых gnss_window_s секунд (как в проверочных прогонах)."""
    ev = []
    for side in ('front', 'rear'):
        d = run[side]
        ev += list(zip(d.t_bag.to_numpy(), [side] * len(d), d.t_hdr.to_numpy() / 1e9, d.velocity.to_numpy()))
    d = run['cmd']
    ev += list(zip(d.t_bag.to_numpy(), ['cmd'] * len(d), d.t_hdr.to_numpy() / 1e9, d.position.to_numpy()))
    for ant in ('master', 'rover'):
        d = run[f'{ant}_fix']
        if len(d):
            d = d[d.t_bag <= d.t_bag.iloc[0] + gnss_window_s * 1e9]
            ev += list(zip(d.t_bag.to_numpy(), [ant] * len(d), d.t_hdr.to_numpy() / 1e9,
                           zip(d.lat, d.lon, d.alt, d.status)))
    ev.sort(key=lambda r: r[0])
    return ev


def run_estimator(est: Estimator, ev: list[tuple]) -> tuple[pd.DataFrame, np.ndarray]:
    """Прогон оценщика. Выход публикуется на каждое входное сообщение (без дублей по stamp)."""
    rows, proc = [], []
    recent = deque(maxlen=64)
    pc = time.perf_counter
    for _, kind, stamp, val in ev:
        t0 = pc()
        if kind == 'cmd':
            o = est.on_cmd(stamp, int(val))
        elif kind in ('front', 'rear'):
            o = est.on_wheel(kind, stamp, float(val))
        else:
            est.on_fix(kind, stamp, *val)
            o = None
        proc.append(pc() - t0)
        if o is not None and o.stamp not in recent:
            recent.append(o.stamp)
            rows.append((o.stamp, o.v, o.a, o.s, o.e, o.n, o.u, o.yaw, o.mode, o.slip, o.slip_kind, o.slip_ratio,
                         o.wheels_ok, o.lat, o.lon, o.h, o.var_v, o.pos_valid))
    out = pd.DataFrame(rows, columns=['t', 'v', 'a', 's', 'e', 'n', 'u', 'yaw', 'mode', 'slip', 'slip_kind',
                                      'slip_ratio', 'wheels_ok', 'lat', 'lon', 'h', 'var_v', 'pos_valid'])
    out = out.sort_values('t', kind='stable').reset_index(drop=True)    # выходы с «запоздавшими» stamp
    return out, np.asarray(proc)


# ============================================================================ эталон
K_BL = 9.873 / 12.436            # base_link = master + K_BL·(rover − master)  (tf организаторов)


def reference(run: dict[str, pd.DataFrame]) -> dict | None:
    """Эталон положения: base_link (ось первой тележки, уровень рельса) по паре антенн status=2
    в плоских координатах судьи (MGRS 37U CB: x = E − 300000, y = N − 6100000, z = эллипс. высота − 3.0 м).
    Скорость: |twist| master (единого эталона скорости у организаторов нет)."""
    fx = run['master_fix'].sort_values('t_hdr')
    rv = run['rover_fix'].sort_values('t_hdr')
    if fx.empty or rv.empty:
        return None
    frame = MgrsFrame()
    fx = fx[fx.status == 2]
    rv = rv[rv.status == 2][['t_hdr', 'lat', 'lon', 'alt']].rename(columns={'lat': 'r_lat', 'lon': 'r_lon', 'alt': 'r_alt'})
    fx = pd.merge_asof(fx, rv, on='t_hdr', direction='nearest', tolerance=60_000_000).dropna(subset=['r_lat'])
    if fx.empty:
        return None
    mx, my, mz = frame.to_enu(fx.lat.to_numpy(), fx.lon.to_numpy(), fx.alt.to_numpy())
    rx, ry, rz = frame.to_enu(fx.r_lat.to_numpy(), fx.r_lon.to_numpy(), fx.r_alt.to_numpy())
    okb = np.abs(np.hypot(rx - mx, ry - my) - 12.436) < 0.5          # пара с корректной базой
    e, n, u = (mx + K_BL * (rx - mx))[okb], (my + K_BL * (ry - my))[okb], (mz + K_BL * (rz - mz) - 3.0)[okb]
    fx = fx[okb]
    t = fx.t_hdr.to_numpy() / 1e9
    pos = pd.DataFrame({'t': t, 'e': e, 'n': n, 'u': u})
    # скачки: скорость по координатам против колёс
    w = pd.concat([run['front'], run['rear']]).sort_values('t_hdr')
    vw = np.interp(t, w.t_hdr.to_numpy() / 1e9, w.velocity.to_numpy() / 3.6)
    dt = np.diff(t, prepend=np.nan)
    vp = np.hypot(np.diff(e, prepend=np.nan), np.diff(n, prepend=np.nan)) / dt
    bad = (dt < 0.5) & (np.abs(vp - vw) > 3.0)
    # выбросы относительно скользящей медианы ±1 с (в т.ч. по высоте — status=2 бывает со скачками в десятки м)
    tix = pd.to_datetime(pos.t, unit='s')
    med = pos[['e', 'n', 'u']].set_index(tix).rolling('2s', center=True, min_periods=5).median()         .reset_index(drop=True)
    few = pos[['e']].set_index(tix).rolling('2s', center=True).count().to_numpy()[:, 0] < 8
    bad |= few
    bad |= (np.hypot(pos.e - med.e, pos.n - med.n) > 2.0).to_numpy() | (np.abs(pos.u - med.u) > 2.0).to_numpy()
    pos = pos[~bad].reset_index(drop=True)
    vel = run['master_vel'].sort_values('t_hdr')
    tv = vel.t_hdr.to_numpy() / 1e9
    vg = np.hypot(vel.vx.to_numpy(), vel.vy.to_numpy())
    vwv = np.interp(tv, w.t_hdr.to_numpy() / 1e9, w.velocity.to_numpy() / 3.6)
    velr = pd.DataFrame({'t': tv, 'v': vg, 'v_wheel': vwv, 'glitch': np.abs(vg - vwv) > 1.0})
    return {'frame': frame, 'pos': pos, 'vel': velr}

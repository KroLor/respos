"""Пути выезда на линию: GNSS-треки прогонов, стартующих вне карты пути (депо, отстойные пути,
не отмеченные в OSM), от точки старта до устойчивого въезда на pathgraph.

Результат: map/approach_tracks.json (копируется в пакет: src/tram_backup_odometry/data/).
Каждый трек: lat/lon/h с шагом 1 м по пути, ребро и s точки въезда на карту.

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/build_approach.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bench'))
from common import MAP_JSON, ROOT, load_run, splits  # noqa: E402

R_E, LAT0, LON0 = 6378137.0, 55.804767, 37.419565
OFF_START = 5.0       # старт дальше этого от карты — «вне карты»
ON_DIST = 2.5         # точка на карте
ON_RUN = 30.0         # столько метров подряд на карте — въезд
K_BL = 9.873 / 12.436   # base_link = master + K_BL·(rover − master)
MAX_LEN = 300.0       # длиннее — это не выезд, а смещённый GNSS вдоль всей линии (30639_253671cc, 9c362687)
GEOID = json.loads(MAP_JSON.read_text(encoding='utf-8'))['meta']['geoid_offset_m']


def xy(lat, lon):
    return np.radians(lon - LON0) * R_E * np.cos(np.radians(LAT0)), np.radians(lat - LAT0) * R_E


def geo(x, y):
    return LAT0 + np.degrees(y / R_E), LON0 + np.degrees(x / (R_E * np.cos(np.radians(LAT0))))


S = pd.read_parquet(ROOT / 'map' / 'pathgraph_samples.parquet')
tree = cKDTree(S[['x', 'y']].to_numpy())
sp = splits()
tracks = []
for b in sp['train'] + sp['val']:
    run = load_run(b)
    m = run['master_fix'].sort_values('t_hdr')
    m = m[m.status >= 0]
    rv = run['rover_fix'].sort_values('t_hdr')
    if len(m) < 100 or rv.empty:
        continue
    # траектория base_link по паре антенн (tf организаторов), высота уровня рельса
    m = pd.merge_asof(m, rv[['t_hdr', 'lat', 'lon', 'alt']].rename(columns={'lat': 'r_lat', 'lon': 'r_lon', 'alt': 'r_alt'}),
                      on='t_hdr', direction='nearest', tolerance=60_000_000).dropna(subset=['r_lat'])
    m = m.assign(lat=m.lat + K_BL * (m.r_lat - m.lat), lon=m.lon + K_BL * (m.r_lon - m.lon),
                 alt=m.alt + K_BL * (m.r_alt - m.alt) - 3.0)
    if len(m) < 100:
        continue
    x, y = xy(m.lat.to_numpy(), m.lon.to_numpy())
    d, j = tree.query(np.c_[x, y])
    if d[0] < OFF_START:
        continue
    # сглаживание, отсев скачков и стоянок: трек по пути с шагом 1 м
    st = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = np.r_[True, np.diff(st) > 0.05]
    x, y, st, d, j = x[keep], y[keep], st[keep], d[keep], j[keep]
    h = m.alt.to_numpy()[keep]
    if len(x) < 10:
        print(f"{b}: старт в {d[0]:.0f} м от карты, почти без движения — пропуск")
        continue
    hd = np.arctan2(np.gradient(gaussian_filter1d(y, 3)), np.gradient(gaussian_filter1d(x, 3)))
    on = (d < ON_DIST) & (np.abs((S.hdg.to_numpy()[j] - hd + np.pi) % (2 * np.pi) - np.pi) < 0.5)
    k = None
    for i in np.nonzero(on)[0]:
        seg = (st >= st[i]) & (st <= st[i] + ON_RUN)
        if on[seg].all() and st[-1] > st[i] + ON_RUN:
            k = i
            break
    if k is None:
        print(f'{b}: старт в {d[0]:.0f} м от карты, на карту не выезжает — пропуск')
        continue
    if st[k] > MAX_LEN:
        print(f'{b}: старт в {d[0]:.0f} м от карты, «въезд» лишь через {st[k]:.0f} м — смещение GNSS, пропуск')
        continue
    sg = np.arange(0.0, st[k], 1.0)
    xs = gaussian_filter1d(np.interp(sg, st[:k + 1], x[:k + 1]), 2)
    ys = gaussian_filter1d(np.interp(sg, st[:k + 1], y[:k + 1]), 2)
    hs = np.interp(sg, st[:k + 1], h[:k + 1])
    la, lo = geo(xs, ys)
    jj = j[k]
    tracks.append({'bag': b, 'length_m': float(sg[-1]), 'merge_edge': int(S.edge.iloc[jj]),
                   'merge_s': float(S.s.iloc[jj]), 'lat': np.round(la, 8).tolist(), 'lon': np.round(lo, 8).tolist(),
                   'h': np.round(hs, 2).tolist()})
    print(f'{b}: старт в {d[0]:.0f} м от карты, въезд через {sg[-1]:.0f} м на ребро {S.edge.iloc[jj]} s={S.s.iloc[jj]:.0f}')

out = ROOT / 'map' / 'approach_tracks.json'
out.write_text(json.dumps({'note': 'пути выезда на линию из точек вне карты; h — эллипсоидальная высота GNSS',
                           'tracks': tracks}, ensure_ascii=False), encoding='utf-8')
shutil.copy(out, MAP_JSON.parent / 'approach_tracks.json')
print(f'треков: {len(tracks)} → {out}')

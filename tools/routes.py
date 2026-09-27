"""Анализ траекторий: одна ли линия у всех прогонов. Пишет reports/tracks.kml и таблицы.

- берёт master_fix (rover — если master пуст), отбрасывает скачки > 30 м/с;
- прореживает до точки каждые ~5 м;
- считает попарное перекрытие прогонов (доля точек A в пределах 15 м от трека B)
  и группирует прогоны в маршруты;
- считает покрытие сети: сетка 20 м, сколько уникальных прогонов прошло через ячейку.
"""
import hashlib
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
EX, REP = ROOT / 'extracted', ROOT / 'reports'
REP.mkdir(exist_ok=True)
R = 6378137.0
NEAR_M = 15.0      # «тот же путь», м
STEP_M = 5.0       # прореживание трека, м
MAX_V = 30.0       # скачок GNSS, м/с

summary = pd.read_csv(EX / 'summary.csv')

# ---- уникальные прогоны (дубли — одинаковые скорости front)
groups = defaultdict(list)
for b in summary.bag:
    f = pd.read_parquet(EX / b / 'front.parquet')
    groups[hashlib.md5(f.velocity.round(4).to_numpy().tobytes()).hexdigest()].append(b)
dup_of = {b: v[0] for v in groups.values() for b in v}

# ---- общий центр для ENU
tracks = {}
for b in sorted(set(dup_of.values())):
    fx = None
    for rx in ('master', 'rover'):
        d = pd.read_parquet(EX / b / f'{rx}_fix.parquet')
        if len(d):
            fx = d.sort_values('t_hdr')
            break
    if fx is None:
        continue
    tracks[b] = fx[['t_hdr', 'lat', 'lon', 'status']].reset_index(drop=True)

lat0 = np.median(np.concatenate([t.lat for t in tracks.values()]))
lon0 = np.median(np.concatenate([t.lon for t in tracks.values()]))


def to_enu(lat, lon):
    return (np.radians(lon - lon0) * R * np.cos(np.radians(lat0)), np.radians(lat - lat0) * R)


clean, info = {}, []
for b, t in tracks.items():
    e, n = to_enu(t.lat.to_numpy(), t.lon.to_numpy())
    tt = t.t_hdr.to_numpy() / 1e9
    keep = [0]
    for i in range(1, len(t)):  # отбрасываем скачки относительно последней принятой точки
        j = keep[-1]
        if np.hypot(e[i] - e[j], n[i] - n[j]) / max(tt[i] - tt[j], 0.05) < MAX_V:
            keep.append(i)
    keep = np.array(keep)
    e, n, lat, lon = e[keep], n[keep], t.lat.to_numpy()[keep], t.lon.to_numpy()[keep]
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(e), np.diff(n)))])
    # прореживание по пройденному пути
    sel = [0]
    for i in range(1, len(dist)):
        if dist[i] - dist[sel[-1]] >= STEP_M:
            sel.append(i)
    sel = np.array(sel)
    clean[b] = pd.DataFrame({'e': e[sel], 'n': n[sel], 'lat': lat[sel], 'lon': lon[sel]})
    info.append({'bag': b, 'vehicle': b.split('_')[0], 'dups': len([x for x in dup_of if dup_of[x] == b]) - 1,
                 'fixes': len(t), 'dropped_jumps': len(t) - len(keep),
                 'path_km': dist[-1] / 1000, 'n_pts': len(sel),
                 'start_e': e[0], 'start_n': n[0], 'end_e': e[-1], 'end_n': n[-1],
                 'start_end_m': np.hypot(e[-1] - e[0], n[-1] - n[0])})
info = pd.DataFrame(info).set_index('bag')

# ---- попарное перекрытие
bags = [b for b in clean if len(clean[b]) >= 20]
trees = {b: cKDTree(clean[b][['e', 'n']].to_numpy()) for b in bags}
ov = pd.DataFrame(0.0, index=bags, columns=bags)
for a in bags:
    pa = clean[a][['e', 'n']].to_numpy()
    for b in bags:
        d, _ = trees[b].query(pa, distance_upper_bound=NEAR_M)
        ov.loc[a, b] = np.isfinite(d).mean()
ov.to_csv(REP / 'route_overlap.csv')

# ---- покрытие сетки и «уникальные» участки
cell = 20.0
cover = defaultdict(set)
for b in bags:
    for k in set(map(tuple, np.floor(clean[b][['e', 'n']].to_numpy() / cell).astype(int))):
        cover[k].add(b)
cnt = pd.Series({k: len(v) for k, v in cover.items()})
for b in bags:
    ks = set(map(tuple, np.floor(clean[b][['e', 'n']].to_numpy() / cell).astype(int)))
    info.loc[b, 'own_cells_%'] = 100 * np.mean([cnt[k] == 1 for k in ks])  # участки, где больше никто не ездил
    info.loc[b, 'overlap_any_%'] = 100 * ov.loc[b].drop(b).max()
    info.loc[b, 'overlap_all_mean_%'] = 100 * ov.loc[b].drop(b).mean()

info.round(2).to_csv(REP / 'routes_info.csv')

# ---- кластеризация: связываем прогоны с симметричным перекрытием > 50 %
sym = np.minimum(ov.to_numpy(), ov.to_numpy().T)
parent = list(range(len(bags)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(bags)):
    for j in range(i + 1, len(bags)):
        if sym[i, j] > 0.5:
            parent[find(i)] = find(j)
clusters = defaultdict(list)
for i, b in enumerate(bags):
    clusters[find(i)].append(b)
clusters = sorted(clusters.values(), key=len, reverse=True)
for ci, cl in enumerate(clusters):
    info.loc[cl, 'cluster'] = ci

# ---- вывод
pd.set_option('display.width', 220)
print(f'Уникальных прогонов с GNSS: {len(tracks)}, с треком ≥100 м: {len(bags)}')
print(f'Центр ENU: lat {lat0:.6f}, lon {lon0:.6f}')
allpts = np.vstack([clean[b][['e', 'n']].to_numpy() for b in bags])
print(f'Охват всех треков: E {allpts[:, 0].min():.0f}..{allpts[:, 0].max():.0f} м, '
      f'N {allpts[:, 1].min():.0f}..{allpts[:, 1].max():.0f} м')
print(f'Ячеек 20 м занято: {len(cnt)}; ≈ длина сети {len(cnt) * cell / 1000:.1f} км (грубо)')
print('Распределение: через сколько прогонов проходит ячейка')
print(cnt.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
print(f'\nКластеров (перекрытие >50 %): {len(clusters)}; размеры: {[len(c) for c in clusters]}')
print('\nПо прогонам:')
cols = ['vehicle', 'dups', 'path_km', 'start_end_m', 'overlap_any_%', 'overlap_all_mean_%', 'own_cells_%', 'dropped_jumps', 'cluster']
print(info.loc[bags, cols].sort_values(['cluster', 'overlap_all_mean_%']).round(1).to_string())

# ---- KML
PALETTE = ['ff0000ff', 'ff00a5ff', 'ff00ffff', 'ff00ff00', 'ffffff00', 'ffff0000', 'ffff00ff',
           'ff800080', 'ff008080', 'ff808000', 'ff0080ff', 'ff80ff00']


def coords(df):
    return ' '.join(f'{lo:.7f},{la:.7f},0' for la, lo in zip(df.lat, df.lon))


def line(name, desc, df, color, width=2):
    return (f'<Placemark><name>{escape(name)}</name><description>{escape(desc)}</description>'
            f'<Style><LineStyle><color>{color}</color><width>{width}</width></LineStyle></Style>'
            f'<LineString><tessellate>1</tessellate><coordinates>{coords(df)}</coordinates></LineString></Placemark>')


def point(name, la, lo, color, icon):
    return (f'<Placemark><name>{escape(name)}</name><Style><IconStyle><color>{color}</color><scale>0.7</scale>'
            f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/{icon}.png</href></Icon></IconStyle></Style>'
            f'<Point><coordinates>{lo:.7f},{la:.7f},0</coordinates></Point></Placemark>')


k = ['<?xml version="1.0" encoding="UTF-8"?>',
     '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Трамвай: треки GNSS</name>']
for veh in sorted(info.vehicle.unique()):
    k.append(f'<Folder><name>Трамвай {veh}</name>')
    for i, b in enumerate(sorted(b for b in bags if info.loc[b, 'vehicle'] == veh)):
        r = info.loc[b]
        col = PALETTE[i % len(PALETTE)]
        desc = (f'{r.path_km:.2f} км, кластер {int(r.cluster)}, перекрытие с другими: макс {r["overlap_any_%"]:.0f}% / '
                f'ср {r["overlap_all_mean_%"]:.0f}%, своих участков {r["own_cells_%"]:.0f}%, дублей {r.dups}')
        c = clean[b]
        k.append(f'<Folder><name>{b}</name>')
        k.append(line(b, desc, c, col))
        k.append(point('start ' + b, c.lat.iloc[0], c.lon.iloc[0], 'ff00ff00', 'placemark_circle'))
        k.append(point('end ' + b, c.lat.iloc[-1], c.lon.iloc[-1], 'ff0000ff', 'placemark_square'))
        k.append('</Folder>')
    k.append('</Folder>')

# слой покрытия: ячейки, окрашенные по числу прогонов
k.append('<Folder><name>Покрытие (сколько прогонов через ячейку 20 м)</name><visibility>0</visibility>')
mx = cnt.max()
for (ie, iN), c in cnt.items():
    ce, cn = (ie + 0.5) * cell, (iN + 0.5) * cell
    la = lat0 + np.degrees(cn / R)
    lo = lon0 + np.degrees(ce / (R * np.cos(np.radians(lat0))))
    t = c / mx
    rgb = (int(255 * (1 - t)), int(255 * t), 0)  # красный — редко, зелёный — часто
    k.append(f'<Placemark><visibility>0</visibility><name>{c}</name><Style><IconStyle><color>ff{rgb[2]:02x}{rgb[1]:02x}{rgb[0]:02x}</color>'
             f'<scale>0.3</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href></Icon></IconStyle>'
             f'<LabelStyle><scale>0</scale></LabelStyle></Style><Point><coordinates>{lo:.7f},{la:.7f},0</coordinates></Point></Placemark>')
k.append('</Folder></Document></kml>')
(REP / 'tracks.kml').write_text('\n'.join(k), encoding='utf-8')
print(f'\nKML: {REP / "tracks.kml"}')

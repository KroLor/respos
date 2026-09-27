"""Сборка карты путей (pathgraph) из OSM + GNSS обучающих прогонов + открытых ЦМР.

Источники:
  map/raw/osm.json            — OSM (Overpass): railway=tram, стрелки, остановки, переезды
  map/raw/cop30_N55_E037.tif  — Copernicus GLO-30 DEM (высоты над геоидом EGM2008)
  OpenTopoData srtm30m        — SRTM 30 м (кэш map/raw/srtm_cache.json)
  extracted/                  — GNSS (status=2) и скорости колёс

Выход:
  map/pathgraph.json   — граф: узлы, рёбра с профилем через 1 м, объекты, переходы, маршруты
  map/pathgraph.kml    — визуализация
  map/profile.png      — профили высоты/уклона/кривизны/ограничений вдоль основных маршрутов
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
EX, MAP = ROOT / 'extracted', ROOT / 'map'
RAW = MAP / 'raw'
R = 6378137.0
LAT0, LON0 = 55.804767, 37.419565   # начало локальной ENU карты
STEP = 1.0          # шаг профиля, м
MATCH_M = 3.0       # GNSS ↔ ребро
SUPPORT_MIN = 20    # минимум GNSS-точек, чтобы путь OSM вошёл в граф
BIN = 5.0           # бин для уточнения геометрии/высоты, м


def enu(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return np.radians(lon - LON0) * R * np.cos(np.radians(LAT0)), np.radians(lat - LAT0) * R


def geo(x, y):
    return LAT0 + np.degrees(np.asarray(y) / R), LON0 + np.degrees(np.asarray(x) / (R * np.cos(np.radians(LAT0))))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def resample(x, y, step=STEP):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s = np.arange(0, d[-1] + 1e-9, step)
    if s[-1] < d[-1] - 1e-6:
        s = np.r_[s, d[-1]]
    return s, np.interp(s, d, x), np.interp(s, d, y), d


def parse_speed(v):
    try:
        return float(str(v).split()[0])
    except (ValueError, IndexError):
        return np.nan


# =============================================================== 1. GNSS обучающих прогонов
print('1. GNSS…')
summary = pd.read_csv(EX / 'summary.csv')
seen, bags = set(), []
for b in summary.bag:
    f = pd.read_parquet(EX / b / 'front.parquet')
    h = hashlib.md5(f.velocity.round(4).to_numpy().tobytes()).hexdigest()
    if h not in seen:
        seen.add(h)
        bags.append(b)

# Карта строится по траектории base_link (ось вращения первой тележки, уровень рельса) — это точка, которую
# публикует решение и в которой судья считает эталон. tf антенн в base_link (данные организаторов):
# master x=−9.873, rover x=+2.563, обе z=+3.0  →  base_link = master + K_BL·(rover − master), высота − 3.0.
# По умолчанию карта строится по антенне master (все её точки → лучшие высоты и уклон), затем
# tools/map_to_base_link.py переводит её в base_link. Флаг --base-link строит сразу по паре антенн (для сравнения).
MASTER_X, ROVER_X, ANT_Z = -9.873, 2.563, 3.0
K_BL = -MASTER_X / (ROVER_X - MASTER_X)
BASE_LINK = '--base-link' in sys.argv


def _to_base_link(fx, rv):
    rv = rv[rv.status == 2].sort_values('t_hdr')[['t_hdr', 'lat', 'lon', 'alt']]
    fx = pd.merge_asof(fx, rv.rename(columns={'lat': 'r_lat', 'lon': 'r_lon', 'alt': 'r_alt'}), on='t_hdr',
                       direction='nearest', tolerance=60_000_000).dropna(subset=['r_lat'])
    mx, my = enu(fx.lat, fx.lon)
    rx, ry = enu(fx.r_lat, fx.r_lon)
    fx = fx[np.abs(np.hypot(rx - mx, ry - my) - (ROVER_X - MASTER_X)) < 0.5]
    return fx.assign(lat=fx.lat + K_BL * (fx.r_lat - fx.lat), lon=fx.lon + K_BL * (fx.r_lon - fx.lon),
                     alt=fx.alt + K_BL * (fx.r_alt - fx.alt) - ANT_Z)


G = []
for b in bags:
    fx = pd.read_parquet(EX / b / 'master_fix.parquet')
    if fx.empty:
        continue
    vel = pd.read_parquet(EX / b / 'master_vel.parquet').sort_values('t_hdr')
    fr = pd.read_parquet(EX / b / 'front.parquet').sort_values('t_hdr')
    fx = fx[fx.status == 2].sort_values('t_hdr')
    rv = pd.read_parquet(EX / b / 'rover_fix.parquet')
    if not BASE_LINK:
        rv = rv.iloc[:0]
    if rv.empty and BASE_LINK:
        continue
    if BASE_LINK:
        fx = _to_base_link(fx, rv)

    fx = pd.merge_asof(fx, vel[['t_hdr', 'vx', 'vy']], on='t_hdr', direction='nearest', tolerance=60_000_000)
    fx = pd.merge_asof(fx, fr[['t_hdr', 'velocity']], on='t_hdr', direction='nearest', tolerance=60_000_000)
    x, y = enu(fx.lat, fx.lon)
    t = fx.t_hdr.to_numpy() / 1e9
    ok = np.ones(len(fx), bool)          # отбрасываем скачки: скорость по позициям vs колёса
    v_pos = np.r_[0, np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 0.05)]
    ok &= v_pos < 30
    G.append(pd.DataFrame({'bag': b, 't': t, 'x': x, 'y': y, 'alt': fx.alt.to_numpy(),
                           'g': np.hypot(fx.vx, fx.vy).to_numpy(), 'hdg': np.arctan2(fx.vy, fx.vx).to_numpy(),
                           'vw': fx.velocity.to_numpy() / 3.6})[ok])
G = pd.concat(G, ignore_index=True).dropna(subset=['x', 'y'])
print(f'   уникальных прогонов {len(bags)}, точек GNSS st=2: {len(G)}')

# =============================================================== 2. OSM → отбор путей
print('2. OSM…')
osm = json.load(open(RAW / 'osm.json', encoding='utf-8'))['elements']
ways = {e['id']: e for e in osm if e['type'] == 'way' and e['tags'].get('railway') == 'tram'}
node_ll = {}
for w in ways.values():
    for nid, g in zip(w['nodes'], w['geometry']):
        node_ll[nid] = (g['lat'], g['lon'])
node_tags = {e['id']: e.get('tags', {}) for e in osm if e['type'] == 'node'}
for e in osm:
    if e['type'] == 'node':
        node_ll.setdefault(e['id'], (e['lat'], e['lon']))

# поддержка каждого OSM-пути GNSS-точками
dens, dens_w = [], []
for wid, w in ways.items():
    x, y = enu([g['lat'] for g in w['geometry']], [g['lon'] for g in w['geometry']])
    _, xs, ys, _ = resample(x, y, 0.5)
    dens.append(np.c_[xs, ys])
    dens_w += [wid] * len(xs)
tree = cKDTree(np.vstack(dens))
dens_w = np.array(dens_w)
d, i = tree.query(G[['x', 'y']].to_numpy(), distance_upper_bound=MATCH_M)
support = Counter(dens_w[i[np.isfinite(d)]])
kept = {wid for wid, c in support.items() if c >= SUPPORT_MIN}
print(f'   путей OSM: {len(ways)}, с поддержкой GNSS ≥{SUPPORT_MIN}: {len(kept)}')

# =============================================================== 3. граф
print('3. Граф…')
occ = Counter()
for wid in kept:
    ns = ways[wid]['nodes']
    occ.update(set(ns))
    occ[ns[0]] += 1          # концы пути — всегда узлы графа
    occ[ns[-1]] += 1
is_vertex = {n for n, c in occ.items() if c >= 2}

pieces = []   # (узлы, теги пути)
for wid in sorted(kept):
    w = ways[wid]
    ns = w['nodes']
    cut = [0] + [k for k in range(1, len(ns) - 1) if ns[k] in is_vertex] + [len(ns) - 1]
    for a, b in zip(cut[:-1], cut[1:]):
        pieces.append({'nodes': ns[a:b + 1], 'way': wid, 'tags': w['tags'],
                       'oneway': w['tags'].get('oneway') == 'yes'})

# склейка цепочек через узлы степени 2
adj = defaultdict(list)
for k, p in enumerate(pieces):
    adj[p['nodes'][0]].append(k)
    adj[p['nodes'][-1]].append(k)
used, chains = set(), []


def oriented(k, start):
    p = pieces[k]
    if p['nodes'][0] == start:
        return p['nodes'], +1
    return p['nodes'][::-1], -1


for k0 in range(len(pieces)):
    if k0 in used:
        continue
    used.add(k0)
    chain = [(k0, +1)]
    # расширяем вперёд и назад, пока узел проходной (ровно 2 куска) и совместим по oneway
    for forward in (True, False):
        while True:
            k, sgn = chain[-1] if forward else chain[0]
            nodes = pieces[k]['nodes'] if sgn > 0 else pieces[k]['nodes'][::-1]
            end = nodes[-1] if forward else nodes[0]
            nb = [q for q in adj[end] if q != k]
            if len(adj[end]) != 2 or len(nb) != 1 or nb[0] in used:
                break
            q = nb[0]
            if pieces[q]['oneway'] != pieces[k]['oneway']:
                break
            qn, qs = oriented(q, end) if forward else (lambda r: (r[0][::-1], -r[1]))(oriented(q, end))
            if pieces[q]['oneway'] and qs != sgn:
                break
            used.add(q)
            if forward:
                chain.append((q, qs))
            else:
                chain.insert(0, (q, qs))
    chains.append(chain)

edges = []
for ch in chains:
    nodes, tag_idx = [], []
    for k, sgn in ch:
        ns = pieces[k]['nodes'] if sgn > 0 else pieces[k]['nodes'][::-1]
        ns = ns if not nodes else ns[1:]
        nodes += ns
        tag_idx += [k] * len(ns)
    oneway = pieces[ch[0][0]]['oneway']
    if oneway and ch[0][1] < 0:          # ориентируем ребро по направлению движения
        nodes, tag_idx = nodes[::-1], tag_idx[::-1]
    edges.append({'nodes': nodes, 'piece_of_vertex': tag_idx, 'oneway': oneway,
                  'ways': sorted({pieces[k]['way'] for k, _ in ch})})
vertices = sorted({e['nodes'][0] for e in edges} | {e['nodes'][-1] for e in edges})
print(f'   рёбер {len(edges)}, узлов {len(vertices)}')

# =============================================================== 4. профиль рёбер 1 м
S = []   # таблица сэмплов всех рёбер
for eid, e in enumerate(edges):
    la, lo = zip(*[node_ll[n] for n in e['nodes']])
    x, y = enu(la, lo)
    s, xs, ys, d = resample(x, y)
    # теги пути на каждом сэмпле (по ближайшей вершине OSM слева)
    vk = np.clip(np.searchsorted(d, s, side='right') - 1, 0, len(e['nodes']) - 2)
    tags = [pieces[e['piece_of_vertex'][k + 1]]['tags'] for k in vk]
    e['id'] = eid
    S.append(pd.DataFrame({'edge': eid, 's': s, 'x0': xs, 'y0': ys,
                           'maxspeed': [parse_speed(t.get('maxspeed')) for t in tags],
                           'bridge': [t.get('bridge') == 'yes' for t in tags],
                           'service': [t.get('service', '') for t in tags],
                           'incline_osm': [t.get('incline', '') for t in tags]}))
S = pd.concat(S, ignore_index=True)


def tangents(df, xc='x0', yc='y0'):
    out = np.zeros(len(df))
    for eid, g in df.groupby('edge'):
        dx = np.gradient(g[xc].to_numpy()) if len(g) > 1 else np.array([1.0])
        dy = np.gradient(g[yc].to_numpy()) if len(g) > 1 else np.array([0.0])
        out[g.index] = np.arctan2(dy, dx)
    return out


S['hdg0'] = tangents(S)

# =============================================================== 5. привязка GNSS к рёбрам
print('5. Привязка GNSS (последовательная, по связности графа)…')
st = cKDTree(S[['x0', 'y0']].to_numpy())
s_edge, s_hdg = S.edge.to_numpy(), S.hdg0.to_numpy()
e_oneway = np.array([e['oneway'] for e in edges])
# преемники: (ребро, направление) → варианты продолжения через узел выхода
entry = defaultdict(list)
for e in edges:
    entry[e['nodes'][0]].append((e['id'], 1))
    if not e['oneway']:
        entry[e['nodes'][-1]].append((e['id'], -1))


def exit_node(eid, d):
    return edges[eid]['nodes'][-1] if d > 0 else edges[eid]['nodes'][0]


succ = {(e['id'], d): [c for c in entry[exit_node(e['id'], d)] if c[0] != e['id']]
        for e in edges for d in (1, -1)}

choice = np.full(len(G), -1)
dirn = np.zeros(len(G), int)
XY = G[['x', 'y']].to_numpy()
balls = st.query_ball_point(XY, MATCH_M)
gh, mv, gbag = G.hdg.to_numpy(), G.g.to_numpy() > 1.0, G.bag.to_numpy()
sx0, sy0 = S.x0.to_numpy(), S.y0.to_numpy()
cur, cur_bag = None, None
for r in range(len(G)):
    if gbag[r] != cur_bag:
        cur, cur_bag = None, gbag[r]
    if not balls[r]:
        continue
    cand = {}            # ребро → (дистанция, индекс сэмпла, направление)
    for j in balls[r]:
        e = s_edge[j]
        dist = np.hypot(XY[r, 0] - sx0[j], XY[r, 1] - sy0[j])
        if mv[r]:
            dh = abs(wrap(gh[r] - s_hdg[j]))
            d = 1 if dh < np.radians(45) else (-1 if dh > np.radians(135) and not e_oneway[e] else 0)
            if d == 0:
                continue
        else:
            d = cur[1] if cur and cur[0] == e else 1
        if e not in cand or dist < cand[e][0]:
            cand[e] = (dist, j, d)
    if not cand:
        continue
    if cur and cur[0] in cand and (cand[cur[0]][2] == cur[1] or not mv[r]):
        pick = cur[0]                                  # остаёмся на текущем ребре
    else:
        nxt = [c for c, _ in succ.get(cur, [])] if cur else []
        opts = [e for e in cand if e in nxt] or list(cand)  # преемник, иначе — переинициализация
        pick = min(opts, key=lambda e: cand[e][0])
    _, j, d = cand[pick]
    choice[r], dirn[r] = j, d if mv[r] or (cur and cur[0] == pick) else 0
    cur = (pick, d)
G['si'] = choice
G['dir'] = dirn
M = G[G.si >= 0].copy()
M['edge'] = S.edge.to_numpy()[M.si]
M['s'] = S.s.to_numpy()[M.si]
hx, hy = np.cos(S.hdg0.to_numpy()[M.si]), np.sin(S.hdg0.to_numpy()[M.si])
dx, dy = M.x - S.x0.to_numpy()[M.si], M.y - S.y0.to_numpy()[M.si]
M['lat_off'] = hx * dy - hy * dx          # >0 — слева от направления ребра
M['s'] += hx * dx + hy * dy
print(f'   привязано {len(M) / len(G):.1%} точек')

# =============================================================== 6. уточнение геометрии и высота по GNSS
print('6. Уточнение геометрии, высоты…')
S['off'] = 0.0
S['z_gnss'] = np.nan
S['n_gnss'] = 0
for eid, g in M.groupby('edge'):
    idx = S.index[S.edge == eid]
    s = S.s.to_numpy()[idx]
    b = np.floor(g.s / BIN).astype(int)
    agg = g.groupby(b).agg(off=('lat_off', 'median'), z=('alt', 'median'), n=('alt', 'size'))
    agg = agg[agg.n >= 5]
    if len(agg) < 2:
        continue
    sc = (agg.index.to_numpy() + 0.5) * BIN
    off = np.interp(s, sc, median_filter(agg.off.to_numpy(), 5, mode='nearest'))
    z = np.interp(s, sc, median_filter(agg.z.to_numpy(), 5, mode='nearest'))
    # за пределами покрытия высоту не экстраполируем
    cover = (s >= sc.min() - BIN) & (s <= sc.max() + BIN)
    S.loc[idx, 'off'] = gaussian_filter1d(np.clip(off, -3, 3), 5 / STEP, mode='nearest')
    S.loc[idx, 'z_gnss'] = np.where(cover, gaussian_filter1d(z, 10 / STEP, mode='nearest'), np.nan)
    S.loc[idx, 'n_gnss'] = np.interp(s, sc, agg.n.to_numpy()).astype(int) * cover
# сдвиг по нормали к OSM → путь антенны master
S['x'] = S.x0 - np.sin(S.hdg0) * S.off
S['y'] = S.y0 + np.cos(S.hdg0) * S.off
S['lat'], S['lon'] = geo(S.x, S.y)

# =============================================================== 7. ЦМР
print('7. ЦМР…')
with rasterio.open(RAW / 'cop30_N55_E037.tif') as r:
    band = r.read(1)
    fr_, fc_ = ~r.transform * (S.lon.to_numpy(), S.lat.to_numpy())  # col,row (дробные)
    c0, r0 = np.floor(fr_ - 0.5).astype(int), np.floor(fc_ - 0.5).astype(int)
    tc, tr = fr_ - 0.5 - c0, fc_ - 0.5 - r0
    z = (band[r0, c0] * (1 - tc) * (1 - tr) + band[r0, c0 + 1] * tc * (1 - tr)
         + band[r0 + 1, c0] * (1 - tc) * tr + band[r0 + 1, c0 + 1] * tc * tr)
S['z_cop'] = z

cache_f = RAW / 'srtm_cache.json'
cache = json.load(open(cache_f)) if cache_f.exists() else {}
sub = S[(S.s % 30) < STEP].copy()          # SRTM ~30 м — берём точку каждые 30 м
keys = [f'{a:.5f},{b:.5f}' for a, b in zip(sub.lat, sub.lon)]
need = [k for k in dict.fromkeys(keys) if k not in cache]
for j in range(0, len(need), 100):
    chunk = need[j:j + 100]
    url = 'https://api.opentopodata.org/v1/srtm30m?locations=' + '|'.join(chunk)
    try:
        res = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'tram-hackathon'}), timeout=60))
        for k, rr in zip(chunk, res['results']):
            cache[k] = rr['elevation']
    except Exception as ex:  # сеть недоступна — работаем без SRTM
        print('   SRTM недоступен:', ex)
        break
    time.sleep(1.1)
json.dump(cache, open(cache_f, 'w'))
sub['z_srtm'] = [cache.get(k) for k in keys]
S['z_srtm'] = np.nan
for eid, g in sub.dropna(subset=['z_srtm']).groupby('edge'):
    idx = S.index[S.edge == eid]
    S.loc[idx, 'z_srtm'] = np.interp(S.s[idx], g.s, g.z_srtm.astype(float))

# сдвиг эллипсоид→ЦМР (геоид + систематика), по участкам не на мосту
ok = S.z_gnss.notna() & ~S.bridge
geoid_off = float((S.z_gnss - S.z_cop)[ok].median())
print(f'   z_gnss − z_cop (медиана, вне моста) = {geoid_off:.2f} м (≈ высота геоида + систематика)')

# итоговая высота: GNSS там, где есть покрытие (≥10 точек в бине);
# пропуски — линейно между ближайшими покрытыми точками / высотами узлов,
# узлы без покрытия — Copernicus, приведённый к GNSS локальным сдвигом.
S['z'] = np.where(S.z_gnss.notna() & (S.n_gnss >= 10), S.z_gnss - geoid_off, np.nan)
S['z_src'] = np.where(S.z.notna(), 'gnss', 'interp')
node_z = defaultdict(list)
for eid, g in S.groupby('edge'):
    e = edges[eid]
    if g.z.notna().any():
        zz = g.z.dropna()
        if zz.index[0] - g.index[0] < 10:
            node_z[e['nodes'][0]].append(zz.iloc[0])
        if g.index[-1] - zz.index[-1] < 10:
            node_z[e['nodes'][-1]].append(zz.iloc[-1])
local_off = float((S.z - S.z_cop).median())
for eid, g in S.groupby('edge'):
    e = edges[eid]
    idx = g.index
    z = g.z.to_numpy().copy()
    for k, n in ((0, e['nodes'][0]), (-1, e['nodes'][-1])):
        if np.isnan(z[k]):
            z[k] = np.median(node_z[n]) if node_z[n] else g.z_cop.iloc[k] + local_off
    ok = ~np.isnan(z)
    S.loc[idx, 'z'] = np.interp(g.s, g.s[ok], z[ok])

# непрерывность высоты в узлах: общая высота узла = медиана концов рёбер,
# концы рёбер подтягиваются к ней линейной «рампой» длиной до 30 м
ends = defaultdict(list)
for eid, g in S.groupby('edge'):
    ends[edges[eid]['nodes'][0]].append(g.z.iloc[0])
    ends[edges[eid]['nodes'][-1]].append(g.z.iloc[-1])
nz = {n: float(np.median(v)) for n, v in ends.items()}
for eid, g in S.groupby('edge'):
    s, L = g.s.to_numpy(), g.s.max()
    c0 = nz[edges[eid]['nodes'][0]] - g.z.iloc[0]
    c1 = nz[edges[eid]['nodes'][-1]] - g.z.iloc[-1]
    if L > 60:
        corr = c0 * np.clip(1 - s / 30, 0, 1) + c1 * np.clip(1 - (L - s) / 30, 0, 1)
    else:
        corr = c0 + (c1 - c0) * s / max(L, 1e-6)
    S.loc[g.index, 'z'] = g.z.to_numpy() + corr

# высота как непрерывное поле z(x, y): гауссово среднее (σ = 10 м) по надёжным GNSS-сэмплам
# в радиусе 30 м — убирает ступеньки на стыках рёбер и шум коротких рёбер
src = S[S.z_src == 'gnss']
tz = cKDTree(src[['x', 'y']].to_numpy())
zsrc, xsrc, ysrc = src.z.to_numpy(), src.x.to_numpy(), src.y.to_numpy()
XYs = S[['x', 'y']].to_numpy()
zf = S.z.to_numpy().copy()
for k, nb in enumerate(tz.query_ball_point(XYs, 30.0)):
    if nb:
        d2 = (xsrc[nb] - XYs[k, 0]) ** 2 + (ysrc[nb] - XYs[k, 1]) ** 2
        w = np.exp(-d2 / (2 * 10.0 ** 2))
        zf[k] = np.sum(w * zsrc[nb]) / np.sum(w)
S['z'] = zf

# =============================================================== 8. курс, кривизна, уклон
for eid, g in S.groupby('edge'):
    idx = g.index
    if len(g) < 3:
        S.loc[idx, ['hdg', 'curv', 'grade']] = [np.arctan2(g.y.iloc[-1] - g.y.iloc[0], g.x.iloc[-1] - g.x.iloc[0]), 0, 0]
        continue
    # курс — по слабо сглаженной геометрии; кривизна — по сильнее сглаженной (~20 м),
    # иначе рябь OSM-вершин даёт ложные «кривые» на прямых
    xs = gaussian_filter1d(g.x.to_numpy(), 3, mode='nearest')
    ys = gaussian_filter1d(g.y.to_numpy(), 3, mode='nearest')
    h = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    S.loc[idx, 'hdg'] = wrap(h)
    hs = gaussian_filter1d(h, 10 / STEP, mode='nearest')
    S.loc[idx, 'curv'] = gaussian_filter1d(np.gradient(hs, g.s.to_numpy()), 5 / STEP, mode='nearest')
    # dz/ds, >0 — подъём по направлению ребра (z уже сглажено как поле); трамвайные уклоны ≤ ~6–9 %
    zz = gaussian_filter1d(g.z.to_numpy(), 8 / STEP, mode='nearest')
    S.loc[idx, 'grade'] = np.clip(np.gradient(zz, g.s.to_numpy()), -0.09, 0.09)

# =============================================================== 9. объекты OSM на графе
print('9. Объекты…')
st2 = cKDTree(S[['x', 'y']].to_numpy())
feat = []
kinds = {'tram_stop': 'stop', 'switch': 'switch', 'tram_level_crossing': 'level_crossing',
         'tram_crossing': 'tram_crossing', 'railway_crossing': 'railway_crossing',
         'buffer_stop': 'buffer_stop', 'signal': 'signal', 'level_crossing': 'level_crossing'}
for nid, t in node_tags.items():
    k = kinds.get(t.get('railway'))
    if k is None and t.get('public_transport') == 'stop_position':
        k = 'stop'
    if k is None:
        continue
    x, y = enu(*node_ll[nid])
    dist, j = st2.query([x, y])
    if dist > (20 if k == 'stop' else 5):
        continue
    feat.append({'kind': k, 'osm_id': nid, 'name': t.get('name', ''), 'edge': int(S.edge.iat[j]),
                 's': round(float(S.s.iat[j]), 1), 'dist': round(float(dist), 1),
                 'lat': node_ll[nid][0], 'lon': node_ll[nid][1]})
feat = pd.DataFrame(feat).drop_duplicates(subset=['kind', 'edge', 's'])

# эмпирические остановки: колёса < 0.2 м/с дольше 5 с
stops = []
for b, g in M.groupby('bag'):
    g = g.sort_values('t')
    still = (g.vw < 0.2).to_numpy()
    grp = np.cumsum(np.r_[1, np.diff(still.astype(int)) != 0])
    for _, q in g[still].groupby(grp[still]):
        if q.t.iloc[-1] - q.t.iloc[0] >= 5:
            stops.append({'bag': b, 'edge': int(q.edge.mode().iat[0]), 's': float(q.s.median()),
                          'dur': q.t.iloc[-1] - q.t.iloc[0]})
stops = pd.DataFrame(stops)
emp = []
for eid, g in stops.groupby('edge'):
    g = g.sort_values('s')
    cl = np.cumsum(np.r_[1, np.diff(g.s) > 15])
    for _, q in g.groupby(cl):
        emp.append({'edge': eid, 's': round(q.s.median(), 1), 'n_stops': len(q), 'n_runs': q.bag.nunique(),
                    'dur_med': round(q.dur.median(), 1)})
emp = pd.DataFrame(emp)
osm_stops = feat[feat.kind == 'stop']
lc = feat[feat.kind == 'level_crossing']


def nearest_feat(r, fdf, tol):
    q = fdf[(fdf.edge == r.edge) & ((fdf.s - r.s).abs() < tol)]
    return q.iloc[(q.s - r.s).abs().argmin()] if len(q) else None


emp['type'] = [('station' if nearest_feat(r, osm_stops, 40) is not None else
                'crossing' if nearest_feat(r, lc, 30) is not None else 'other') for r in emp.itertuples()]
emp['name'] = [(lambda f: f['name'] if f is not None else '')(nearest_feat(r, osm_stops, 40)) for r in emp.itertuples()]
n_runs = M.bag.nunique()
passed = M.groupby('edge').bag.nunique()      # сколько прогонов проезжало ребро
emp['n_passed'] = emp.edge.map(passed).fillna(0).astype(int)
emp['p_stop'] = (emp.n_runs / emp.n_passed.clip(lower=1)).clip(upper=1).round(3)

# =============================================================== 10. переходы и маршруты
print('10. Переходы…')
edge_len = S.groupby('edge').s.max()
trans, seqs = Counter(), {}
for b, g in M[M.dir != 0].sort_values(['bag', 't']).groupby('bag'):
    ev = list(zip(g.edge, g.dir))
    runs = []
    for e in ev:          # сжимаем повторы
        if not runs or runs[-1] != e:
            runs.append(e)
    # оставляем только переходы, допустимые по графу (привязка уже по связности,
    # «не по графу» — это переинициализация после пропуска GNSS)
    clean = runs
    seqs[b] = clean
    for a, c in zip(clean[:-1], clean[1:]):
        trans[(a, c)] += 1


trans_out = defaultdict(dict)
for (a, c), n in trans.items():
    trans_out[f'{a[0]}:{a[1]}'][f'{c[0]}:{c[1]}'] = n
trans_prob = {k: {kk: round(vv / sum(v.values()), 3) for kk, vv in v.items()} for k, v in trans_out.items()}

# основные маршруты: самые частые полные последовательности (по направлению)
full = Counter(tuple(v) for v in seqs.values() if len(v) >= 3)
routes = []
for sq, n in full.most_common(6):
    L = sum(edge_len[e] for e, _ in sq)
    routes.append({'edges': [[int(e), int(d)] for e, d in sq], 'n_runs': n, 'length_m': round(float(L), 1)})

# =============================================================== 11. сохранение
print('11. Сохранение…')
nodes_out = []
for n in vertices:
    x, y = enu(*node_ll[n])
    deg = sum((e['nodes'][0] == n) + (e['nodes'][-1] == n) for e in edges)
    nodes_out.append({'id': int(n), 'lat': node_ll[n][0], 'lon': node_ll[n][1], 'x': round(float(x), 2),
                      'y': round(float(y), 2), 'degree': deg,
                      'kind': node_tags.get(n, {}).get('railway', 'junction' if deg > 2 else 'end' if deg == 1 else 'joint')})
edges_out = []
for e in edges:
    g = S[S.edge == e['id']]
    edges_out.append({
        'id': e['id'], 'from': int(e['nodes'][0]), 'to': int(e['nodes'][-1]), 'oneway': e['oneway'],
        'length_m': round(float(g.s.max()), 2), 'osm_ways': e['ways'],
        'gnss_support': int(M.edge.eq(e['id']).sum()),
        'profile': {c: [round(float(v), p) if pd.notna(v) else None for v in g[c]]
                    for c, p in [('s', 2), ('x', 3), ('y', 3), ('lat', 8), ('lon', 8), ('z', 2), ('grade', 5),
                                 ('hdg', 5), ('curv', 6), ('maxspeed', 0), ('z_gnss', 2), ('z_cop', 2), ('z_srtm', 1)]}
                   | {'bridge': g.bridge.astype(int).tolist()}})
pg = {
    'meta': {'version': 1, 'created': time.strftime('%Y-%m-%d %H:%M'),
             'origin': {'lat': LAT0, 'lon': LON0, 'note': 'x — восток, y — север, м (равнопромежуточная проекция от origin)'},
             'step_m': STEP, 'z': 'высота над геоидом, м (GNSS эллипсоидальная − сдвиг; вне покрытия — Copernicus GLO-30)',
             'geoid_offset_m': round(geoid_off, 3),
             'grade': 'dz/ds по направлению ребра (from→to), безразмерно',
             'curv': 'dψ/ds, 1/м, >0 — поворот влево по направлению ребра',
             'hdg': 'курс касательной, рад, от оси x (восток) против часовой',
             'sources': ['OpenStreetMap (ODbL) — геометрия, maxspeed, объекты',
                         'GNSS master status=2 обучающих прогонов — уточнение геометрии, высота, остановки, переходы',
                         'Copernicus GLO-30 DEM', 'SRTM 30 m (OpenTopoData)'],
             'n_runs': int(n_runs)},
    'nodes': nodes_out, 'edges': edges_out,
    'features': feat.to_dict('records'),
    'stops_empirical': emp.to_dict('records'),
    'transitions': {'counts': trans_out, 'prob': trans_prob},
    'routes': routes,
}
MAP.mkdir(exist_ok=True)
json.dump(pg, open(MAP / 'pathgraph.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'),
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
S.to_parquet(MAP / 'pathgraph_samples.parquet', index=False)

# =============================================================== 12. KML
def kcolor(r, g, b, a=255):
    return f'{a:02x}{b:02x}{g:02x}{r:02x}'


def grade_color(gr):
    t = float(np.clip(gr / 0.04, -1, 1))       # ±4 %
    return kcolor(int(255 * max(t, 0)), int(255 * (1 - abs(t))), int(255 * max(-t, 0)))


def speed_color(v):
    pal = {10: (255, 0, 0), 15: (255, 120, 0), 20: (255, 200, 0), 25: (230, 230, 0), 30: (150, 220, 0), 60: (0, 180, 255)}
    return kcolor(*pal.get(int(v), (200, 200, 200))) if pd.notna(v) else kcolor(200, 200, 200)


def pm_line(name, desc, la, lo, color, width=4, vis=1):
    c = ' '.join(f'{b:.8f},{a:.8f},0' for a, b in zip(la, lo))
    return (f'<Placemark><name>{escape(name)}</name><visibility>{vis}</visibility><description>{escape(desc)}</description>'
            f'<Style><LineStyle><color>{color}</color><width>{width}</width></LineStyle></Style>'
            f'<LineString><tessellate>1</tessellate><coordinates>{c}</coordinates></LineString></Placemark>')


def pm_point(name, desc, la, lo, color, icon, scale=0.8, vis=1):
    return (f'<Placemark><name>{escape(name)}</name><visibility>{vis}</visibility><description>{escape(desc)}</description>'
            f'<Style><IconStyle><color>{color}</color><scale>{scale}</scale><Icon><href>'
            f'http://maps.google.com/mapfiles/kml/shapes/{icon}.png</href></Icon></IconStyle></Style>'
            f'<Point><coordinates>{lo:.8f},{la:.8f},0</coordinates></Point></Placemark>')


def segments(g, key):   # разрезаем ребро на куски с одинаковым значением key
    v = g[key].to_numpy()
    cut = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(g)]
    for a, b in zip(cut[:-1], cut[1:]):
        yield g.iloc[max(a - 1, 0):b]


K = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
     '<name>Pathgraph трамвая</name>']
K.append('<Folder><name>Рёбра графа</name>')
for e in edges_out:
    g = S[S.edge == e['id']]
    desc = (f'ребро {e["id"]}: {e["length_m"]:.0f} м, oneway={e["oneway"]}, GNSS-точек {e["gnss_support"]}, '
            f'OSM {e["osm_ways"]}, сдвиг к GNSS med {g.off.abs().median():.2f} м')
    K.append(pm_line(f'e{e["id"]}', desc, g.lat, g.lon, kcolor(255, 255, 255) if e['gnss_support'] > 200 else kcolor(160, 160, 160), 3))
K.append('</Folder><Folder><name>Уклон (красный — подъём по ребру, синий — спуск, ±4 %)</name><visibility>0</visibility>')
S['gb'] = (S.grade * 1000 // 5).astype(int)
for eid, g in S.groupby('edge'):
    for q in segments(g.reset_index(drop=True), 'gb'):
        K.append(pm_line(f'{q.grade.mean() * 100:+.1f} %', f'ребро {eid}, s {q.s.iloc[0]:.0f}–{q.s.iloc[-1]:.0f} м, '
                         f'z {q.z.iloc[0]:.1f}→{q.z.iloc[-1]:.1f} м', q.lat, q.lon, grade_color(q.grade.mean()), 6, 0))
K.append('</Folder><Folder><name>Ограничения скорости OSM, км/ч</name><visibility>0</visibility>')
for eid, g in S.groupby('edge'):
    for q in segments(g.reset_index(drop=True).fillna({'maxspeed': -1}), 'maxspeed'):
        v = q.maxspeed.iat[0]
        K.append(pm_line(f'{v:.0f} км/ч' if v > 0 else 'нет', f'ребро {eid}', q.lat, q.lon, speed_color(v if v > 0 else np.nan), 6, 0))
K.append('</Folder><Folder><name>Кривизна (радиус &lt; 200 м)</name><visibility>0</visibility>')
S['rb'] = (S.curv.abs() > 1 / 200).astype(int)
for eid, g in S.groupby('edge'):
    for q in segments(g.reset_index(drop=True), 'rb'):
        if q.rb.iat[-1]:
            K.append(pm_line(f'R≈{1 / max(q.curv.abs().max(), 1e-6):.0f} м', f'ребро {eid}', q.lat, q.lon, kcolor(255, 0, 255), 7, 0))
K.append('</Folder>')
icons = {'stop': ('bus', kcolor(0, 120, 255)), 'switch': ('triangle', kcolor(255, 255, 0)),
         'level_crossing': ('caution', kcolor(255, 80, 0)), 'tram_crossing': ('cross-hairs', kcolor(200, 200, 200)),
         'railway_crossing': ('cross-hairs', kcolor(200, 200, 200)), 'buffer_stop': ('forbidden', kcolor(255, 0, 0)),
         'signal': ('flag', kcolor(0, 255, 0))}
for k, grp in feat.groupby('kind'):
    ic, col = icons.get(k, ('placemark_circle', kcolor(255, 255, 255)))
    K.append(f'<Folder><name>OSM: {k} ({len(grp)})</name>')
    for r in grp.itertuples():
        K.append(pm_point(r.name or k, f'ребро {r.edge}, s={r.s} м, OSM node {r.osm_id}', r.lat, r.lon, col, ic))
    K.append('</Folder>')
K.append('<Folder><name>Остановки по данным (колёса стоят ≥5 с)</name>')
for r in emp.itertuples():
    g = S[(S.edge == r.edge)]
    j = (g.s - r.s).abs().idxmin()
    col = {'station': kcolor(0, 200, 0), 'crossing': kcolor(255, 150, 0), 'other': kcolor(255, 0, 0)}[r.type]
    K.append(pm_point(f'{r.p_stop:.0%}', f'{r.type} {r.name}; остановились {r.n_runs} из {r.n_passed} проехавших, '
                      f'медиана {r.dur_med} с; ребро {r.edge}, s={r.s}', S.lat[j], S.lon[j], col, 'placemark_circle',
                      0.5 + r.p_stop))
K.append('</Folder><Folder><name>Узлы графа</name><visibility>0</visibility>')
for n in nodes_out:
    K.append(pm_point(str(n['id']), f'{n["kind"]}, степень {n["degree"]}', n['lat'], n['lon'],
                      kcolor(255, 255, 255), 'placemark_square', 0.5, 0))
K.append('</Folder></Document></kml>')
(MAP / 'pathgraph.kml').write_text('\n'.join(K), encoding='utf-8')

# =============================================================== 13. сводка
pd.set_option('display.width', 220)
print(f'\nРёбер: {len(edges_out)}, узлов: {len(nodes_out)}, суммарная длина {S.groupby("edge").s.max().sum() / 1000:.2f} км')
print('Сдвиг OSM → путь антенны (|off|, м):', S.off.abs().describe(percentiles=[.5, .9, .99]).round(2).to_dict())
cmp = S[S.z_gnss.notna()]
for c in ('z_cop', 'z_srtm'):
    dz = (cmp.z_gnss - geoid_off - cmp[c]).dropna()
    print(f'Высота GNSS vs {c}: med {dz.median():+.2f}, СКО {dz.std():.2f}, p95|d| {dz.abs().quantile(.95):.2f} м '
          f'(на мосту med {(cmp.z_gnss - geoid_off - cmp[c])[cmp.bridge].median():+.2f})')
print(f'Уклон, %: |p50| {S.grade.abs().median() * 100:.2f}, |p95| {S.grade.abs().quantile(.95) * 100:.2f}, '
      f'max {S.grade.abs().max() * 100:.2f}; z {S.z.min():.1f}..{S.z.max():.1f} м')
print(f'Объекты OSM на графе: {feat.kind.value_counts().to_dict()}')
print(f'Остановок по данным: {len(emp)} мест; {emp.type.value_counts().to_dict()}')
print('Маршруты (частые последовательности рёбер):')
for r in routes:
    print(f'   {r["n_runs"]} прогонов, {r["length_m"]:.0f} м: {r["edges"]}')

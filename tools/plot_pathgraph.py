"""Профили вдоль двух основных маршрутов pathgraph → map/profile.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / 'map'
pg = json.load(open(MAP / 'pathgraph.json', encoding='utf-8'))
S = pd.read_parquet(MAP / 'pathgraph_samples.parquet')
emp = pd.DataFrame(pg['stops_empirical'])
feat = pd.DataFrame(pg['features'])

# по одному самому длинному маршруту на каждое направление (различаются первым ребром)
routes, seen = [], set()
for r in sorted(pg['routes'], key=lambda r: -r['length_m']):
    key = frozenset(e for e, _ in r['edges'][2:-2])
    if not any(len(key & k) > len(key) / 2 for k in seen):
        seen.add(key)
        routes.append(r)
routes = routes[:2]


def chain(route):
    parts, off, marks = [], 0.0, []
    for eid, d in route['edges']:
        g = S[S.edge == eid].sort_values('s', ascending=d > 0).copy()
        L = g.s.max()
        g['sr'] = off + (g.s if d > 0 else L - g.s)
        g['grade_r'] = g.grade * d
        parts.append(g)
        for q in emp[emp.edge == eid].itertuples():
            marks.append((off + (q.s if d > 0 else L - q.s), q))
        for q in feat[(feat.edge == eid) & (feat.kind == 'stop')].itertuples():
            marks.append((off + (q.s if d > 0 else L - q.s), q))
        off += L
    return pd.concat(parts), marks


fig, ax = plt.subplots(4, len(routes), figsize=(9 * len(routes), 12), sharex='col', squeeze=False)
for c, r in enumerate(routes):
    P, marks = chain(r)
    km = P.sr / 1000
    a = ax[0, c]
    a.plot(km, P.z, 'k', lw=2, label='итог (GNSS, ~70 проходов)')
    a.plot(km, P.z_cop, color='tab:green', lw=0.8, alpha=.8, label='Copernicus GLO-30')
    a.plot(km, P.z_srtm, color='tab:orange', lw=0.8, alpha=.8, label='SRTM 30 м')
    br = P.bridge.to_numpy()
    a.fill_between(km, P.z.min() - 5, P.z.max() + 5, where=br, color='tab:blue', alpha=.15, label='мост (OSM)')
    a.set_ylabel('высота над геоидом, м')
    a.set_title(f'Маршрут {c + 1}: {r["length_m"] / 1000:.2f} км, рёбра {[e for e, _ in r["edges"]]}', fontsize=9)
    a.legend(fontsize=8, loc='best')
    a = ax[1, c]
    a.plot(km, P.grade_r * 100, 'tab:red')
    a.axhline(0, color='k', lw=.5)
    a.set_ylabel('уклон по ходу, %')
    a = ax[2, c]
    R = 1 / np.maximum(P.curv.abs(), 1e-4)
    a.semilogy(km, R, 'tab:purple')
    a.set_ylim(10, 1e4)
    a.set_ylabel('радиус кривой, м')
    a = ax[3, c]
    a.step(km, P.maxspeed, 'tab:blue', where='post', label='maxspeed OSM')
    for s, q in marks:
        if hasattr(q, 'n_runs'):
            col = {'station': 'green', 'crossing': 'orange', 'other': 'red'}[q.type]
            a.scatter(s / 1000, 2, s=10 + 60 * q.p_stop, color=col, alpha=.7)
        else:
            a.axvline(s / 1000, color='green', lw=.6, ls='--')
    a.set_ylabel('км/ч; ● остановки по данным\n(зел. — станция, оранж. — переезд, красн. — прочее)')
    a.set_xlabel('расстояние вдоль маршрута, км')
fig.tight_layout()
fig.savefig(MAP / 'profile.png', dpi=110)
print('ok', MAP / 'profile.png')

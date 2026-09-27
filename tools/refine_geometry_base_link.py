"""Уточнение плановой геометрии карты по траекториям base_link.

Геометрия pathgraph — линия OSM, сдвинутая к медиане GNSS антенны master. Но master закреплён на кузове
за задней тележкой (свес): на кривых он выносится наружу, и дуга карты длиннее пути, который проходит
base_link (ось первой тележки на рельсе). На R ≈ 20 м (депо, кольца) это +3 % длины: оценщик отстаёт
вдоль пути на метры. base_link = master + K·(rover − master) лежит на оси кузова над рельсом, поэтому
по парам антенн (оба фикса status=2, база 12.44 ± 0.5 м) получаем боковое смещение рельса относительно
текущей линии карты и сдвигаем её. Топология, id рёбер и высоты не меняются; профиль пересэмплируется
с шагом 1 м по новой длине, места остановок переносятся пропорционально.

Запуск (после tools/map_to_base_link.py; затем tools/build_approach.py и tools/build_branch_profiles.py):
PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/refine_geometry_base_link.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(ROOT / 'src' / 'tram_backup_odometry'))
from common import load_run, splits  # noqa: E402
from tram_backup_odometry.geo import EnuFrame  # noqa: E402

PKG = ROOT / 'src' / 'tram_backup_odometry' / 'data' / 'pathgraph.json'
MASTER_X, ROVER_X = -9.873, 2.563
K_BL = -MASTER_X / (ROVER_X - MASTER_X)
BIN = 5.0
MATCH = 2.5          # м: точка base_link относится к ребру ближе
MIN_N = 8            # точек в бине
CLIP = 1.5           # м: максимальный сдвиг
SMOOTH = 8.0         # м: σ сглаживания сдвига (геометрия рельса плавная; рябь удлиняет прямые)
TAPER = 5.0          # м у концов ребра: сдвиг плавно → 0 (стык рёбер в узлах не рвётся)


def base_points(fr):
    """(e, n, курс) точек base_link по train-прогонам."""
    out = []
    for bag in splits()['train']:
        run = load_run(bag)
        m, r = run['master_fix'], run['rover_fix']
        m = m[m.status == 2].sort_values('t_hdr')
        r = r[r.status == 2].sort_values('t_hdr')
        if len(m) < 50 or len(r) < 50:
            continue
        j = pd.merge_asof(m[['t_hdr', 'lat', 'lon', 'alt']],
                          r[['t_hdr', 'lat', 'lon', 'alt']].rename(columns={'lat': 'rl', 'lon': 'ro', 'alt': 'ra'}),
                          on='t_hdr', direction='nearest', tolerance=60_000_000).dropna()
        me, mn, _ = fr.to_enu(j.lat.to_numpy(), j.lon.to_numpy(), j.alt.to_numpy())
        re_, rn, _ = fr.to_enu(j.rl.to_numpy(), j.ro.to_numpy(), j.ra.to_numpy())
        d = np.hypot(re_ - me, rn - mn)
        ok = np.abs(d - (ROVER_X - MASTER_X)) < 0.5
        e = me + K_BL * (re_ - me)
        n = mn + K_BL * (rn - mn)
        out.append(np.c_[e[ok], n[ok], np.arctan2(rn - mn, re_ - me)[ok]])
    return np.vstack(out)


def main():
    g = json.loads(PKG.read_text(encoding='utf-8'))
    geoid = float(g['meta'].get('geoid_offset_m', 0.0))
    p0 = g['edges'][0]['profile']
    fr = EnuFrame(float(p0['lat'][0]), float(p0['lon'][0]), 150.0)
    P = base_points(fr)
    print('точек base_link:', len(P))
    tree = cKDTree(P[:, :2])
    rep, factor = [], {}
    for ed in g['edges']:
        p = ed['profile']
        lat, lon = np.asarray(p['lat'], float), np.asarray(p['lon'], float)
        z = np.asarray(p['z'], float)
        e, n, _ = fr.to_enu(lat, lon, z + geoid)
        s = np.asarray(p['s'], float)
        if len(s) < 3:
            continue
        hdg = np.unwrap(np.arctan2(np.gradient(n), np.gradient(e)))
        # точки у каждого сэмпла ребра: боковое смещение (влево +) и совпадение курса
        offs = [[] for _ in range(int(s[-1] // BIN) + 1)]
        for i in range(len(s)):
            for k in tree.query_ball_point([e[i], n[i]], MATCH):
                dx, dy, h = P[k, 0] - e[i], P[k, 1] - n[i], P[k, 2]
                if abs((h - hdg[i] + np.pi) % (2 * np.pi) - np.pi) > np.radians(30):
                    continue
                along = dx * np.cos(hdg[i]) + dy * np.sin(hdg[i])
                if abs(along) > 0.5:
                    continue
                offs[int(s[i] // BIN)].append(-dx * np.sin(hdg[i]) + dy * np.cos(hdg[i]))
        sc = np.array([(b + 0.5) * BIN for b, o in enumerate(offs) if len(o) >= MIN_N])
        if len(sc) < 2:
            continue
        ov = np.array([np.median(o) for o in offs if len(o) >= MIN_N])
        off = np.interp(s, sc, median_filter(ov, 5, mode="nearest"))
        cover = (s >= sc.min() - BIN) & (s <= sc.max() + BIN)
        off = np.where(cover, off, 0.0)
        off = gaussian_filter1d(np.clip(off, -CLIP, CLIP), SMOOTH, mode="nearest")
        w = np.clip(np.minimum(s, s[-1] - s) / TAPER, 0.0, 1.0)
        off *= w
        e2, n2 = e - np.sin(hdg) * off, n + np.cos(hdg) * off
        # пересэмплирование с шагом 1 м по новой длине
        c = np.r_[0.0, np.cumsum(np.hypot(np.diff(e2), np.diff(n2)))]
        L = float(c[-1])
        s_new = np.arange(0.0, L + 1e-9, 1.0)
        if L - s_new[-1] > 1e-6:
            s_new = np.r_[s_new, L]
        u = np.interp(s_new, c, s)                 # соответствующая старая координата
        E, N = np.interp(u, s, e2), np.interp(u, s, n2)
        la, lo, _ = fr.to_geodetic(E, N, np.interp(u, s, z) + geoid)
        ratio = np.gradient(u) / np.maximum(np.gradient(s_new), 1e-9)
        newp = {'s': np.round(s_new, 2).tolist(), 'lat': np.round(la, 8).tolist(), 'lon': np.round(lo, 8).tolist(),
                # x, y — в исходной локальной системе карты: тот же сдвиг, что и в ENU
                'x': np.round(np.interp(u, s, np.asarray(p['x'], float) + (e2 - e)), 3).tolist(),
                'y': np.round(np.interp(u, s, np.asarray(p['y'], float) + (n2 - n)), 3).tolist()}
        for key, vals in p.items():
            if key in newp or not isinstance(vals, list) or len(vals) != len(s):
                continue
            a = np.array([np.nan if v is None else v for v in vals], dtype=float) if not isinstance(vals[0], str) else None
            if a is None:
                newp[key] = [vals[int(round(x))] for x in np.clip(u, 0, len(vals) - 1)]
                continue
            v = np.interp(u, s, a)
            if key == 'grade':
                v = v * ratio                         # dz/ds при новой длине
            newp[key] = [None if not np.isfinite(x) else round(float(x), 5) for x in v]
        ed['profile'] = newp
        factor[int(ed['id'])] = (s, s_new, u)
        rep.append((int(ed['id']), round(float(s[-1]), 2), round(L, 2), round(100 * (L / s[-1] - 1), 2),
                    round(float(np.abs(off).max()), 2)))
        ed['length_m'] = round(L, 2)
    for st in g.get('stops_empirical', []):
        f = factor.get(int(st['edge']))
        if f is not None:
            s, s_new, u = f
            st['s'] = round(float(np.interp(st['s'], u, s_new)), 1)
    g['meta']['geometry'] = 'OSM + боковое смещение по траекториям base_link (tools/refine_geometry_base_link.py)'
    PKG.write_text(json.dumps(g, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    R = pd.DataFrame(rep, columns=['edge', 'len_old', 'len_new', 'd_%', 'max_shift_m'])
    R = R.reindex(R['d_%'].abs().sort_values(ascending=False).index)
    (ROOT / 'reports' / 'geometry_base_link.md').write_text(
        '# Уточнение геометрии карты по base_link\n\n' + R.to_markdown(index=False) + '\n', encoding='utf-8')
    print(R.head(20).to_string(index=False))


if __name__ == '__main__':
    main()

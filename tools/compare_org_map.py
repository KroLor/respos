"""Сравнение нашей карты (map/pathgraph.json) с картой организаторов (*_-_*.json в корне проекта):
поперечное расстояние, разница курса и высоты. Запуск: .venv/Scripts/python tools/compare_org_map.py [pathgraph.json]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'tram_backup_odometry'))
from tram_backup_odometry.geo import MgrsFrame  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'map' / 'pathgraph.json'
g = json.loads(path.read_text(encoding='utf-8'))
geoid = g['meta']['geoid_offset_m']
fr = MgrsFrame()
rows = []
for e in g['edges']:
    p = e['profile']
    x, y, _ = fr.to_enu(np.array(p['lat']), np.array(p['lon']), 0)
    rows.append(pd.DataFrame({'edge': e['id'], 'x': x, 'y': y, 'h': np.array(p['z']) + geoid}))
S = pd.concat(rows, ignore_index=True)
S['hdg'] = np.arctan2(S.groupby('edge').y.transform(np.gradient), S.groupby('edge').x.transform(np.gradient))
tr = cKDTree(S[['x', 'y']].to_numpy())
for f in sorted(ROOT.glob('*_-_*.json')):
    P = pd.DataFrame(json.loads(f.read_text(encoding='utf-8'))['points'])
    d, j = tr.query(P[['x', 'y']].to_numpy(), k=4)
    # ближайшая точка с тем же направлением
    dh = np.abs((S.hdg.to_numpy()[j] - P.tang.to_numpy()[:, None] + np.pi) % (2 * np.pi) - np.pi)
    k = np.argmin(np.where(dh < 0.5, d, np.inf), axis=1)
    dd = d[np.arange(len(P)), k]
    jj = j[np.arange(len(P)), k]
    # знаковое поперечное смещение (влево от их направления > 0)
    dx, dy = S.x.to_numpy()[jj] - P.x.to_numpy(), S.y.to_numpy()[jj] - P.y.to_numpy()
    lat = -dx * np.sin(P.tang) + dy * np.cos(P.tang)
    dz = S.h.to_numpy()[jj] - P.z.to_numpy()
    dhd = np.degrees((S.hdg.to_numpy()[jj] - P.tang.to_numpy() + np.pi) % (2 * np.pi) - np.pi)
    q = lambda a: ' '.join(f'{v:+.2f}' for v in np.percentile(a, [5, 50, 95]))  # noqa: E731
    print(f'{f.name}: {len(P)} точек; расстояние |d| med {np.median(dd):.2f} p95 {np.percentile(dd, 95):.2f} '
          f'max {dd.max():.2f} м; поперечное (p5/p50/p95) {q(lat)}; Δz {q(dz)} м; Δкурс {q(dhd)}°; '
          f'рёбра {sorted(set(S.edge.to_numpy()[jj]))}')

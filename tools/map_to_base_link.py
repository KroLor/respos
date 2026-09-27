"""Перевод карты пути, построенной по антенне master (tools/build_pathgraph.py), в карту для base_link —
точки, которую публикует решение (ось первой тележки на уровне рельса, tf организаторов):
  * высоты профиля −3.0 м (антенны на 3.0 м над base_link);
  * места остановок сдвигаются на +9.873 м вперёд по пути (base_link впереди master на столько же);
  * геометрия (рельсовый путь) и уклон не меняются; уклон оценщик берёт под антенной master (≈ середина вагона).
Результат: src/tram_backup_odometry/data/pathgraph.json (+ map/pathgraph_base_link.json).
Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/map_to_base_link.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'tram_backup_odometry'))
from tram_backup_odometry.pathmap import PathMap  # noqa: E402

MASTER_X, ANT_Z = -9.873, 3.0
src = ROOT / 'map' / 'pathgraph.json'
g = json.loads(src.read_text(encoding='utf-8'))
pm = PathMap(str(src))
for e in g['edges']:
    p = e['profile']
    p['z'] = [round(z - ANT_Z, 3) for z in p['z']]
    if 'z_gnss' in p:
        p['z_gnss'] = [None if z is None else round(z - ANT_Z, 3) for z in p['z_gnss']]
moved = 0
for st in g.get('stops_empirical', []):
    eid, s, over = pm.advance(int(st['edge']), float(st['s']), -MASTER_X)
    if over > 0:
        continue
    moved += int(eid != st['edge'])
    st['edge'], st['s'] = int(eid), round(float(s), 1)
g['meta']['reference_point'] = ('base_link: ось вращения первой тележки на уровне рельса; геометрия — путь антенны '
                                'master по рельсам, высоты −3.0 м, остановки +9.873 м по ходу')
g['meta']['grade_back_m'] = -MASTER_X
out = ROOT / 'map' / 'pathgraph_base_link.json'
out.write_text(json.dumps(g, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
dst = ROOT / 'src' / 'tram_backup_odometry' / 'data' / 'pathgraph.json'
shutil.copy(out, dst)
print(f'остановок: {len(g.get("stops_empirical", []))}, перешли на соседнее ребро: {moved}; → {dst}')

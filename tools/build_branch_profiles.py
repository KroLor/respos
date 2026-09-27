"""Профили скорости по веткам стрелок: по какой ветке ушёл трамвай, видно по тому, как он едет после стрелки.

Для каждой стрелки карты (узел, из которого выходят ≥ 2 различимые ветки) по train-прогонам с GNSS
собираются проходы: трек base_link (по треку master, на 9.873 м впереди) проходит через узел, ветка определяется
по положению через CLS_M метров. Для каждой ветки — медиана и разброс скорости колёс по дистанции от узла
(шаг STEP_M, до PROF_M). Онлайн оценщик сравнивает свою скорость с профилями веток (правдоподобие) и,
если профиль другой ветки объясняет движение намного лучше, переставляет положение на неё.

Стрелка сохраняется, только если выбор по профилю на train (leave-one-out) не хуже выбора по плотности
проездов и точность ≥ MIN_ACC.

Результат: src/tram_backup_odometry/data/branch_profiles.json, отчёт reports/branch_profiles.md.
Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/build_branch_profiles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(ROOT / 'src' / 'tram_backup_odometry'))
from common import load_run, splits  # noqa: E402
from tram_backup_odometry.geo import MgrsFrame  # noqa: E402
from tram_backup_odometry.pathmap import PathMap, branch_fit  # noqa: E402

PKG = ROOT / 'src' / 'tram_backup_odometry' / 'data'
MASTER_X = 9.873      # base_link впереди антенны master, м
KMH = 3.5988
STEP_M = 2.0
PROF_M = 60.0        # длина профиля после узла
CLS_M = 45.0         # на этой дистанции определяется ветка прохода
CLS_MIN = 25.0       # (или на последней точке прогона, но не ближе)
NODE_R = 4.0         # проход через узел: трек ближе
MIN_SEP = 4.0        # ветки различимы, если их точки на CLS_M дальше друг от друга
MIN_PASS = 3
MIN_ACC = 0.9
SD_FLOOR = 0.5
SD_REL = 0.15
LLR, FIT_MAX = 6.0, 3.0   # как fork_llr, fork_fit_max оценщика


def base_track(run, fr):
    """Трек base_link: точка трека антенны master на MASTER_X м дальше по пути (от rover не зависит —
    в части прогонов rover сбоит). -> (t, e, n, пройденный путь, t колёс, v колёс)."""
    m = run['master_fix']
    m = m[m.status >= 0]
    if len(m) < 50:
        return None
    tm = m.t_bag.to_numpy() / 1e9
    me, mn, _ = fr.to_enu(m.lat.to_numpy(), m.lon.to_numpy(), m.alt.to_numpy())
    c = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(me), np.diff(mn)))])
    ok = c + MASTER_X <= c[-1]
    cu = np.maximum.accumulate(c)
    e = np.interp(c[ok] + MASTER_X, cu, me)
    n = np.interp(c[ok] + MASTER_X, cu, mn)
    f = run['front']
    ft, fv = f.t_bag.to_numpy() / 1e9, f.velocity.to_numpy() / KMH
    rr = run['rear']
    if len(rr) > 10:          # среднее тележек — меньше влияние сбоев одной
        fv = 0.5 * (fv + np.interp(ft, rr.t_bag.to_numpy() / 1e9, rr.velocity.to_numpy() / KMH))
    return tm[ok], e, n, c[ok], ft, fv


def profile(V):
    """Медиана и разброс скорости по бинам; -1 — бин без данных (онлайн пропускается).
    Разброс: полуширина 16–84 %, но не меньше SD_FLOOR и SD_REL·v (темп разгона у водителей разный),
    с поправкой на малую выборку √(1 + 3/n): по 3–4 проходам реальный разброс занижен."""
    med = np.nanmedian(V, 0)
    q16, q84 = np.nanpercentile(V, 16, 0), np.nanpercentile(V, 84, 0)
    n = np.sum(np.isfinite(V), 0)
    sd = np.maximum(np.nan_to_num(0.5 * (q84 - q16), nan=1.0), np.maximum(SD_FLOOR, SD_REL * np.nan_to_num(med)))
    sd = sd * np.sqrt(1.0 + 3.0 / np.maximum(n, 1))
    return np.where(np.isfinite(med), med, -1.0), sd


def main():
    fr = MgrsFrame()
    pm = PathMap(str(PKG / 'pathgraph.json'))
    pm.set_frame(fr)
    g = json.loads((PKG / 'pathgraph.json').read_text(encoding='utf-8'))
    out_of: dict = {}
    to_of: dict = {}
    for ed in g['edges']:
        if int(ed['id']) in pm.edges:
            out_of.setdefault(ed['from'], []).append(int(ed['id']))
            to_of[int(ed['id'])] = ed['to']

    def reach(b, L):
        """Точки всех путей с началом на ветке b (по всем преемникам) в пределах L м от узла."""
        pts, stack = [], [(b, 0.0)]
        while stack:
            eid, off = stack.pop()
            ed = pm.edges[eid]
            m = int(min(ed.length, L - off) / pm.step) + 1
            pts.append(np.c_[ed.e[:m], ed.n[:m]])
            if off + ed.length < L:
                stack += [(c, off + ed.length) for c in out_of.get(to_of[eid], []) if c != eid]
        return np.vstack(pts)
    forks = {}
    for node, br in out_of.items():
        if len(br) < 2:
            continue
        # точка каждой ветки через CLS_M (по цепочке преемников) — склеиваем неразличимые (параллельные рёбра)
        pts = {b: pm.pose(*pm.advance(b, 0.0, CLS_M)) for b in br}
        groups: list[list[int]] = []
        for b in sorted(br, key=lambda b: -pm.edges[b].density):
            for gr in groups:
                if np.hypot(pts[b].e - pts[gr[0]].e, pts[b].n - pts[gr[0]].n) < MIN_SEP:
                    gr.append(b)
                    break
            else:
                groups.append([b])
        if len(groups) >= 2:
            e0 = pm.edges[br[0]]
            # входящие рёбра: последние 25 м — проход должен прийти по ним (а не встречным ходом рядом)
            inc = [np.c_[pm.edges[k].e[-26:], pm.edges[k].n[-26:]] for k, t in to_of.items() if t == node]
            if not inc:
                continue
            cloud = [np.vstack([reach(b, CLS_M + 15.0) for b in gr]) for gr in groups]
            forks[node] = {'groups': groups, 'pts': pts, 'cloud': cloud, 'inc': np.vstack(inc), 'e': float(e0.e[0]), 'n': float(e0.n[0]),
                           'hdg': {b: float(pm.edges[b].hdg[0]) for b in br}, 'passes': []}
    print('стрелок с различимыми ветками:', len(forks))

    grid = np.arange(0.0, PROF_M + 1e-6, STEP_M)
    sp = splits()
    dbg = []
    for bag in sp['train']:
        tr = base_track(load_run(bag), fr)
        if tr is None:
            continue
        tm, e, n, c, ft, fv = tr
        for node, F in forks.items():
            d = np.hypot(e - F['e'], n - F['n'])
            i = 0
            while i < len(d):
                if d[i] >= NODE_R:
                    i += 1
                    continue
                j = i + int(np.argmin(d[i:i + 200]))
                i = j + 1
                while i < len(d) and d[i] < 2 * NODE_R:
                    i += 1
                c0 = c[j]
                cls = min(CLS_M, c[-1] - c0 - 1.0)        # прогон может кончиться вскоре после стрелки
                if cls < CLS_MIN or j + 2 >= len(e):
                    dbg.append((bag, node, f'коротко {cls:.0f}'))
                    continue
                # направление движения через узел
                k = int(np.searchsorted(c, c0 + 5.0))
                hd = np.arctan2(n[k] - n[j], e[k] - e[j])
                if min(abs((h - hd + np.pi) % (2 * np.pi) - np.pi) for h in F['hdg'].values()) > np.radians(45):
                    dbg.append((bag, node, 'курс'))
                    continue
                kb = max(int(np.searchsorted(c, c0 - 8.0)), 0)
                if c0 < 8.0 or np.min(np.hypot(F['inc'][:, 0] - e[kb], F['inc'][:, 1] - n[kb])) > 2.5:
                    dbg.append((bag, node, 'не по входящему'))
                    continue
                kc = int(np.searchsorted(c, c0 + cls))
                dist = {gi: float(np.min(np.hypot(cl[:, 0] - e[kc], cl[:, 1] - n[kc])))
                        for gi, cl in enumerate(F['cloud'])}
                srt = sorted(dist, key=dist.get)
                best = srt[0]
                if dist[best] > 2.5 or dist[srt[1]] < dist[best] + 2.0:
                    dbg.append((bag, node, f'ветка неоднозначна {dist}'))
                    continue
                tt = np.interp(c0 + grid, c, tm)
                v = np.interp(tt, ft, fv)
                v[c0 + grid > c[-1]] = np.nan          # прогон закончился раньше — профиль неполный
                F['passes'].append((bag, best, v))

    res, rep = [], ['# Профили скорости по веткам стрелок\n',
                    f'train-прогоны, шаг {STEP_M} м, профиль {PROF_M} м после узла, ветка по положению через {CLS_M} м.\n',
                    '| узел | ветки (группы рёбер) | проходов по веткам | точность по плотности | по профилю (LOO) | в пакет |',
                    '|---|---|---|---|---|---|']
    for node, F in forks.items():
        P = F['passes']
        cnt = {gi: sum(1 for p in P if p[1] == gi) for gi in range(len(F['groups']))}
        used = [gi for gi, k in cnt.items() if k >= MIN_PASS]

        if len(used) < 2:
            continue
        dens_acc = cnt[0] / max(len(P), 1)          # группа 0 — с наибольшей плотностью (текущий выбор)
        hits = 0
        for q, (_, lab, v) in enumerate(P):
            prof = {}
            for gi in used:
                V = np.array([p[2] for r, p in enumerate(P) if p[1] == gi and r != q])
                prof[gi] = profile(V)
            fit = {gi: branch_fit(v, *prof[gi]) for gi in used}
            pred = max(fit, key=lambda g: fit[g][0])
            if pred != 0 and (fit[pred][0] - fit[0][0] < LLR or fit[pred][1] > FIT_MAX):
                pred = 0
            hits += pred == lab
        acc = hits / len(P)
        keep = acc >= MIN_ACC and acc >= dens_acc and acc > dens_acc
        rep.append(f'| {node} | {F["groups"]} | {cnt} | {dens_acc:.2f} | {acc:.2f} | {"да" if keep else "нет"} |')
        if keep:
            br = {}
            for gi in used:
                V = np.array([p[2] for p in P if p[1] == gi])
                med, sd = profile(V)
                for b in F['groups'][gi]:
                    br[str(b)] = {'v': np.round(med, 3).tolist(), 'sd': np.round(sd, 3).tolist(), 'n': cnt[gi]}
            res.append({'node': node, 'branches': br})
    (PKG / 'branch_profiles.json').write_text(json.dumps({'step_m': STEP_M, 'forks': res}, ensure_ascii=False),
                                              encoding='utf-8')
    (ROOT / 'reports' / 'branch_profiles.md').write_text('\n'.join(rep) + '\n', encoding='utf-8')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()

"""Карта пути (pathgraph): привязка к пути и движение по дуговой координате s.

Геометрия хранится с шагом 1 м по каждому ориентированному ребру (все используемые рёбра — oneway).
На стрелке преемник выбирается только среди топологически связных рёбер, по плотности проходов GNSS
на метр пути (gnss_support / length): статистика переходов pathgraph содержит несвязные пары из-за
ошибок привязки на коротких рёбрах, поэтому для выбора ветки она не используется.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .geo import EnuFrame


def branch_fit(v, med, sd, sig_meas: float = 0.15, z2_cap: float = 9.0) -> tuple[float, float]:
    """Согласие профиля скорости v (по бинам дистанции от стрелки) с профилем ветки med ± sd.
    -> (робастный логарифм правдоподобия: вклад бина ограничен z2_cap — остановка или сбой не решает всё;
        средний z² — насколько ветка вообще объясняет движение)."""
    v = np.asarray(v, float)
    k = min(len(v), len(med))
    m, sd = np.asarray(med[:k], float), np.asarray(sd[:k], float)
    ok = np.isfinite(v[:k]) & np.isfinite(m) & (m >= 0.0) & np.isfinite(sd)   # m < 0 — бин без данных
    if not ok.any():
        return 0.0, float('inf')
    var = sd[ok] ** 2 + sig_meas ** 2
    z2 = (v[:k][ok] - m[ok]) ** 2 / var
    return float(np.sum(-0.5 * np.minimum(z2, z2_cap) - 0.5 * np.log(var))), float(np.mean(z2))


@dataclass
class Edge:
    id: int
    length: float
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray          # эллипсоидальная высота, м
    grade: np.ndarray
    hdg: np.ndarray        # курс касательной от оси east против часовой, рад
    curv: np.ndarray
    density: float        # проходов GNSS на метр
    e: np.ndarray | None = None
    n: np.ndarray | None = None
    u: np.ndarray | None = None
    succ: int | None = None


@dataclass
class MapPose:
    e: float
    n: float
    u: float
    hdg: float
    grade: float
    curv: float
    lat: float
    lon: float
    h: float


class PathMap:
    def __init__(self, path: str):
        with open(path, encoding='utf-8') as f:
            g = json.load(f)
        self.step = float(g['meta'].get('step_m', 1.0))
        geoid = float(g['meta'].get('geoid_offset_m', 0.0))
        self.edges: dict[int, Edge] = {}
        start_of: dict[int, list[int]] = {}
        ends: dict[int, int] = {}
        for e in g['edges']:
            p = e['profile']
            z = np.asarray(p['z'], float)
            ed = Edge(id=int(e['id']), length=float(p['s'][-1]),
                      lat=np.asarray(p['lat'], float), lon=np.asarray(p['lon'], float), h=z + geoid,
                      grade=np.nan_to_num(np.asarray(p['grade'], float)),
                      hdg=np.asarray(p['hdg'], float), curv=np.nan_to_num(np.asarray(p['curv'], float)),
                      density=float(e.get('gnss_support', 0)) / max(float(e['length_m']), 1.0))
            self.edges[ed.id] = ed
            start_of.setdefault(e['from'], []).append(ed.id)
            ends[ed.id] = e['to']
        for eid, ed in self.edges.items():
            cand = [c for c in start_of.get(ends[eid], []) if c != eid]
            if cand:
                ed.succ = max(cand, key=lambda c: self.edges[c].density)
        # предшественник — для сдвига назад при коррекции (ребро с наибольшей плотностью, ведущее в это)
        self.pred: dict[int, int] = {}
        for eid, ed in self.edges.items():
            if ed.succ is not None:
                cur = self.pred.get(ed.succ)
                if cur is None or ed.density > self.edges[cur].density:
                    self.pred[ed.succ] = eid
        # эмпирические места остановок (станции, стоп-линии): положение повторяется с IQR ≲ 1 м
        self.stops: dict[int, list[tuple[float, str, float]]] = {}
        for st in g.get('stops_empirical', []):
            if st.get('n_stops', 0) >= 5 and st.get('p_stop', 0.0) >= 0.1:
                self.stops.setdefault(int(st['edge']), []).append(
                    (float(st['s']), st.get('type', ''), float(st['p_stop'])))
        # пути выезда на линию из точек вне карты (tools/build_approach.py), если файл лежит рядом
        self.approach: list[dict] = []
        ap = os.path.join(os.path.dirname(os.path.abspath(path)), 'approach_tracks.json')
        if os.path.exists(ap):
            with open(ap, encoding='utf-8') as f:
                for t in json.load(f).get('tracks', []):
                    self.approach.append({'lat': np.asarray(t['lat'], float), 'lon': np.asarray(t['lon'], float),
                                          'h': np.asarray(t['h'], float), 'edge': int(t['merge_edge']),
                                          's': float(t['merge_s']), 'bag': t.get('bag', '')})
        # профили скорости по веткам стрелок (tools/build_branch_profiles.py): ребро-ветка -> стрелка
        self.fork_step = 2.0
        self.fork_of: dict[int, dict] = {}
        bp = os.path.join(os.path.dirname(os.path.abspath(path)), 'branch_profiles.json')
        if os.path.exists(bp):
            with open(bp, encoding='utf-8') as f:
                d = json.load(f)
            self.fork_step = float(d.get('step_m', 2.0))
            for fk in d.get('forks', []):
                br = {int(k): (np.asarray(b['v'], float), np.asarray(b['sd'], float))
                      for k, b in fk['branches'].items() if int(k) in self.edges}
                if len(br) >= 2:
                    for b in br:
                        self.fork_of[b] = br
        self.frame: EnuFrame | None = None

    # ------------------------------------------------------------------ система координат
    def set_frame(self, frame: EnuFrame):
        self.frame = frame
        for ed in self.edges.values():
            ed.e, ed.n, ed.u = frame.to_enu(ed.lat, ed.lon, ed.h)
            # курс касательной в осях выбранной системы (в UTM сетка повёрнута на ~1.3° к географической)
            ed.hdg = np.unwrap(np.arctan2(np.gradient(ed.n), np.gradient(ed.e))) if len(ed.e) > 1 else ed.hdg
        for t in self.approach:
            t['e'], t['n'], t['u'] = frame.to_enu(t['lat'], t['lon'], t['h'])
            t['hdg'] = np.arctan2(np.gradient(t['n']), np.gradient(t['e']))

    # ------------------------------------------------------------------ привязка
    def match(self, e: float, n: float, hdg: float | None, max_dist: float = 6.0,
              max_dhdg: float = math.radians(45.0), min_density: float = 0.5):
        """Ближайшая точка пути к (e, n) с совпадающим направлением. -> (edge_id, s, dist) | None."""
        best = None
        for ed in self.edges.values():
            d2 = (ed.e - e) ** 2 + (ed.n - n) ** 2
            ok = d2 <= max_dist ** 2
            if hdg is not None:
                ok &= np.abs((ed.hdg - hdg + np.pi) % (2 * np.pi) - np.pi) <= max_dhdg
            if not ok.any():
                continue
            i = int(np.argmin(np.where(ok, d2, np.inf)))
            dist = math.sqrt(d2[i])
            # рёбра без проходов GNSS (тупики, съезды) — только если больше ничего нет
            score = dist + (0.0 if ed.density >= min_density else 5.0)
            if best is None or score < best[3]:
                best = (ed.id, i * self.step, dist, score)
        if best is None:
            return None
        eid, s, dist, _ = best
        # уточнение проекцией на касательную
        ed = self.edges[eid]
        i = int(round(s / self.step))
        c, sn = math.cos(ed.hdg[i]), math.sin(ed.hdg[i])
        s = float(np.clip(s + c * (e - ed.e[i]) + sn * (n - ed.n[i]), 0.0, ed.length))
        return eid, s, dist

    # ------------------------------------------------------------------ движение по пути
    def advance(self, eid: int, s: float, ds: float):
        """Сдвиг на ds (м) вдоль пути. -> (edge_id, s, overflow): overflow > 0 — съехали с конца карты."""
        s += ds
        if not math.isfinite(s):                 # защита: NaN/inf не должны зацикливать проход по рёбрам
            return eid, 0.0, 0.0
        while True:
            ed = self.edges[eid]
            if s < 0.0:
                pe = self.pred.get(eid)
                if pe is None:
                    return eid, 0.0, 0.0
                eid = pe
                s += self.edges[pe].length
                continue
            if s <= ed.length:
                return eid, s, 0.0
            if ed.succ is None:
                return eid, ed.length, s - ed.length
            s -= ed.length
            eid = ed.succ

    def pose(self, eid: int, s: float, overflow: float = 0.0) -> MapPose:
        ed = self.edges[eid]
        x = min(max(s / self.step, 0.0), len(ed.e) - 1.0)
        i = min(int(x), len(ed.e) - 2)
        w = x - i

        def lerp(a):
            return float(a[i] * (1.0 - w) + a[i + 1] * w)

        dh = (ed.hdg[i + 1] - ed.hdg[i] + np.pi) % (2 * np.pi) - np.pi
        hdg = float(ed.hdg[i] + w * dh)
        e, n = lerp(ed.e), lerp(ed.n)
        if overflow > 0.0:           # за концом карты — прямолинейная экстраполяция
            e += overflow * math.cos(hdg)
            n += overflow * math.sin(hdg)
        return MapPose(e=e, n=n, u=lerp(ed.u), hdg=hdg, grade=lerp(ed.grade) if overflow <= 0 else 0.0,
                       curv=lerp(ed.curv), lat=lerp(ed.lat), lon=lerp(ed.lon), h=lerp(ed.h))

    def stops_near(self, eid: int, s: float, window: float):
        """Остановки в пределах window по пути (вперёд по преемникам и назад по предшественникам).
        -> список (знаковое расстояние от текущей точки до остановки, тип, p_stop), по возрастанию |d|."""
        out = []
        # вперёд
        e, off = eid, -s
        while off <= window and e is not None:
            for ss, typ, pst in self.stops.get(e, []):
                d = off + ss
                if -window <= d <= window and (e != eid or ss >= s):
                    out.append((d, typ, pst))
            off += self.edges[e].length
            e = self.edges[e].succ
            if e == eid:
                break
        # назад
        e, off = eid, s
        while off <= window + self.edges[e].length:
            for ss, typ, pst in self.stops.get(e, []):
                d = -(off - ss)
                if -window <= d < 0 and (e != eid or ss < s):
                    out.append((d, typ, pst))
            pe = self.pred.get(e)
            if pe is None or pe == eid:
                break
            e = pe
            off += self.edges[e].length
        out.sort(key=lambda r: abs(r[0]))
        return out

    def match_approach(self, e: float, n: float, hdg: float | None, max_dist: float = 6.0,
                       max_dhdg: float = math.radians(45.0), head_m: float = 40.0):
        """Привязка к началу пути выезда (первые head_m метров). -> (индекс, s на треке, dist) | None."""
        best = None
        for k, t in enumerate(self.approach):
            m = min(len(t['e']), int(head_m) + 1)
            d = np.hypot(t['e'][:m] - e, t['n'][:m] - n)
            ok = d <= max_dist
            if hdg is not None:
                ok &= np.abs((t['hdg'][:m] - hdg + np.pi) % (2 * np.pi) - np.pi) <= max_dhdg
            if ok.any():
                i = int(np.argmin(np.where(ok, d, np.inf)))
                if best is None or d[i] < best[2]:
                    best = (k, float(i), float(d[i]))
        return best

    def approach_pose(self, k: int, s: float):
        t = self.approach[k]
        x = min(max(s, 0.0), len(t['e']) - 1.0)
        i = min(int(x), len(t['e']) - 2)
        w = x - i
        return (float(t['e'][i] * (1 - w) + t['e'][i + 1] * w), float(t['n'][i] * (1 - w) + t['n'][i + 1] * w),
                float(t['u'][i] * (1 - w) + t['u'][i + 1] * w), float(t['hdg'][i]))


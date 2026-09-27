"""Шаг 0: офлайн-проверка эхо-сети (ESN) на остатке физической модели привода.

Вопрос: насколько ESN (по истории команд, скорости, уклона — только прошлые данные) уменьшает ошибку
прогноза «только по модели» — это режим, в котором фильтр работает при пропадании/проскальзывании колёс.

Варианты прогноза ускорения:
  P    — физическая модель (таблица + инерционное звено + уклон), как в оценщике;
  PB   — P + смещение b: средний остаток за 10 с до начала, затухание 5 с (как смещение в текущем фильтре);
  E    — P + ESN (выходной слой обучен офлайн ridge на train);
  ER   — P + ESN, выходной слой дообучается онлайн (RLS) по ходу прогона до начала горизонта.
Метрики на val: доля объяснённой дисперсии остатка (один шаг) и ошибка скорости/пути при прогнозе
без колёс на горизонтах 5/10/30 с (замкнутый контур: в модель и сеть подаётся прогнозная скорость).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/esn_feasibility.py
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from common import MODEL_JSON, ROOT, load_run, splits

DT = 0.1
R_E, LAT0, LON0 = 6378137.0, 55.804767, 37.419565
RELEASE = 99
M = json.loads(MODEL_JSON.read_text(encoding='utf-8'))
U_GRID, V_GRID = np.asarray(M['u_grid']), np.asarray(M['v_grid'])
TABLE, REL_ROW = np.asarray(M['a_table']), np.asarray(M['a_release_m8'])
K_GRADE, TAU, SCALE = M['k_grade'], M['lag_tau_s'], M['wheel_scale']
K_CURVE = M.get('k_curve', 0.0)
OUT = ROOT / 'reports' / 'esn'


# ============================================================================ данные
def effective_u(u):
    out = np.asarray(u).copy()
    prev = 0
    for i, x in enumerate(out):
        if x != -8:
            prev = x
        elif prev < -8:
            out[i] = RELEASE
    return out


def run_grid(b: str) -> pd.DataFrame | None:
    S = pd.read_parquet(ROOT / 'map' / 'pathgraph_samples.parquet')
    tree = cKDTree(S[['x', 'y']].to_numpy())
    s_hdg, s_grade, s_curv = S.hdg.to_numpy(), S.grade.to_numpy(), S.curv.to_numpy()
    run = load_run(b)
    fr, re, cmd = run['front'], run['rear'], run['cmd']
    fx = run['master_fix']
    fx = fx[fx.status == 2].sort_values('t_hdr')
    if len(fr) < 600 or len(fx) < 300:
        return None
    t0 = min(fr.t_hdr.iloc[0], re.t_hdr.iloc[0]) / 1e9
    t1 = max(fr.t_hdr.iloc[-1], re.t_hdr.iloc[-1]) / 1e9
    tg = np.arange(t0, t1, DT)
    f, r = fr.sort_values('t_hdr'), re.sort_values('t_hdr')
    v = 0.5 * (np.interp(tg, f.t_hdr / 1e9, f.velocity) + np.interp(tg, r.t_hdr / 1e9, r.velocity)) * SCALE
    c = cmd.sort_values('t_hdr')
    idx = np.searchsorted(c.t_hdr.to_numpy() / 1e9, tg, side='right') - 1
    u = effective_u(np.where(idx >= 0, c.position.to_numpy()[np.clip(idx, 0, None)], 0))
    vel = run['master_vel'].sort_values('t_hdr')
    tf = fx.t_hdr.to_numpy() / 1e9
    x = np.radians(fx.lon.to_numpy() - LON0) * R_E * math.cos(math.radians(LAT0))
    y = np.radians(fx.lat.to_numpy() - LAT0) * R_E
    vx, vy = np.interp(tf, vel.t_hdr / 1e9, vel.vx), np.interp(tf, vel.t_hdr / 1e9, vel.vy)
    d, j = tree.query(np.c_[x, y], k=8, distance_upper_bound=4.0)
    hd = np.arctan2(vy, vx)
    gr = np.full(len(tf), np.nan)
    cv = np.zeros(len(tf))
    last, lastc = np.nan, 0.0
    for i in range(len(tf)):
        if np.hypot(vx[i], vy[i]) < 1.0:          # стоянка: уклон прежний
            gr[i], cv[i] = last, lastc
            continue
        for dd, jj in zip(d[i], j[i]):
            if np.isfinite(dd) and abs((s_hdg[jj] - hd[i] + np.pi) % (2 * np.pi) - np.pi) < 0.6:
                gr[i] = last = s_grade[jj]
                cv[i] = lastc = s_curv[jj]
                break
    ok = np.isfinite(gr)
    if ok.sum() < 100:
        return None
    gi = np.clip(np.searchsorted(tf, tg), 1, len(tf) - 1)
    near = np.minimum(np.abs(tf[gi] - tg), np.abs(tf[gi - 1] - tg)) < 0.5
    grade = np.where(near, np.interp(tg, tf[ok], gr[ok]), np.nan)
    a = np.gradient(gaussian_filter1d(v, 3), DT)
    curv = np.interp(tg, tf, cv)
    return pd.DataFrame({'bag': b, 't': tg, 'v': v, 'u': u, 'grade': grade, 'a': a, 'curv': curv})


def table_acc(u, v):
    u = np.asarray(u)
    out = np.empty(len(u))
    ui = np.clip(u, -15, 15).astype(int) + 15
    for k in np.unique(ui):
        m = (ui == k) & (u != RELEASE)
        out[m] = np.interp(v[m], V_GRID, TABLE[k])
    m = u == RELEASE
    out[m] = np.interp(v[m], V_GRID, REL_ROW)
    return out


def physics(u, v, grade, a0=0.0, curv=None):
    """Физическая модель вдоль заданной траектории скорости (для обучения остатка)."""
    ass = table_acc(u, v) - K_GRADE * np.nan_to_num(grade)
    if curv is not None:
        ass = ass - K_CURVE * np.abs(curv)
    al = 1.0 - math.exp(-DT / TAU)
    af = np.empty(len(ass))
    y = a0
    for i in range(len(ass)):
        y += al * (ass[i] - y)
        if v[i] <= 0.0 and y < 0.0:
            y = 0.0
        af[i] = y
    return af


# ============================================================================ ESN
class ESN:
    def __init__(self, n=200, rho=0.9, leak=0.3, win=0.5, density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < density)
        W *= rho / max(abs(np.linalg.eigvals(W)))
        self.W, self.leak, self.n = W, leak, n
        self.Win = rng.uniform(-win, win, (n, 8))
        self.w = None

    @staticmethod
    def inputs(u, v, a_phys, grade, u_prev1s):
        """Вход резервуара — только внешние воздействия (история команд и уклона): иначе сеть учится
        дифференцировать историю измеренной скорости и в замкнутом контуре (без колёс) разваливается."""
        rel = 1.0 if u == RELEASE else 0.0
        uu = -8.0 if u == RELEASE else float(u)
        up = -8.0 if u_prev1s == RELEASE else float(u_prev1s)
        g = 0.0 if not np.isfinite(grade) else grade
        return np.array([uu / 15.0, rel, float(uu > 0), float(uu < 0), 20.0 * g, (uu - up) / 15.0, 0.0, 0.0]),             np.array([v / 15.0, (v / 15.0) ** 2, a_phys, 1.0])

    def step(self, x, inp):
        return (1.0 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.Win @ inp[0])

    def feats(self, x, inp):
        """Выходной слой: состояние резервуара + мгновенные скорость и физическое ускорение (без истории)."""
        return np.concatenate([x, inp[0][:6], inp[1]])


def run_states(esn, g):
    """Состояния резервуара вдоль прогона (вход — истинная скорость) и признаки для выходного слоя."""
    u, v, gr, ap = g.u.to_numpy(), g.v.to_numpy(), g.grade.to_numpy(), g.a_phys.to_numpy()
    x = np.zeros(esn.n)
    Z = np.empty((len(g), esn.n + 10), np.float32)
    for i in range(len(g)):
        inp = esn.inputs(u[i], v[i], ap[i], gr[i], u[max(i - 10, 0)])
        x = esn.step(x, inp)
        Z[i] = esn.feats(x, inp)
    return Z


def _accum(args):
    esn, g = args
    Z = run_states(esn, g).astype(np.float64)
    m = (g.v.to_numpy() > 0.3) & np.isfinite(g.grade.to_numpy())
    Z, y = Z[m], g.res.to_numpy()[m]
    return Z.T @ Z, Z.T @ y


def _sse(args):
    esn, g = args
    Z = run_states(esn, g)
    m = (g.v.to_numpy() > 0.3) & np.isfinite(g.grade.to_numpy())
    y = g.res.to_numpy()[m]
    return float(np.sum((y - Z[m] @ esn.w) ** 2)), y


def train_esn(esn, grids, lam, ex):
    A, bvec = None, None
    for Ai, bi in ex.map(_accum, [(esn, g) for g in grids]):
        A = Ai if A is None else A + Ai
        bvec = bi if bvec is None else bvec + bi
    esn.w = np.linalg.solve(A + lam * np.eye(A.shape[0]), bvec)
    return esn


def onestep_r2(esn, grids, ex):
    res = list(ex.map(_sse, [(esn, g) for g in grids]))
    y = np.concatenate([r[1] for r in res])
    return 1.0 - sum(r[0] for r in res) / np.sum((y - y.mean()) ** 2)


# ============================================================================ прогноз без колёс
HOR = (50, 100, 300)          # шагов по 0.1 с: 5, 10, 30 с


def horizon_eval(args):
    esn, g = args
    u, v, gr, ap, a, res = (g[c].to_numpy() for c in ('u', 'v', 'grade', 'a_phys', 'a', 'res'))
    n = len(g)
    Hmax = HOR[-1]
    starts = [i for i in range(150, n - Hmax - 1, 150)
              if v[i] > 2.0 and np.isfinite(gr[i:i + Hmax]).all()]
    # теневой прогон по истинным данным: состояния резервуара и онлайн-RLS
    nf = esn.n + 10
    x = np.zeros(esn.n)
    states = {}
    w_rls = esn.w.copy()
    P = np.eye(nf) * 0.1
    lam_f = 0.9995
    wr = {}
    Zbuf = []
    sset = set(starts)
    for i in range(n):
        inp = esn.inputs(u[i], v[i], ap[i], gr[i], u[max(i - 10, 0)])
        x = esn.step(x, inp)
        z = esn.feats(x, inp)
        Zbuf.append(z)
        k = i - 5                                   # цель известна с задержкой 0.5 с (сглаживание)
        if k >= 0 and v[k] > 0.3 and np.isfinite(gr[k]):
            zk = Zbuf[k]
            Pz = P @ zk
            gain = Pz / (lam_f + zk @ Pz)
            w_rls = w_rls + gain * (res[k] - zk @ w_rls)
            P = (P - np.outer(gain, Pz)) / lam_f
            tr = np.trace(P)
            if tr > 10.0 * nf:                      # защита от «взрыва» P при слабом возбуждении
                P *= 10.0 * nf / tr
        if i in sset:
            states[i] = x.copy()
            wr[i] = w_rls.copy()
    al = 1.0 - math.exp(-DT / TAU)
    out = []
    for i0 in starts:
        b0 = float(np.nanmean(res[max(i0 - 100, 0):i0]))
        for var in ('P', 'PB', 'E', 'ER'):
            vs, af = v[i0], a[i0]
            xs = states[i0].copy()
            wv = esn.w if var == 'E' else wr[i0]
            dv, pos = {}, 0.0
            for h in range(1, Hmax + 1):
                k = i0 + h - 1
                ass = table_acc(np.array([u[k]]), np.array([vs]))[0] - K_GRADE * gr[k]
                af += al * (ass - af)
                if vs <= 0 and af < 0:
                    af = 0.0
                acc = af
                if var == 'PB':
                    acc += b0 * math.exp(-h * DT / 5.0)
                elif var in ('E', 'ER'):
                    inp = esn.inputs(u[k], vs, af, gr[k], u[max(k - 10, 0)])
                    xs = esn.step(xs, inp)
                    acc += float(esn.feats(xs, inp) @ wv)
                vn = max(vs + acc * DT, 0.0)
                pos += 0.5 * (vs + vn) * DT - 0.5 * (v[k] + v[k + 1]) * DT
                vs = vn
                if h in HOR:
                    dv[h] = (vs - v[i0 + h], pos)
            for h, (e, p) in dv.items():
                out.append({'bag': g.bag.iloc[0], 'i0': i0, 'var': var, 'H': h * DT, 'dv': e, 'dpos': p})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = splits()
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        tr = [g for g in ex.map(run_grid, sp['train']) if g is not None]
        va = [g for g in ex.map(run_grid, sp['val']) if g is not None]
    for g in tr + va:
        g['a_phys'] = physics(g.u.to_numpy(), g.v.to_numpy(), g.grade.to_numpy())
        g['res'] = g.a - g.a_phys
    print(f'сетки: train {len(tr)}, val {len(va)} прогонов, {time.time() - t0:.0f} с')
    # подбор гиперпараметров: внутренняя валидация на части train
    inner_tr, inner_va = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)):]
    ex = ProcessPoolExecutor(16)
    best = None
    for rho in (0.8, 0.95):
        for leak in (0.05, 0.1, 0.3):
            for lam in (1.0, 30.0):
                esn = train_esn(ESN(rho=rho, leak=leak), inner_tr, lam, ex)
                r2 = onestep_r2(esn, inner_va, ex)
                print(f'  rho={rho} leak={leak} lam={lam}: R² остатка (внутр. вал.) {r2:.3f}')
                if best is None or r2 > best[0]:
                    best = (r2, rho, leak, lam)
    _, rho, leak, lam = best
    esn = train_esn(ESN(rho=rho, leak=leak), tr, lam, ex)
    r2_val = onestep_r2(esn, va, ex)
    # доля объяснённой дисперсии полного ускорения
    num_p = num_e = den = 0.0
    for g in va:
        m = (g.v > 0.3) & g.grade.notna()
        Z = run_states(esn, g)[m.to_numpy()]
        a = g.a[m].to_numpy()
        num_p += np.sum((a - g.a_phys[m]) ** 2)
        num_e += np.sum((a - g.a_phys[m] - Z @ esn.w) ** 2)
        den += np.sum((a - a.mean()) ** 2)
    print(f'выбрано rho={rho} leak={leak} lam={lam}; val: R² остатка {r2_val:.3f}; '
          f'R² ускорения: физика {1 - num_p / den:.3f} → физика+ESN {1 - num_e / den:.3f}')
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(horizon_eval, [(esn, g) for g in va]) for r in part]
    H = pd.DataFrame(rows)
    H.to_csv(OUT / 'horizon.csv', index=False)
    tab = H.groupby(['H', 'var']).agg(v_rmse=('dv', lambda x: np.sqrt(np.mean(x ** 2))),
                                      v_mae=('dv', lambda x: np.mean(np.abs(x))),
                                      pos_rmse=('dpos', lambda x: np.sqrt(np.mean(x ** 2))),
                                      n=('dv', 'size')).round(3)
    print(tab.to_string())
    md = ['# ESN: офлайн-проверка (шаг 0)\n',
          f'- train {len(tr)} / val {len(va)} прогонов с уклоном; ESN 200 нейронов, rho={rho}, leak={leak}, ridge λ={lam}',
          f'- val, один шаг: R² остатка {r2_val:.3f}; R² ускорения: физика {1 - num_p / den:.3f} → физика+ESN {1 - num_e / den:.3f}',
          '- прогноз без колёс (замкнутый контур), старт каждые 15 с при v > 2 м/с; ошибка скорости в конце горизонта и '
          'пути за горизонт\n', tab.to_markdown()]
    (OUT / 'feasibility.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    np.savez(OUT / 'esn_offline.npz', W=esn.W, Win=esn.Win, w=esn.w, leak=esn.leak, rho=rho, lam=lam)


if __name__ == '__main__':
    main()

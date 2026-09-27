"""ESN, итерация 2: память до провала, контекст прогона, физически осмысленная поправка, ансамбль, NG-RC,
и честные контроли без сети. Все варианты — один и тот же прогноз «без колёс» в замкнутом контуре (PyTorch).

Варианты (обучение — многошаговое, горизонт 30 с, ранняя остановка по внутренней валидации 20 % train):
  P    — физическая модель, как в оценщике;
  C1   — физика, дообученная многошагово (множители тяги/торможения/выбега, коэф. уклона, τ, смещение) — контроль;
  C2   — адаптивная физика: те же множители оцениваются онлайн (МНК с забыванием) по истории с колёсами
         и замораживаются на время провала — контроль «адаптация без сети»;
  E0   — C1 + ESN (резервуар видит историю команд и уклона) — как в итерации 1;
  E1   — C1 + ESN с памятью: пока колёса есть, в резервуар идёт измеренный остаток модели и флаг;
  E2   — E1 + контекст прогона (онлайн-оценки множителей из C2) во входе выходного слоя;
  E3   — E2 + поправка в виде множителей к тяге/торможению (+ добавка);
  NG   — NG-RC: вместо резервуара — задержки остатка и команд (без случайных матриц);
  ENS  — ансамбль из 5 резервуаров лучшего варианта E*.
Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/esn_v2.py [--variants P,C1,...] [--epochs 40]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import esn_feasibility as F
from common import splits

DT = F.DT
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 5e-4
LAG_R = 5            # остаток известен с задержкой 0.5 с (сглаживание)
NG_LAGS = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80)


# ============================================================================ данные
def rls_context(g):
    """Онлайн-МНК (забывание ~200 с) множителей физики по истории с колёсами: y = gT·aT + gB·aB + g0·a0 + c."""
    u, v, gr, a = g.u.to_numpy(), g.v.to_numpy(), np.nan_to_num(g.grade.to_numpy()), g.a_c.to_numpy()
    cvv = np.abs(g.curv.to_numpy())
    tv = F.table_acc(u, v)
    comps = np.c_[tv * (u > 0) * (u != F.RELEASE), tv * ((u < 0) | (u == F.RELEASE)), tv * (u == 0)]
    al = 1.0 - math.exp(-DT / F.TAU)
    lagc = np.zeros_like(comps)
    lg = np.zeros(len(u))
    y = np.zeros(3)
    yg = 0.0
    for i in range(len(u)):
        y += al * (comps[i] - y)
        yg += al * (-F.K_GRADE * gr[i] - F.K_CURVE * cvv[i] - yg)
        lagc[i], lg[i] = y, yg
    th = np.array([1.0, 1.0, 1.0, 0.0])
    P = np.eye(4) * 0.02
    lam = 0.9995
    out = np.zeros((len(u), 4))
    for i in range(len(u)):
        k = i - LAG_R
        if k >= 0 and v[k] > 0.3 and np.isfinite(g.grade.iat[k]) and np.isfinite(a[k]):
            z = np.r_[lagc[k], 1.0]
            e = a[k] - lg[k] - z @ th
            Pz = P @ z
            gain = Pz / (lam + z @ Pz)
            th = th + gain * e
            P = (P - np.outer(gain, Pz)) / lam
            if np.trace(P) > 1.0:
                P *= 1.0 / np.trace(P)
        out[i] = th
    return out


def prep(g):
    g = g.copy()
    g['a_phys'] = F.physics(g.u.to_numpy(), g.v.to_numpy(), g.grade.to_numpy(), curv=g.curv.to_numpy())
    g['res'] = g.a - g.a_phys
    v = g.v.to_numpy()
    a_c = np.full(len(v), np.nan)
    a_c[LAG_R:-LAG_R] = (v[2 * LAG_R:] - v[:-2 * LAG_R]) / (2 * LAG_R * DT)      # как в ноде
    g['a_c'] = a_c
    r = a_c - g.a_phys.to_numpy()
    r_in = np.r_[np.zeros(LAG_R), r[:-LAG_R]]
    r_in[~np.isfinite(r_in) | (np.r_[np.zeros(LAG_R), g.v.to_numpy()[:-LAG_R]] <= 0.3)] = 0.0
    g['r_in'] = np.clip(r_in, -1.5, 1.5)
    th = rls_context(g)
    for j, c in enumerate(('thT', 'thB', 'th0', 'thc')):
        g[c] = th[:, j]
    return g


def exo(u, gr):
    """Внешние входы: позиция (знак/режимы), уклон, изменение позиции за 1 с."""
    rel = (u == F.RELEASE)
    uu = np.where(rel, -8.0, u).astype(float)
    up = np.r_[np.full(10, uu[0]), uu[:-10]]
    return np.c_[uu / 15.0, rel, uu > 0, uu < 0, 20.0 * np.nan_to_num(gr), (uu - up) / 15.0].astype(np.float32)


def build(grids):
    cat = lambda c: np.concatenate([g[c].to_numpy() for g in grids])  # noqa: E731
    u = cat('u')
    D = {'urow': np.where(u == F.RELEASE, 31, np.clip(u, -15, 15) + 15).astype(np.int64),
         'sgn': np.where(u == F.RELEASE, -1, np.sign(u)).astype(np.float32),
         'grade': cat('grade').astype(np.float32), 'v': cat('v').astype(np.float32), 'a': cat('a').astype(np.float32),
         'curv': np.abs(cat('curv')).astype(np.float32),
         'r_in': cat('r_in').astype(np.float32),
         'th': np.c_[cat('thT'), cat('thB'), cat('th0'), cat('thc')].astype(np.float32),
         'exo': np.vstack([exo(g.u.to_numpy(), g.grade.to_numpy()) for g in grids]),
         'run': np.concatenate([np.full(len(g), i) for i, g in enumerate(grids)])}
    D['lens'] = [len(g) for g in grids]
    # задержки остатка для NG-RC
    L = np.zeros((len(u), len(NG_LAGS)), np.float32)
    off = 0
    for g in grids:
        r = g.r_in.to_numpy()
        for j, lag in enumerate(NG_LAGS):
            L[off + lag:off + len(g), j] = r[:len(g) - lag]
        off += len(g)
    D['rlag'] = L
    return D


def windows(D, H, stride, v_min=1.0, first=150):
    run, gr, v = D['run'], np.isfinite(D['grade']), D['v']
    n = len(run)
    ok_run = np.r_[run[H:] == run[:-H], np.zeros(H, bool)]
    cg = np.r_[0, np.cumsum(~gr)]
    ok_gr = np.zeros(n, bool)
    ok_gr[: n - H] = (cg[H:n] - cg[: n - H]) == 0
    pos_in_run = np.concatenate([np.arange(L) for L in D['lens']])
    s = np.nonzero(ok_run & ok_gr & (v > v_min) & (pos_in_run >= first))[0]
    return s[::stride]


def val_windows(D, H):
    """Как в шаге 0: старт каждые 15 с при v > 2 м/с, уклон известен на всём горизонте."""
    out, off = [], 0
    for L in D['lens']:
        v, gr = D['v'][off:off + L], D['grade'][off:off + L]
        for i in range(150, L - H - 1, 150):
            if v[i] > 2.0 and np.isfinite(gr[i:i + H]).all():
                out.append(off + i)
        off += L
    return np.array(out)


class Reservoir:
    def __init__(self, n=200, rho=0.8, leak=0.05, win=0.5, seed=0, n_in=8):
        rng = np.random.default_rng(seed)
        W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < 0.1)
        W *= rho / max(abs(np.linalg.eigvals(W)))
        self.W = torch.tensor(W, dtype=torch.float32, device=DEV)
        self.Win = torch.tensor(rng.uniform(-win, win, (n, n_in)), dtype=torch.float32, device=DEV)
        self.leak, self.n = leak, n

    def step(self, x, inp):
        return (1 - self.leak) * x + self.leak * torch.tanh(x @ self.W.T + inp @ self.Win.T)

    def states(self, D, memory):
        """Состояния по всем строкам (прогоны параллельно, с выравниванием). memory: вход — остаток и флаг."""
        lens = D['lens']
        Lmax, R = max(lens), len(lens)
        inp = torch.zeros((R, Lmax, 8), device=DEV)
        off = 0
        for i, L in enumerate(lens):
            inp[i, :L, :6] = torch.tensor(D['exo'][off:off + L], device=DEV)
            if memory:
                inp[i, :L, 6] = 2.0 * torch.tensor(D['r_in'][off:off + L], device=DEV)
                inp[i, :L, 7] = 1.0
            off += L
        x = torch.zeros((R, self.n), device=DEV)
        out = torch.zeros((R, Lmax, self.n), device=DEV)
        with torch.no_grad():
            for t in range(Lmax):
                x = self.step(x, inp[:, t])
                out[:, t] = x
        return torch.cat([out[i, :L] for i, L in enumerate(lens)])


# ============================================================================ модель прогноза
class Model(torch.nn.Module):
    def __init__(self, kind, nfeat, res: Reservoir | None, phys: dict | None):
        super().__init__()
        self.kind = kind
        self.res = res
        T = np.vstack([F.TABLE, F.REL_ROW[None, :]])
        self.T = torch.tensor(T, dtype=torch.float32, device=DEV)
        self.V = torch.tensor(F.V_GRID, dtype=torch.float32, device=DEV)
        # физика: множители тяги/торможения/выбега, поправка к коэф. уклона, log τ, смещение
        self.phys = torch.nn.Parameter(torch.zeros(6, device=DEV))
        if phys is not None:
            with torch.no_grad():
                self.phys.copy_(phys)
        self.phys.requires_grad_(kind == 'C1')
        nw = {'E0': 1, 'E1': 1, 'E2': 1, 'E3': 3, 'E3x': 3, 'E3m': 3, 'E3h': 3, 'NG': 1}.get(kind, 0)
        self.w = torch.nn.Parameter(torch.zeros(max(nw, 1), nfeat, device=DEV))
        self.w.requires_grad_(nw > 0)

    def table(self, row, v):
        idx = torch.clamp(torch.searchsorted(self.V, v.detach().contiguous()) - 1, 0, len(self.V) - 2)
        w = torch.clamp((v - self.V[idx]) / (self.V[idx + 1] - self.V[idx]), 0.0, 1.0)
        return self.T[row, idx] * (1 - w) + self.T[row, idx + 1] * w

    def feats(self, D, k, h, x, v, af, s):
        vs = v / 15.0
        dyn = torch.stack([vs, vs * vs, af, torch.ones_like(v)], 1)
        parts = []
        if self.kind in ('E0', 'E1', 'E2', 'E3', 'E3x', 'E3m', 'E3h'):
            parts.append(x)
        if self.kind == 'NG':
            lagm = (torch.tensor(NG_LAGS, device=DEV)[None, :] > h).float()      # после старта остаток неизвестен
            parts.append(D['rlag'][k] * lagm)
        parts += [D['exo'][k], dyn]
        if self.kind == 'E3h':          # время с начала провала: 0 — колёса есть, растёт без них
            hh = torch.full_like(v, float(h))
            parts.append(torch.stack([torch.exp(-hh / 20.0), torch.exp(-hh / 100.0), torch.clamp(hh / 300.0, max=1.0)], 1))
        if self.kind in ('E2', 'E3', 'E3x', 'E3h', 'NG'):
            parts.append(D['th'][s] - torch.tensor([1.0, 1.0, 1.0, 0.0], device=DEV))
        return torch.cat(parts, 1)

    def forward(self, D, s, H, xs=None, keep=(50, 100, 300, 600), wv=None, wp=None):
        v = D['v'][s].clone()
        af = D['a'][s].clone()
        pT, pB, p0, pk, plt, pc = self.phys
        tau = 0.3 * torch.exp(plt)
        al = 1.0 - torch.exp(-DT / tau)
        x = None
        if self.kind in ('E1', 'E2', 'E3', 'E3m', 'E3h'):
            x = D['xmem'][s].clone()
        th0 = D['th'][s]
        lv = torch.zeros((), device=DEV)
        lp = torch.zeros((), device=DEV)
        pos = torch.zeros_like(v)
        rec = {}
        for h in range(H):
            k = s + h
            tv = self.table(D['urow'][k], v)
            sg = D['sgn'][k]
            aT, aB, a0 = tv * (sg > 0), tv * (sg < 0), tv * (sg == 0)
            if self.kind == 'C2':
                ass = th0[:, 0] * aT + th0[:, 1] * aB + th0[:, 2] * a0 + th0[:, 3] - F.K_GRADE * D['grade'][k] - F.K_CURVE * D['curv'][k]
            else:
                ass = (1 + pT) * aT + (1 + pB) * aB + (1 + p0) * a0 + pc - F.K_GRADE * (1 + pk) * D['grade'][k] - F.K_CURVE * D['curv'][k]
            if self.kind in ('E0', 'E3x'):
                x = D['xexo'][k]
            elif x is not None:
                inp = torch.cat([D['exo'][k], torch.zeros((len(s), 2), device=DEV)], 1)
                with torch.no_grad():
                    x = self.res.step(x, inp)
            f = self.feats(D, k, h, x, v, af, s) if self.kind.startswith(('E', 'NG')) else None
            if self.kind.startswith('E3'):
                ass = ass + (f @ self.w[0]) * aT + (f @ self.w[1]) * aB
            af = af + al * (ass - af)
            af = torch.where((v <= 0) & (af < 0), torch.zeros_like(af), af)
            acc = af
            if self.kind in ('E0', 'E1', 'E2', 'NG'):
                acc = af + f @ self.w[0]
            elif self.kind.startswith('E3'):
                acc = af + f @ self.w[2]
            vn = torch.relu(v + DT * acc)
            pos = pos + 0.5 * DT * ((v + vn) - (D['v'][k] + D['v'][k + 1]))
            v = vn
            ev = ((v - D['v'][k + 1]) ** 2).mean()
            ep = (pos ** 2).mean()
            if wv is not None:
                ev, ep = ev * wv[h], ep * wp[h]
            lv = lv + ev
            lp = lp + ep
            if (h + 1) in keep:
                rec[h + 1] = (v - D['v'][k + 1]).detach(), pos.detach()
        return lv / H, lp / H, rec


def to_dev(D):
    return {k: (torch.tensor(v, device=DEV) if isinstance(v, np.ndarray) and k not in ('run',) else v)
            for k, v in D.items()}


NORM = False


def phys_profile(Dt, st, H):
    """Среднеквадратичная ошибка физики по шагам горизонта (для нормированных потерь)."""
    m = Model('P', 1, None, None)
    s = torch.tensor(st[::5], device=DEV)
    with torch.no_grad():
        v = Dt['v'][s].clone()
        af = Dt['a'][s].clone()
        pos = torch.zeros_like(v)
        ev, ep = [], []
        al = 1.0 - math.exp(-DT / 0.3)
        for h in range(H):
            k = s + h
            ass = m.table(Dt['urow'][k], v) - F.K_GRADE * Dt['grade'][k] - F.K_CURVE * Dt['curv'][k]
            af = af + al * (ass - af)
            af = torch.where((v <= 0) & (af < 0), torch.zeros_like(af), af)
            vn = torch.relu(v + DT * af)
            pos = pos + 0.5 * DT * ((v + vn) - (Dt['v'][k] + Dt['v'][k + 1]))
            v = vn
            ev.append(float(((v - Dt['v'][k + 1]) ** 2).mean()))
            ep.append(float((pos ** 2).mean()))
    ev, ep = np.maximum(ev, 1e-4), np.maximum(ep, 1e-4)
    return (torch.tensor(np.mean(ev) / ev, device=DEV, dtype=torch.float32),
            torch.tensor(np.mean(ep) / ep, device=DEV, dtype=torch.float32))


def train(kind, Dt, Di, st, si, nfeat, res, phys, epochs, beta=0.01, H=300, seed=0, log=print):
    torch.manual_seed(seed)
    m = Model(kind, nfeat, res, phys)
    params = [p for p in m.parameters() if p.requires_grad]
    if not params:
        return m
    lr = 2e-3 if kind == 'C1' else LR
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0 if kind == 'C1' else 1e-3)
    si_t = torch.tensor(si, device=DEV)
    wv = wp = None
    if NORM:
        wv, wp = phys_profile(Dt, st, H)
    with torch.no_grad():
        lv, lp, _ = m(Di, si_t, H, wv=wv, wp=wp)
    best = (float(lv + beta * lp), {k: v.detach().clone() for k, v in m.state_dict().items()})
    B = 4096
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 1000 + ep).permutation(st)
        for i in range(0, len(perm), B):
            lv, lp, _ = m(Dt, torch.tensor(perm[i:i + B], device=DEV), H, wv=wv, wp=wp)
            opt.zero_grad()
            (lv + beta * lp).backward()
            opt.step()
        with torch.no_grad():
            lv, lp, _ = m(Di, si_t, H, wv=wv, wp=wp)
        sc = float(lv + beta * lp)
        if sc < best[0]:
            best = (sc, {k: v.detach().clone() for k, v in m.state_dict().items()})
        if ep % 10 == 9 or ep == epochs - 1:
            log(f'    {kind} эпоха {ep + 1}: внутр. вал. {sc:.4f} (лучшее {best[0]:.4f})')
    m.load_state_dict(best[1])
    return m


def evaluate(models, Dv, sv, H=600):
    """Ансамбль — среднее ошибок недопустимо (замкнутый контур), поэтому для ансамбля — отдельный прогон
    со средним ускорением (см. evaluate_ens); здесь — один вариант."""
    with torch.no_grad():
        _, _, rec = models(Dv, torch.tensor(sv, device=DEV), H)
    return {h * DT: (float(torch.sqrt((e ** 2).mean())), float(torch.sqrt((p ** 2).mean()))) for h, (e, p) in rec.items()}


def evaluate_ens(ms, Dv, sv, H=600):
    """Ансамбль: на каждом шаге ускорение — среднее по моделям при общей скорости (как будет в оценщике)."""
    s = torch.tensor(sv, device=DEV)
    m0 = ms[0]
    with torch.no_grad():
        v = Dv['v'][s].clone()
        afs = [Dv['a'][s].clone() for _ in ms]
        xs = [Dv[f'xmem{j}'][s].clone() if m.kind in ('E1', 'E2', 'E3', 'E3h') else None for j, m in enumerate(ms)]
        pos = torch.zeros_like(v)
        out = {}
        for h in range(H):
            k = s + h
            tv = m0.table(Dv['urow'][k], v)
            sg = Dv['sgn'][k]
            aT, aB, a0 = tv * (sg > 0), tv * (sg < 0), tv * (sg == 0)
            accs = []
            for j, m in enumerate(ms):
                pT, pB, p0, pk, plt, pc = m.phys
                al = 1.0 - torch.exp(-DT / (0.3 * torch.exp(plt)))
                ass = (1 + pT) * aT + (1 + pB) * aB + (1 + p0) * a0 + pc - F.K_GRADE * (1 + pk) * Dv['grade'][k] - F.K_CURVE * Dv['curv'][k]
                if xs[j] is not None:
                    inp = torch.cat([Dv['exo'][k], torch.zeros((len(s), 2), device=DEV)], 1)
                    xs[j] = m.res.step(xs[j], inp)
                Dm = dict(Dv, xmem=Dv[f'xmem{j}'])
                f = m.feats(Dm, k, h, xs[j], v, afs[j], s)
                if m.kind.startswith('E3'):
                    ass = ass + (f @ m.w[0]) * aT + (f @ m.w[1]) * aB
                afs[j] = afs[j] + al * (ass - afs[j])
                afs[j] = torch.where((v <= 0) & (afs[j] < 0), torch.zeros_like(afs[j]), afs[j])
                accs.append(afs[j] + f @ m.w[2 if m.kind.startswith('E3') else 0])
            acc = torch.stack(accs).mean(0)
            vn = torch.relu(v + DT * acc)
            pos = pos + 0.5 * DT * ((v + vn) - (Dv['v'][k] + Dv['v'][k + 1]))
            v = vn
            if (h + 1) in (50, 100, 300, 600):
                e = v - Dv['v'][k + 1]
                out[(h + 1) * DT] = (float(torch.sqrt((e ** 2).mean())), float(torch.sqrt((pos ** 2).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='P,C1,C2,E0,E1,E2,E3,NG,ENS')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--final', action='store_true', help='обучить лучший вариант на train+val (для сдачи)')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--norm', action='store_true', help='нормированные по шагам потери (ранние секунды не тонут)')
    ap.add_argument('--export', action='store_true', help='выгрузить веса E3 в пакет (config/esn.json)')
    a = ap.parse_args()
    global LR, NORM
    LR, NORM = a.lr, a.norm
    t0 = time.time()
    OUT = F.OUT
    logf = open(OUT / f'v2{a.tag}.log', 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'=== esn_v2 {time.strftime("%H:%M:%S")} устройство {DEV}')
    sp = splits()
    with ProcessPoolExecutor(16) as ex:
        tr = [g for g in ex.map(F.run_grid, sp['train']) if g is not None]
        va = [g for g in ex.map(F.run_grid, sp['val']) if g is not None]
        tr = list(ex.map(prep, tr))
        va = list(ex.map(prep, va))
    ntr = int(0.8 * len(tr))
    Dt, Di, Dv = build(tr[:ntr]), build(tr[ntr:]), build(va)
    log(f'данные {time.time() - t0:.0f} с: train {ntr}, внутр. вал. {len(tr) - ntr}, val {len(va)} прогонов')
    st, si = windows(Dt, 300, 10), windows(Di, 300, 20)
    sv = val_windows(Dv, 600)
    Dt, Di, Dv = to_dev(Dt), to_dev(Di), to_dev(Dv)
    log(f'окон: train {len(st)}, внутр. вал. {len(si)}, val {len(sv)} (горизонт 60 с)')
    res = Reservoir(n=a.n)
    for D in (Dt, Di, Dv):
        D['xexo'] = res.states(D, memory=False)
        D['xmem'] = res.states(D, memory=True)
    nf = {'E0': a.n + 10, 'E1': a.n + 10, 'E2': a.n + 14, 'E3': a.n + 14, 'E3x': a.n + 14, 'E3m': a.n + 10, 'E3h': a.n + 17,
          'NG': len(NG_LAGS) + 14}
    results, models = {}, {}
    variants = a.variants.split(',')
    if 'P' in variants:
        results['P'] = evaluate(Model('P', 1, None, None), Dv, sv)
    phys_c1 = None
    if any(v in variants for v in ('C1', 'E0', 'E1', 'E2', 'E3', 'NG', 'ENS')):
        m = train('C1', Dt, Di, st, si, 1, None, None, a.epochs, log=log)
        phys_c1 = m.phys.detach().clone()
        p = phys_c1.cpu().numpy()
        log(f'  C1: тяга ×{1 + p[0]:.3f}, торм. ×{1 + p[1]:.3f}, выбег ×{1 + p[2]:.3f}, уклон ×{1 + p[3]:.3f}, '
            f'τ={0.3 * math.exp(p[4]):.2f} с, смещение {p[5]:+.3f}')
        results['C1'] = evaluate(m, Dv, sv)
    if 'C2' in variants:
        results['C2'] = evaluate(Model('C2', 1, None, None), Dv, sv)
    for kind in ('E0', 'E1', 'E2', 'E3', 'E3x', 'E3m', 'E3h', 'NG'):
        if kind in variants:
            m = train(kind, Dt, Di, st, si, nf[kind], res, phys_c1, a.epochs, log=log)
            models[kind] = m
            results[kind] = evaluate(m, Dv, sv)
            log(f'  {kind}: ' + ', '.join(f'{h:g} с: v {e:.3f} / путь {p:.2f}' for h, (e, p) in results[kind].items()))
    if 'ENS' in variants:
        cand = [k for k in ('E1', 'E2', 'E3', 'E3h') if k in results]
        best = min(cand, key=lambda k: results[k][30.0][0])
        ms = [models[best]]
        for seed in range(1, 5):
            r = Reservoir(n=a.n, seed=seed)
            for D in (Dt, Di, Dv):
                D[f'xmem{seed}'] = r.states(D, memory=True)
            for D in (Dt, Di):
                D['xmem_bak'] = D['xmem']
                D['xmem'] = D[f'xmem{seed}']
            ms.append(train(best, Dt, Di, st, si, nf[best], r, phys_c1, a.epochs, seed=seed, log=log))
            for D in (Dt, Di):
                D['xmem'] = D['xmem_bak']
        Dv['xmem0'] = Dv['xmem']
        results[f'ENS({best}×5)'] = evaluate_ens(ms, Dv, sv)
        torch.save({'models': [m.state_dict() for m in ms], 'kind': best, 'W': [mm.res.W.cpu() for mm in ms],
                    'Win': [mm.res.Win.cpu() for mm in ms], 'leak': res.leak}, OUT / 'v2_ens.pt')
    rows = [{'вариант': k, 'горизонт, с': h, 'v RMSE, м/с': round(e, 3), 'путь RMSE, м': round(p, 2)}
            for k, r in results.items() for h, (e, p) in r.items()]
    T = pd.DataFrame(rows)
    piv = T.pivot_table(index='вариант', columns='горизонт, с', values=['v RMSE, м/с', 'путь RMSE, м'], sort=False)
    log(piv.to_string())
    T.to_csv(OUT / f'v2_results{a.tag}.csv', index=False)
    for k, m in models.items():
        torch.save(m.state_dict(), OUT / f'v2{a.tag}_{k}.pt')
    if a.export and 'E3' in models:
        m = models['E3']
        pkg = F.MODEL_JSON.parent / 'esn.json'
        pkg.write_text(json.dumps({
            'kind': 'E3', 'grid_dt': DT, 'lag_r': LAG_R, 'rls_lambda': 0.9995, 'base_tau': F.TAU,
            'phys': m.phys.detach().cpu().numpy().round(6).tolist(),
            'members': [{'W': np.round(res.W.cpu().numpy(), 6).tolist(), 'Win': np.round(res.Win.cpu().numpy(), 6).tolist(),
                         'w': np.round(m.w.detach().cpu().numpy(), 7).tolist(), 'leak': res.leak}],
            'note': 'ESN E3: резервуар 200, память остатка + контекст (онлайн-МНК) + множители тяги/торможения; '
                    'обучение многошаговое (30 с, нормированные потери) на train',
            'val': {str(h): [round(e, 4), round(pp, 3)] for h, (e, pp) in results['E3'].items()}},
            ensure_ascii=False), encoding='utf-8')
        log(f'веса выгружены: {pkg}')
    torch.save({'phys_c1': phys_c1, 'W': res.W.cpu(), 'Win': res.Win.cpu(), 'leak': res.leak}, OUT / 'v2_common.pt')
    log(f'всего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()

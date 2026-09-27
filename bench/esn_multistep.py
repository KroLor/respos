"""ESN, обученная на многошаговый прогноз (замкнутый контур, «без колёс»), а не на один шаг.

Резервуар получает только внешние воздействия (история позиций контроллера и уклона), поэтому его состояния
не зависят от прогнозируемой скорости и вычисляются один раз. Выходной слой обучается градиентным спуском
через развёртку модели на H шагов: v ← v + dt·(a_физ(u, v, i) + ESN(x, v, a_физ)), потери — ошибка скорости и
пути относительно измеренной на всём горизонте. Старт весов — ноль (= чистая физика).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/esn_multistep.py [--leak 0.05] [--H 300]
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import esn_feasibility as F
from common import splits

DT = F.DT


def exo_states(args):
    esn, g = args
    Z = F.run_states(esn, g)
    return Z[:, : esn.n + 6]          # резервуар + внешние входы (не зависят от скорости)


def build(grids, esn, ex):
    Zs = list(ex.map(exo_states, [(esn, g) for g in grids]))
    rows = np.vstack([np.c_[np.full(len(g), i)] for i, g in enumerate(grids)])[:, 0]
    cat = lambda c: np.concatenate([g[c].to_numpy() for g in grids])  # noqa: E731
    u = cat('u')
    urow = np.where(u == F.RELEASE, 31, np.clip(u, -15, 15) + 15).astype(np.int64)
    return {'Z': np.vstack(Zs).astype(np.float32), 'urow': urow, 'grade': cat('grade'), 'v': cat('v'),
            'a': cat('a'), 'run': rows}


def windows(D, H, stride):
    run, gr = D['run'], np.isfinite(D['grade'])
    n = len(run)
    ok_run = np.r_[run[H:] == run[:-H], np.zeros(H, bool)]          # окно внутри одного прогона
    cg = np.r_[0, np.cumsum(~gr)]
    ok_gr = np.zeros(n, bool)
    ok_gr[: n - H] = (cg[H:n] - cg[: n - H]) == 0                  # уклон известен на всём окне
    s = np.nonzero(ok_run & ok_gr)[0]
    return s[::stride]


class Rollout(torch.nn.Module):
    def __init__(self, nz, dev):
        super().__init__()
        self.w_z = torch.nn.Parameter(torch.zeros(nz, device=dev))
        self.w_d = torch.nn.Parameter(torch.zeros(4, device=dev))        # v/15, (v/15)², a_физ, 1
        T = np.vstack([F.TABLE, F.REL_ROW[None, :]])
        self.T = torch.tensor(T, dtype=torch.float32, device=dev)
        self.V = torch.tensor(F.V_GRID, dtype=torch.float32, device=dev)
        self.al = 1.0 - math.exp(-DT / F.TAU)

    def table(self, row, v):
        idx = torch.clamp(torch.searchsorted(self.V, v.detach().contiguous()) - 1, 0, len(self.V) - 2)
        v0, v1 = self.V[idx], self.V[idx + 1]
        w = torch.clamp((v - v0) / (v1 - v0), 0.0, 1.0)
        return self.T[row, idx] * (1 - w) + self.T[row, idx + 1] * w

    def forward(self, D, s, H, use_esn=True):
        v = D['v'][s].clone()
        af = D['a'][s].clone()
        loss_v = torch.zeros((), device=v.device)
        pos = torch.zeros_like(v)
        loss_p = torch.zeros((), device=v.device)
        for h in range(H):
            k = s + h
            ass = self.table(D['urow'][k], v) - F.K_GRADE * D['grade'][k]
            af = af + self.al * (ass - af)
            af = torch.where((v <= 0) & (af < 0), torch.zeros_like(af), af)
            acc = af
            if use_esn:
                vs = v / 15.0
                corr = D['Z'][k] @ self.w_z + self.w_d[0] * vs + self.w_d[1] * vs * vs + self.w_d[2] * af + self.w_d[3]
                acc = af + corr
            vn = torch.relu(v + DT * acc)
            pos = pos + 0.5 * DT * ((v + vn) - (D['v'][k] + D['v'][k + 1]))
            v = vn
            loss_v = loss_v + ((v - D['v'][k + 1]) ** 2).mean()
            loss_p = loss_p + (pos ** 2).mean()
        return loss_v / H, loss_p / H, v - D['v'][s + H], pos


def to_dev(D, dev):
    return {k: torch.tensor(v, device=dev) if k != 'run' else v for k, v in D.items()
            if k in ('Z', 'urow', 'grade', 'v', 'a')} | {'run': D['run']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leak', type=float, default=0.05)
    ap.add_argument('--rho', type=float, default=0.8)
    ap.add_argument('--H', type=int, default=300)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--beta', type=float, default=0.01, help='вес ошибки пути')
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('устройство:', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')
    t0 = time.time()
    sp = splits()
    with ProcessPoolExecutor(16) as ex:
        tr = [g for g in ex.map(F.run_grid, sp['train']) if g is not None]
        va = [g for g in ex.map(F.run_grid, sp['val']) if g is not None]
        for g in tr + va:
            g['a_phys'] = F.physics(g.u.to_numpy(), g.v.to_numpy(), g.grade.to_numpy())
            g['res'] = g.a - g.a_phys
        esn = F.ESN(n=a.n, rho=a.rho, leak=a.leak)
        ntr = int(0.8 * len(tr))
        Dtr = build(tr[:ntr], esn, ex)
        Div = build(tr[ntr:], esn, ex)
    print(f'данные: {time.time() - t0:.0f} с; train-строк {len(Dtr["v"])}, внутр. вал. {len(Div["v"])}')
    H = a.H
    s_tr, s_iv = windows(Dtr, H, 10), windows(Div, H, 20)
    Tt, Ti = to_dev(Dtr, dev), to_dev(Div, dev)
    s_iv_t = torch.tensor(s_iv, device=dev)
    model = Rollout(Dtr['Z'].shape[1], dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    with torch.no_grad():
        lv0, lp0, _, _ = model(Ti, s_iv_t, H, use_esn=False)
    print(f'окон: train {len(s_tr)}, внутр. вал. {len(s_iv)}; физика на внутр. вал.: v² {lv0:.4f}, путь² {lp0:.2f}')
    best = (float(lv0 + a.beta * lp0), {k: v.detach().clone() for k, v in model.state_dict().items()})  # чистая физика
    B = 4096
    for ep in range(a.epochs):
        te = time.time()
        perm = np.random.default_rng(ep).permutation(s_tr)
        for i in range(0, len(perm), B):
            sb = torch.tensor(perm[i:i + B], device=dev)
            lv, lp, _, _ = model(Tt, sb, H)
            loss = lv + a.beta * lp
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            lv, lp, _, _ = model(Ti, s_iv_t, H)
        score = float(lv + a.beta * lp)
        if score < best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f'  эпоха {ep:2d}: внутр. вал. v² {lv:.4f} (физика {lv0:.4f}), путь² {lp:.2f} (физика {lp0:.2f}); '
              f'{time.time() - te:.1f} с')
    model.load_state_dict(best[1])
    w = np.r_[model.w_z.detach().cpu().numpy(), model.w_d.detach().cpu().numpy()]
    esn.w = w.astype(np.float64)
    # честная оценка на val (тот же код, что в шаге 0: замкнутый контур, numpy)
    with ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(F.horizon_eval, [(esn, g) for g in va]) for r in part]
    Hd = pd.DataFrame(rows)
    Hd = Hd[Hd['var'].isin(['P', 'PB', 'E'])]
    tab = Hd.groupby(['H', 'var']).agg(v_rmse=('dv', lambda x: np.sqrt(np.mean(x ** 2))),
                                       pos_rmse=('dpos', lambda x: np.sqrt(np.mean(x ** 2))),
                                       n=('dv', 'size')).round(3)
    print(tab.to_string())
    F.OUT.mkdir(parents=True, exist_ok=True)
    tag = f'ms_n{a.n}_leak{a.leak}_H{H}_b{a.beta}'
    tab.to_csv(F.OUT / f'{tag}.csv')
    np.savez(F.OUT / f'{tag}.npz', W=esn.W, Win=esn.Win, w=esn.w, leak=esn.leak, rho=a.rho)
    print('всего', f'{time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()

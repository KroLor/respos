"""Инъекция сбоев во входные топики. Каждый сценарий возвращает изменённый прогон и окна сбоев (stamp, с).

Все случайные величины — от seed, зависящего от прогона и сценария (воспроизводимо).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

SCENARIOS = ['clean', 'dropout10', 'dropout30', 'front_lost', 'slip', 'slip_one', 'slide', 'outliers',
             'frozen', 'noise', 'msg_loss', 'cmd_loss', 'mix']


def _rng(bag: str, name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f'{bag}/{name}'.encode()))


def _grid(run):
    """Сетка 10 Гц: t (header, с), v (м/с, среднее тележек), u — для выбора моментов сбоев."""
    w = pd.concat([run['front'], run['rear']]).sort_values('t_hdr')
    t = np.arange(w.t_hdr.iloc[0], w.t_hdr.iloc[-1], 1e8) / 1e9
    v = np.interp(t, w.t_hdr / 1e9, w.velocity / 3.6)
    c = run['cmd'].sort_values('t_hdr')
    i = np.clip(np.searchsorted(c.t_hdr / 1e9, t, side='right') - 1, 0, len(c) - 1)
    u = c.position.to_numpy()[i] if len(c) else np.zeros(len(t))
    return t, v, u


def _pick(rng, t, cond, n, dur, gap=30.0):
    """n непересекающихся окон длительностью dur, начинающихся там, где cond истинно."""
    cand = np.nonzero(cond)[0]
    wins = []
    for i in rng.permutation(cand):
        t0 = t[i]
        if t0 + dur > t[-1] or t0 < t[0] + 20:
            continue
        if all(abs(t0 - a) > dur + gap for a, _ in wins):
            wins.append((t0, t0 + dur))
        if len(wins) == n:
            break
    return sorted(wins)


def _mask(df, wins):
    t = df.t_hdr.to_numpy() / 1e9
    m = np.zeros(len(df), bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


def _scale_in(df, wins, profile):
    """Умножение показаний на (1 + profile(доля окна)) внутри окон."""
    df = df.copy()
    t = df.t_hdr.to_numpy() / 1e9
    v = df.velocity.to_numpy().astype(float)
    for a, b in wins:
        m = (t >= a) & (t <= b)
        v[m] = v[m] * (1.0 + profile((t[m] - a) / (b - a)))
    df['velocity'] = v
    return df


def apply(run: dict, bag: str, name: str) -> tuple[dict, list]:
    run = {k: v.copy() for k, v in run.items()}
    if name == 'clean':
        return run, []
    rng = _rng(bag, name)
    t, v, u = _grid(run)
    moving = v > 3.0
    wins: list = []

    def drop(keys, wins_):
        for k in keys:
            run[k] = run[k][~_mask(run[k], wins_)]

    if name == 'dropout10':
        wins = _pick(rng, t, moving, 4, 10.0)
        drop(['front', 'rear'], wins)
    elif name == 'dropout30':
        wins = _pick(rng, t, moving, 2, 30.0, gap=60)
        drop(['front', 'rear'], wins)
    elif name == 'front_lost':
        wins = [(t[0] + 60.0, t[-1])]
        drop(['front'], wins)
    elif name in ('slip', 'slip_one'):
        # боксование при тяге: показания растут до +20…40 % и возвращаются за 3 с
        wins = _pick(rng, t, (u > 2) & (v > 1.5) & (v < 10), 6, 3.0)
        amp = rng.uniform(0.2, 0.4)
        prof = lambda x: amp * np.sin(np.pi * x)  # noqa: E731
        run['front'] = _scale_in(run['front'], wins, prof)
        if name == 'slip':
            run['rear'] = _scale_in(run['rear'], wins, prof)
    elif name == 'slide':
        # юз при торможении: колёса замедляются до −30…−60 % за 2.5 с
        wins = _pick(rng, t, (u < -2) & (v > 4), 6, 2.5)
        amp = rng.uniform(0.3, 0.6)
        prof = lambda x: -amp * np.sin(np.pi * x)  # noqa: E731
        run['front'] = _scale_in(run['front'], wins, prof)
        run['rear'] = _scale_in(run['rear'], wins, prof)
    elif name == 'outliers':
        for k in ('front', 'rear'):
            d = run[k]
            m = rng.random(len(d)) < 0.01
            vv = d.velocity.to_numpy().astype(float)
            vv[m] = rng.uniform(0, 80, m.sum())
            d['velocity'] = vv
            run[k] = d
        wins = []
    elif name == 'frozen':
        wins = _pick(rng, t, moving, 3, 8.0)
        for k in ('front', 'rear'):
            d = run[k]
            tt = d.t_hdr.to_numpy() / 1e9
            vv = d.velocity.to_numpy().astype(float)
            for a, b in wins:
                m = (tt >= a) & (tt <= b)
                if m.any():
                    i0 = np.argmax(m)
                    vv[m] = vv[max(i0 - 1, 0)]
            d['velocity'] = vv
            run[k] = d
    elif name == 'noise':
        for k in ('front', 'rear'):
            d = run[k]
            d['velocity'] = np.maximum(d.velocity + rng.normal(0, 1.5, len(d)), 0.0)
            run[k] = d
    elif name == 'msg_loss':
        for k in ('front', 'rear', 'cmd'):
            run[k] = run[k][rng.random(len(run[k])) > 0.3]
    elif name == 'cmd_loss':
        wins = _pick(rng, t, moving, 3, 20.0)
        drop(['cmd'], wins)
    elif name == 'mix':
        r1, w1 = apply(run, bag, 'slip')
        r2, w2 = apply(r1, bag, 'slide')
        r3, _ = apply(r2, bag, 'outliers')
        r4, w4 = apply(r3, bag, 'dropout10')
        return r4, sorted(w1 + w2 + w4)
    else:
        raise ValueError(name)
    return run, wins


SLIP_SCENARIOS = {'slip', 'slip_one', 'slide'}

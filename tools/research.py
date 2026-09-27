"""Исследование выгруженных данных (extracted/): единицы скорости, разброс тележек,
ковариации GNSS, курс и качество приёма. Печатает сводные таблицы, пишет CSV в reports/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / 'extracted'
REP = ROOT / 'reports'
REP.mkdir(exist_ok=True)
R_EARTH = 6378137.0
NS = 1_000_000_000
pd.set_option('display.width', 220, 'display.max_columns', 40)


def load(bag, name):
    return pd.read_parquet(EX / bag / f'{name}.parquet').sort_values('t_hdr').reset_index(drop=True)


def enu(df, lat0, lon0):
    e = np.radians(df.lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    n = np.radians(df.lat - lat0) * R_EARTH
    return e.to_numpy(), n.to_numpy()


def wrap(a):
    return (a + 180) % 360 - 180


def asof(left, right, cols, tol_ms=60, lag_ns=0):
    r = right[['t_hdr'] + cols].copy()
    r['t_hdr'] = r.t_hdr + lag_ns
    return pd.merge_asof(left, r, on='t_hdr', direction='nearest',
                         tolerance=tol_ms * 1_000_000)


summary = pd.read_csv(EX / 'summary.csv')
bags_all = summary.bag.tolist()
bags_gnss = summary[summary.n_master_fix > 0].bag.tolist()

# ---------------------------------------------------------------- 3. ковариации и статус
cov_rows, st_rows = [], []
for b in bags_gnss:
    for rx in ('master', 'rover'):
        f = load(b, f'{rx}_fix')
        if f.empty:
            continue
        nz = (f[['cov_e', 'cov_n', 'cov_u']] != 0).any(axis=1)
        cov_rows.append({'bag': b, 'rx': rx, 'n': len(f), 'n_cov_nonzero': int(nz.sum())})
        for s, c in f.status.value_counts().items():
            st_rows.append({'bag': b, 'rx': rx, 'status': s, 'n': c})
        # пропуски внутри записи
        gaps = f.t_hdr.diff() / NS
        st_rows.append({'bag': b, 'rx': rx, 'status': 'gaps>0.5s', 'n': int((gaps > 0.5).sum())})
cov = pd.DataFrame(cov_rows)
st = pd.DataFrame(st_rows)
cov.to_csv(REP / 'gnss_cov.csv', index=False)
st.to_csv(REP / 'gnss_status.csv', index=False)

# ---------------------------------------------------------------- основной проход по прогонам
unit_rows, bogie_rows, abs_rows, head_rows = [], [], [], []
lag_grid = np.arange(-600, 601, 20)  # мс, сдвиг колёс относительно GNSS
lag_acc = np.zeros(len(lag_grid))
lag_cnt = np.zeros(len(lag_grid))
pooled = []

for b in bags_all:
    fr, re, cmd = load(b, 'front'), load(b, 'rear'), load(b, 'cmd')

    # --- 2a. тележки относительно друг друга (все прогоны, GNSS не нужен)
    fb = asof(fr.rename(columns={'velocity': 'vf'}), re.rename(columns={'velocity': 'vr'}), ['vr'], tol_ms=60)
    fb = asof(fb, cmd, ['position'], tol_ms=60).dropna()
    same_stamp = (fr.t_hdr.isin(re.t_hdr)).mean()
    mv = fb[(fb.vf.abs() > 3.6) | (fb.vr.abs() > 3.6)]  # > 1 м/с, если км/ч
    d = mv.vf - mv.vr
    bogie_rows.append({'bag': b, 'n_moving': len(mv), 'same_stamp_frac': same_stamp,
                       'neg_frac': float((fb[['vf', 'vr']] < -0.1).any(axis=1).mean()),
                       'diff_mean': d.mean(), 'diff_std': d.std(),
                       'diff_p99_abs': d.abs().quantile(.99) if len(d) else np.nan,
                       'diff_max_abs': d.abs().max() if len(d) else np.nan,
                       'rel_diff_med_%': (d / mv[['vf', 'vr']].mean(axis=1)).median() * 100 if len(d) else np.nan})

    if b not in bags_gnss:
        continue
    mfix, rfix, mvel = load(b, 'master_fix'), load(b, 'rover_fix'), load(b, 'master_vel')
    lat0, lon0 = mfix.lat.iloc[0], mfix.lon.iloc[0]
    mfix['e'], mfix['n'] = enu(mfix, lat0, lon0)
    rfix['e'], rfix['n'] = enu(rfix, lat0, lon0)
    mvel['g'] = np.hypot(mvel.vx, mvel.vy)

    # скорость из позиций (центральная разность ±0.5 с)
    t = mfix.t_hdr.to_numpy() / NS
    k = 5
    if len(mfix) > 2 * k:
        de = mfix.e.shift(-k) - mfix.e.shift(k)
        dn = mfix.n.shift(-k) - mfix.n.shift(k)
        dt = pd.Series(t).shift(-k) - pd.Series(t).shift(k)
        mfix['pe'], mfix['pn'] = de / dt, dn / dt
        mfix['pg'] = np.hypot(mfix.pe, mfix.pn)

    # --- 1. единицы: колёса vs GNSS vel и vs скорость из позиций
    m = asof(fb, mvel, ['g', 'vx', 'vy'], tol_ms=60)
    m = asof(m, mfix, ['pg', 'pe', 'pn'], tol_ms=60).dropna()
    mm = m[m.g > 2]
    # проверка системы координат twist: vx≈East, vy≈North ?
    cxe = np.corrcoef(m.vx, m.pe)[0, 1] if len(m) > 10 else np.nan
    cyn = np.corrcoef(m.vy, m.pn)[0, 1] if len(m) > 10 else np.nan
    unit_rows.append({'bag': b, 'n': len(mm),
                      'front/gnss_vel': (mm.vf / mm.g).median(), 'rear/gnss_vel': (mm.vr / mm.g).median(),
                      'front/gnss_pos': (mm.vf / mm.pg).median(), 'gnss_vel/gnss_pos': (mm.g / mm.pg).median(),
                      'corr(vx,dE/dt)': cxe, 'corr(vy,dN/dt)': cyn,
                      'max_front': fr.velocity.max(), 'max_gnss': mvel.g.max()})

    # --- 2b. абсолютная ошибка (после /3.6) относительно GNSS, с поиском задержки
    w = fb.copy()
    w['vf'] /= 3.6
    w['vr'] /= 3.6
    for i, lag in enumerate(lag_grid):
        x = asof(w, mvel, ['g'], tol_ms=60, lag_ns=int(lag * 1e6)).dropna()
        if len(x):
            lag_acc[i] += ((x.vf - x.g) ** 2).sum()
            lag_cnt[i] += len(x)
    x = asof(w, mvel, ['g'], tol_ms=60).dropna()
    x['bag'] = b
    pooled.append(x[['bag', 'vf', 'vr', 'g', 'position']])

    # --- 4. курс: из twist, из разности позиций, из базы master→rover
    h = asof(mfix[['t_hdr', 'e', 'n', 'pe', 'pn', 'pg']], rfix.rename(columns={'e': 're', 'n': 'rn'}),
             ['re', 'rn'], tol_ms=20)
    h = asof(h, mvel, ['vx', 'vy', 'g'], tol_ms=60).dropna()
    h['base_len'] = np.hypot(h.re - h.e, h.rn - h.n)
    h['hdg_base'] = np.degrees(np.arctan2(h.re - h.e, h.rn - h.n))   # азимут от севера, по часовой
    h['hdg_pos'] = np.degrees(np.arctan2(h.pe, h.pn))
    h['hdg_vel'] = np.degrees(np.arctan2(h.vx, h.vy))
    hm = h[h.g > 2]
    if len(hm) < 20:
        continue
    d_vp = wrap(hm.hdg_vel - hm.hdg_pos)
    d_bp = wrap(hm.hdg_base - hm.hdg_pos)
    off = np.degrees(np.angle(np.exp(1j * np.radians(d_bp)).mean()))  # круговое среднее
    d_bp_c = wrap(d_bp - off)
    head_rows.append({'bag': b, 'n': len(hm),
                      'base_len_med': h.base_len.median(), 'base_len_std': h.base_len.std(),
                      'vel_vs_pos_med': d_vp.median(), 'vel_vs_pos_p95': d_vp.abs().quantile(.95),
                      'base_offset_deg': off, 'base_vs_pos_std': d_bp_c.std(),
                      'base_vs_pos_p95': d_bp_c.abs().quantile(.95),
                      'frac_reversed(>90)': float((d_bp_c.abs() > 90).mean())})

units = pd.DataFrame(unit_rows)
bog = pd.DataFrame(bogie_rows)
head = pd.DataFrame(head_rows)
P = pd.concat(pooled, ignore_index=True)
units.to_csv(REP / 'units.csv', index=False)
bog.to_csv(REP / 'bogie_diff.csv', index=False)
head.to_csv(REP / 'heading.csv', index=False)

# ---------------------------------------------------------------- вывод
q = lambda s: s.describe(percentiles=[.05, .5, .95])[['count', 'mean', 'std', 'min', '5%', '50%', '95%', 'max']]

print('\n######## 1. ЕДИНИЦЫ')
print(units.drop(columns='bag').apply(q).T.round(3).to_string())

print('\n######## 2a. ТЕЛЕЖКИ ДРУГ ОТНОСИТЕЛЬНО ДРУГА (исходные единицы, в движении)')
print(bog.drop(columns='bag').apply(q).T.round(3).to_string())
print('Худшие прогоны по p99 |front-rear|:')
print(bog.sort_values('diff_p99_abs', ascending=False).head(8).round(3).to_string(index=False))

print('\n######## 2b. АБСОЛЮТНО ОТНОСИТЕЛЬНО GNSS (колёса /3.6, м/с)')
rm = np.sqrt(lag_acc / np.maximum(lag_cnt, 1))
best = lag_grid[np.argmin(rm)]
print(f'RMSE front vs GNSS по сдвигу: 0 мс → {rm[lag_grid == 0][0]:.3f}; лучший сдвиг {best} мс → {rm.min():.3f}')


def err_stats(df, label):
    rows = []
    for name, col in (('front', 'vf'), ('rear', 'vr'), ('min(f,r)', None), ('mean(f,r)', None)):
        v = df[['vf', 'vr']].min(axis=1) if name == 'min(f,r)' else \
            df[['vf', 'vr']].mean(axis=1) if name == 'mean(f,r)' else df[col]
        e = v - df.g
        rows.append({'режим': label, 'сигнал': name, 'n': len(e), 'bias': e.mean(), 'std': e.std(),
                     'RMSE': np.sqrt((e ** 2).mean()), 'MAE': e.abs().mean(),
                     'p99|e|': e.abs().quantile(.99), 'max|e|': e.abs().max(),
                     'scale(v/g)': (v[df.g > 2] / df.g[df.g > 2]).median()})
    return rows


regimes = {'все': P, 'стоянка (g<0.3)': P[P.g < 0.3], 'тяга (pos>0)': P[P.position > 0],
           'выбег (pos=0, g>0.3)': P[(P.position == 0) & (P.g > 0.3)], 'торможение (pos<0)': P[P.position < 0]}
E = pd.DataFrame([r for k, v in regimes.items() for r in err_stats(v, k)])
E.to_csv(REP / 'abs_error.csv', index=False)
print(E.round(3).to_string(index=False))

pb = P.assign(ef=P.vf - P.g).groupby('bag').ef.agg(bias='mean', rmse=lambda s: np.sqrt((s ** 2).mean()))
pb['scale'] = P[P.g > 2].assign(r=lambda d: d.vf / d.g).groupby('bag').r.median()
pb.to_csv(REP / 'abs_error_by_bag.csv')
print('По прогонам (front):')
print(pb.apply(q).T.round(3).to_string())

print('\n######## 3. КОВАРИАЦИИ')
print(f'Фиксов всего: {cov.n.sum()}, с ненулевой ковариацией: {cov.n_cov_nonzero.sum()}')

print('\n######## 4. СТАТУС И КУРС')
sv = st[st.status != 'gaps>0.5s'].groupby(['rx', 'status']).n.sum().unstack(fill_value=0)
print('Статус (число фиксов):'); print(sv.to_string())
print('Доля status=2 по прогонам (master):')
ms = st[(st.rx == 'master') & (st.status != 'gaps>0.5s')].pivot_table(index='bag', columns='status', values='n', fill_value=0)
print((ms.get(2, 0) / ms.sum(axis=1)).describe().round(4).to_string())
g = st[st.status == 'gaps>0.5s'].groupby('rx').n.agg(['sum', lambda s: (s > 0).sum()])
print('Пропуски фиксов >0.5 с (всего, прогонов с пропусками):'); print(g.to_string())
print('Курс (град): twist vs позиции; база master→rover vs позиции')
print(head.drop(columns='bag').apply(q).T.round(3).to_string())

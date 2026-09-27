"""Офлайн-калибровка параметров модели по обучающей выборке (bench/splits.json → train).

1. Масштаб колёс: v_GNSS / v_колёс (скорость).
2. Масштаб пути: длина трека GNSS / ∫v_колёс dt на непрерывных участках status=2.
3. Характеристика привода a(u, v): ускорение по сглаженной скорости колёс, приведённое к нулевому уклону
   (уклон — из pathgraph по привязке GNSS), таблица медиан по (позиция контроллера, скорость).
4. Задержка привода τ и коэффициент уклона k_i — перебором/МНК по R².
Результат — src/tram_backup_odometry/config/model.json (+ reports/calibration.md).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/calibrate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bench'))
from common import MODEL_JSON, ROOT, load_run, splits  # noqa: E402

G0 = 9.80665
R_E = 6378137.0
LAT0, LON0 = 55.804767, 37.419565
DT = 0.1
U_GRID = np.arange(-15, 16)
V_GRID = np.r_[0.0, 0.5, np.arange(1.0, 21.0, 1.0)]

sp = splits()
train = sp['train']
S = pd.read_parquet(ROOT / 'map' / 'pathgraph_samples.parquet')
tree = cKDTree(S[['x', 'y']].to_numpy())
s_hdg, s_grade, s_curv = S.hdg.to_numpy(), S.grade.to_numpy(), S.curv.to_numpy()
K_C = 0.0          # сопротивление на кривых: a = −K_C·|кривизна| (м/с²·м), подбирается вместе с уклоном


def enu_eq(lat, lon):
    return np.radians(lon - LON0) * R_E * math.cos(math.radians(LAT0)), np.radians(lat - LAT0) * R_E


U_RELEASE = 99      # позиция −8, в которую пришли со стороны более глубокого торможения (отпуск тормоза)


def effective_u(u):
    """−8 со стороны −9…−15 — режим отпуска (≈ выбег), со стороны −7…+15 — обычное торможение."""
    out = np.asarray(u).copy()
    prev = 0
    for i, x in enumerate(out):
        if x != -8:
            prev = x
        elif prev < -8:
            out[i] = U_RELEASE
    return out


rows, scale_rows, dist_rows = [], [], []
for b in train:
    run = load_run(b)
    fr, re, cmd = run['front'], run['rear'], run['cmd']
    if len(fr) < 600:
        continue
    t0 = min(fr.t_hdr.iloc[0], re.t_hdr.iloc[0]) / 1e9
    t1 = max(fr.t_hdr.iloc[-1], re.t_hdr.iloc[-1]) / 1e9
    tg = np.arange(t0, t1, DT)
    vf = np.interp(tg, fr.sort_values('t_hdr').t_hdr / 1e9, fr.sort_values('t_hdr').velocity)
    vr = np.interp(tg, re.sort_values('t_hdr').t_hdr / 1e9, re.sort_values('t_hdr').velocity)
    vk = 0.5 * (vf + vr)                                     # км/ч
    c = cmd.sort_values('t_hdr')
    idx = np.searchsorted(c.t_hdr.to_numpy() / 1e9, tg, side='right') - 1
    u = np.where(idx >= 0, c.position.to_numpy()[np.clip(idx, 0, None)], 0)
    u = effective_u(u)
    grade = np.full(len(tg), np.nan)
    fx = run['master_fix']
    fx = fx[fx.status == 2].sort_values('t_hdr')
    if len(fx) > 100:
        vel = run['master_vel'].sort_values('t_hdr')
        tf = fx.t_hdr.to_numpy() / 1e9
        x, y = enu_eq(fx.lat.to_numpy(), fx.lon.to_numpy())
        # уклон — из карты в точке антенны master (≈9.9 м позади base_link, ближе к середине вагона):
        # сила тяжести действует на весь вагон, эффективен средний по длине уклон; оценщик берёт его там же
        vx = np.interp(tf, vel.t_hdr / 1e9, vel.vx)
        vy = np.interp(tf, vel.t_hdr / 1e9, vel.vy)
        vg = np.hypot(vx, vy)
        vwf = np.interp(tf, tg, vk)
        ok = (vg > 2.0) & (vwf > 7.2) & (np.abs(vg - vwf / 3.6) < 1.0)
        if ok.sum() > 100:
            scale_rows.append({'bag': b, 'vehicle': b[:5], 'n': int(ok.sum()),
                               'k': float(np.sum(vg[ok] * vwf[ok]) / np.sum(vwf[ok] ** 2))})
        # масштаб пути: непрерывные участки (шаг ≤ 0.15 с) без скачков
        dtf = np.diff(tf)
        dl = np.hypot(np.diff(x), np.diff(y))
        vw_mid = np.interp(0.5 * (tf[1:] + tf[:-1]), tg, vk)
        good = (dtf < 0.15) & (np.abs(dl / np.maximum(dtf, 1e-3) - vw_mid / 3.6) < 1.5)
        dist_rows.append({'bag': b, 'vehicle': b[:5], 'gnss_m': float(dl[good].sum()),
                          'wheel_kmh_s': float((vw_mid * dtf)[good].sum())})
        # уклон по привязке к карте
        d, j = tree.query(np.c_[x, y], k=8, distance_upper_bound=4.0)
        hd = np.arctan2(vy, vx)
        gr = np.full(len(tf), np.nan)
        cv = np.full(len(tf), np.nan)
        for i in range(len(tf)):
            if vg[i] < 1.0:
                continue
            for dd, jj in zip(d[i], j[i]):
                if np.isfinite(dd) and abs((s_hdg[jj] - hd[i] + np.pi) % (2 * np.pi) - np.pi) < 0.6:
                    gr[i] = s_grade[jj]
                    cv[i] = s_curv[jj]
                    break
        gi = np.searchsorted(tf, tg)
        gi = np.clip(gi, 1, len(tf) - 1)
        near = np.minimum(np.abs(tf[gi] - tg), np.abs(tf[gi - 1] - tg)) < 0.3
        grade = np.where(near, np.interp(tg, tf[np.isfinite(gr)], gr[np.isfinite(gr)]) if np.isfinite(gr).sum() > 10
                         else np.nan, np.nan)
        curv = np.where(near, np.interp(tg, tf[np.isfinite(cv)], cv[np.isfinite(cv)]) if np.isfinite(cv).sum() > 10
                        else np.nan, np.nan)
    else:
        curv = np.full(len(tg), np.nan)
    rows.append(pd.DataFrame({'bag': b, 't': tg, 'vk': vk, 'u': u, 'grade': grade, 'curv': curv}))

D = pd.concat(rows, ignore_index=True)
SC = pd.DataFrame(scale_rows)
DI = pd.DataFrame(dist_rows)
k_all = float(np.median(SC.k))
k_veh = SC.groupby('vehicle').k.median().to_dict()
dist_scale = float(DI.gnss_m.sum() / (DI.wheel_kmh_s.sum() / 3.6) / (k_all * 3.6))
DI['ratio'] = DI.gnss_m / (DI.wheel_kmh_s * k_all) if len(DI) else []
print(f'масштаб колёс: {k_all:.6f} (1/{1 / k_all:.4f}); по вагонам {k_veh}; прогонов {len(SC)}')
print(f'масштаб пути (к скорости): {dist_scale:.5f}; по прогонам медиана {DI.ratio.median():.5f}, '
      f'СКО {DI.ratio.std():.5f}')

# ---------------------------------------------------------------- ускорение
D['v'] = D.vk * k_all
D['a'] = D.groupby('bag').v.transform(lambda s: pd.Series(np.gradient(gaussian_filter1d(s.to_numpy(), 3), DT), index=s.index))
Dm = D[(D.v > 0.3)].dropna(subset=['grade']).copy()
print(f'точек для характеристики: {len(Dm)} (движение, с уклоном)')


def fit_table(df, ucol, k_grade, ycol='a'):
    ac = df[ycol] + k_grade * df.grade + K_C * np.abs(np.nan_to_num(df.curv))
    vb = np.digitize(df.v, 0.5 * (V_GRID[1:] + V_GRID[:-1]))  # ближайший узел сетки
    tab = pd.DataFrame({'u': df[ucol].to_numpy(), 'vb': vb, 'ac': ac.to_numpy(), 'bag': df.bag.to_numpy()})
    # устойчиво к отдельным нетипичным прогонам: медиана помпрогонных медиан, ≥3 прогонов в ячейке
    per_bag = tab.groupby(['u', 'vb', 'bag']).ac.agg(['median', 'size']).reset_index()
    per_bag = per_bag[per_bag['size'] >= 5]
    g = per_bag.groupby(['u', 'vb']).agg(med=('median', 'median'), nb=('median', 'size'), n=('size', 'sum'))
    pooled = tab.groupby(['u', 'vb']).ac.agg(['median', 'size'])
    T = np.full((len(U_GRID), len(V_GRID)), np.nan)
    N = np.zeros((len(U_GRID), len(V_GRID)))
    for (uu, vv), r in pooled.iterrows():
        if not -15 <= uu <= 15:
            continue
        if (uu, vv) in g.index and g.loc[(uu, vv), 'nb'] >= 3 and g.loc[(uu, vv), 'n'] >= 30:
            T[int(uu) + 15, int(vv)] = g.loc[(uu, vv), 'med']
            N[int(uu) + 15, int(vv)] = g.loc[(uu, vv), 'n']
        elif r['size'] >= 50:
            T[int(uu) + 15, int(vv)] = r['median']
            N[int(uu) + 15, int(vv)] = r['size']
    # физически невозможные ячейки (торможение сильнее разгоняет, чем выбег, и наоборот) — выбрасываются
    a0 = T[15]
    for i, uu in enumerate(U_GRID):
        bad = (uu < 0) & (T[i] > a0 + 0.05) if uu < 0 else ((T[i] < a0 - 0.05) if uu > 0 else np.zeros(len(V_GRID), bool))
        T[i][np.nan_to_num(bad, nan=False).astype(bool)] = np.nan
    N[~np.isfinite(T)] = 0
    return T, N


def pav(y):
    """Изотоническая (неубывающая) регрессия, алгоритм PAV."""
    y = list(map(float, y))
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * n for v, n in blocks])


def fill(TN):
    """Полная характеристика по разреженной таблице медиан.
    Торможение (u<0): разделимая модель a = a0(v) + B(u)·φ(v) — уровень торможения позиции × зависимость
    от скорости (затухание электрического торможения на малой скорости); подгонка ALS с весами √n,
    B(u) монотонно (PAV). Тяга (u>0): таблица, интерполяция по v и u, монотонность по позиции.
    Выбег (u=0): строка медиан (данных много)."""
    T, N = TN
    T = T.copy()
    iz = 15
    ok0 = np.isfinite(T[iz])
    a0 = np.interp(V_GRID, V_GRID[ok0], T[iz][ok0])
    # --- торможение
    Br = T[:iz] - a0[None, :]
    W = np.sqrt(N[:iz]) * np.isfinite(Br)
    Br = np.nan_to_num(Br)
    g = np.ones(len(V_GRID))
    b = np.zeros(iz)
    for _ in range(30):
        b = (W * Br * g[None, :]).sum(1) / np.maximum((W * g[None, :] ** 2).sum(1), 1e-9)
        b = pav(b)                                   # u=-15 (индекс 0) — самое сильное торможение
        g = (W * Br * b[:, None]).sum(0) / np.maximum((W * b[:, None] ** 2).sum(0), 1e-9)
        has = (W.sum(0) > 0)
        g = np.where(has, g, np.nan)
        okg = np.isfinite(g)
        g = np.interp(V_GRID, V_GRID[okg], g[okg])
        sc = np.max(g)
        g, b = g / sc, b * sc
    T[:iz] = a0[None, :] + b[:, None] * g[None, :]
    T[iz] = a0
    # --- тяга
    for i in range(iz + 1, len(U_GRID)):
        ok = np.isfinite(T[i])
        if ok.sum() >= 2:
            T[i] = np.interp(V_GRID, V_GRID[ok], T[i][ok])
        elif ok.sum() == 1:
            T[i] = T[i][ok][0]
    for j in range(len(V_GRID)):
        col = T[iz:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            T[iz:, j] = np.interp(U_GRID[iz:], U_GRID[iz:][ok], col[ok])
        T[iz + 1:, j] = np.maximum(T[iz + 1:, j], a0[j])
        T[iz:, j] = pav(T[iz:, j])
    return T


def lag(x, bag, tau):
    """Инерционное звено первого порядка (дискретно, шаг DT), по каждому прогону отдельно."""
    if tau <= 0:
        return x
    al = 1.0 - math.exp(-DT / tau)
    out = np.empty_like(x)
    prev_b, y = None, 0.0
    for i in range(len(x)):
        if bag[i] != prev_b:
            prev_b, y = bag[i], x[i]
        y += al * (x[i] - y)
        out[i] = y
    return out


def predict(df, ucol, T, k_grade, release=None):
    uu = df[ucol].to_numpy().astype(int)
    ui = np.clip(uu, -15, 15) + 15
    out = np.empty(len(df))
    for i in np.unique(ui):
        m = ui == i
        out[m] = np.interp(df.v.to_numpy()[m], V_GRID, T[i])
    m = uu == U_RELEASE
    if m.any():
        out[m] = np.interp(df.v.to_numpy()[m], V_GRID, release if release is not None else T[15])
    return out - k_grade * df.grade.to_numpy() - K_C * np.abs(np.nan_to_num(df.curv.to_numpy()))


D['da'] = D.groupby('bag').a.transform(lambda s: pd.Series(np.gradient(gaussian_filter1d(s.to_numpy(), 2), DT), index=s.index))
Dm['da'] = D.da.reindex(Dm.index)
best = None
for L in range(0, 7):
    Dm[f'u{L}'] = D.groupby('bag').u.shift(L).reindex(Dm.index)
for tau in (0.0, 0.3, 0.5, 0.8, 1.2):
    for L in range(0, 7):
        q = Dm.dropna(subset=[f'u{L}']).copy()
        q['y'] = q.a + tau * q.da              # a_ss ≈ a + τ·ȧ (обратное к инерционному звену)
        k = 0.8 * G0
        K_C = 0.0
        for _ in range(4):
            T = fill(fit_table(q, f'u{L}', k, 'y'))
            base = predict(q.assign(grade=0.0, curv=0.0), f'u{L}', T, 0.0)
            res = q.y.to_numpy() - base
            X = np.c_[-q.grade.to_numpy(), -np.abs(np.nan_to_num(q.curv.to_numpy()))]
            k, K_C = (float(c) for c in np.linalg.lstsq(X, res, rcond=None)[0])
        # проверка прямым прогоном: a_hat = lag(a_ss) против измеренного a (на полной сетке по времени)
        full = D.copy()
        full['u_l'] = D.groupby('bag').u.shift(L).fillna(0)
        full['grade'] = full.grade.fillna(0.0)
        full['curv'] = full.curv.fillna(0.0)
        ass = predict(full.assign(v=full.v.clip(lower=0)), 'u_l', T, k)
        ahat = lag(ass, full.bag.to_numpy(), tau)
        m = full.index.isin(q.index)
        r2 = 1 - np.sum((full.a[m] - ahat[m]) ** 2) / np.sum((full.a[m] - full.a[m].mean()) ** 2)
        sig = float(np.std(full.a[m] - ahat[m]))
        print(f'  τ={tau:.1f} с, задержка {L * DT:.1f} с: R²={r2:.3f}, СКО={sig:.3f}, k_i={k:.2f}, k_кр={K_C:.2f}')
        if best is None or r2 > best[0]:
            best = (r2, L, T, k, sig, tau, K_C)
r2, L, T, k_grade, sig, tau, K_C = best
qb = Dm.dropna(subset=[f'u{L}']).copy()
qb['y'] = qb.a + tau * qb.da
T_raw, _ = fit_table(qb, f'u{L}', k_grade, 'y')
print('сырые медианы (u=-15..-5, v=0..8):')
print(pd.DataFrame(T_raw[:11, :10], index=U_GRID[:11], columns=V_GRID[:10]).round(2).to_string())
q = Dm.dropna(subset=[f'u{L}']).copy()
q = q[q[f'u{L}'] == U_RELEASE]
q['y'] = q.a + tau * q.da + k_grade * q.grade + K_C * np.abs(np.nan_to_num(q.curv))
vb = np.digitize(q.v, 0.5 * (V_GRID[1:] + V_GRID[:-1]))
g = q.groupby(vb).y.agg(['median', 'size'])
rr = np.full(len(V_GRID), np.nan)
for j, r in g.iterrows():
    if r['size'] >= 30:
        rr[j] = r['median']
ok = np.isfinite(rr)
release_row = np.minimum(np.interp(V_GRID, V_GRID[ok], rr[ok]), T[15]) if ok.sum() >= 2 else T[15].copy()
print('режим отпуска (−8 после −9…−15):', np.round(release_row, 2).tolist(), 'точек', len(q))
print(f'выбрано: τ={tau:.1f} с, задержка {L * DT:.1f} с, R²={r2:.3f}, СКО={sig:.3f} м/с², k_i={k_grade:.2f}, '
      f'k_кр={K_C:.2f} (w_r = {K_C / G0 * 1000:.0f}/R Н/кН)')

out = {
    'wheel_scale': k_all,
    'wheel_scale_by_vehicle': k_veh,
    'distance_scale': dist_scale,
    'cmd_delay_s': round(L * DT, 2),
    'lag_tau_s': tau,
    'k_grade': round(k_grade, 3),
    'k_curve': round(K_C, 3),
    'a_sigma': round(sig, 3),
    'u_grid': U_GRID.tolist(),
    'v_grid': V_GRID.tolist(),
    'a_table': np.round(T, 4).tolist(),
    'a_release_m8': np.round(release_row, 4).tolist(),
    'fit': {'r2': round(float(r2), 4), 'n_points': int(len(Dm)), 'train_runs': len(train)},
    'note': 'a_table — ускорение привода при нулевом уклоне, м/с², строки u=-15..15, столбцы v_grid (м/с)',
}
MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
MODEL_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')

tab = pd.DataFrame(T, index=U_GRID, columns=[f'{v:g}' for v in V_GRID]).round(2)
rep = ROOT / 'reports' / 'calibration.md'
rep.write_text(
    '# Калибровка модели (train)\n\n'
    f'- обучающих уникальных прогонов: {len(train)}; точек характеристики: {len(Dm)}\n'
    f'- масштаб колёс: {k_all:.6f} м/с на ед. (1/{1 / k_all:.4f}); по вагонам: '
    + ', '.join(f'{kk}: 1/{1 / vv:.4f}' for kk, vv in k_veh.items()) + '\n'
    f'- масштаб пути: {dist_scale:.5f} (медиана по прогонам {DI.ratio.median():.5f}, СКО {DI.ratio.std():.5f})\n'
    f'- задержка привода: {L * DT:.1f} с; постоянная инерционного звена τ = {tau:.1f} с; коэффициент уклона k_i = {k_grade:.2f} м/с² (g = 9.81); '
    f'сопротивление на кривых {K_C:.2f}/R м/с² (w_r = {K_C / G0 * 1000:.0f}/R Н/кН)\n'
    f'- R² модели ускорения: {r2:.3f}; СКО остатка: {sig:.3f} м/с²\n\n'
    '## a(u, v) при нулевом уклоне, м/с²\n\n' + tab.to_markdown() + '\n', encoding='utf-8')
SC.to_csv(ROOT / 'reports' / 'calibration_scale.csv', index=False)
DI.to_csv(ROOT / 'reports' / 'calibration_distance.csv', index=False)
print('записано', MODEL_JSON)

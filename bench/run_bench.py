"""Офлайн-стенд валидации: проигрывание прогонов через оценщик, метрики жюри, инъекция сбоев.

Примеры (из корня проекта):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/run_bench.py --tag baseline
  ... --scenarios clean,slip,dropout30 --split val -j 8
  ... --only 30618_01f73500 --scenarios clean --plots
  ... --set gate_sigma=3 --set use_map=false --tag nomap

Результат: reports/bench/<tag>/per_run.csv, summary.csv, summary.md, plots/*.png
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import faults
import metrics
from common import ROOT, events, load_run, make_estimator, reference, run_estimator, splits

KEY = [('v_rmse', 'RMSE v, м/с'), ('v_mae', 'MAE v'), ('v_bias', 'bias v'), ('v_trans_rmse', 'RMSE v перех.'),
       ('v_fault_rmse', 'RMSE v в сбое'), ('final_drift_pct', 'дрейф конеч., %'), ('along_mae', 'along MAE, м'),
       ('along_rmse', 'along RMSE'), ('along_max', 'along MAX'), ('cross_rmse', 'cross RMSE'),
       ('xtrack_map_rmse', 'от карты RMSE'), ('err3d_rmse', '3D RMSE'), ('err3d_max', '3D MAX'),
       ('slip_detect_rate', 'детект. проск.'), ('slip_false_frac', 'ложн. флаг'),
       ('rate_hz', 'частота, Гц'), ('proc_p99_ms', 'обраб. p99, мс')]


def parse_val(s):
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def job(bag, scen, overrides, gnss_window, want_series):
    run = load_run(bag)
    ref = reference(run)
    frun, wins = faults.apply(run, bag, scen)
    use_map = overrides.get('use_map', True)
    est = make_estimator({k: v for k, v in overrides.items() if k != 'use_map'}, use_map=use_map)
    out, proc = run_estimator(est, events(frun, gnss_window))
    row = {'bag': bag, 'scenario': scen, 'mode': out['mode'].iloc[-1] if len(out) else '',
           'duration_s': float(out.t.iloc[-1] - out.t.iloc[0]) if len(out) else 0.0, 'n_faults': len(wins)}
    series = None
    if ref is not None and len(out):
        row.update(metrics.speed_metrics(out, ref, wins))
        pm, series = metrics.position_metrics(out, ref)
        row.update(pm)
    row.update(metrics.timing_metrics(out, proc))
    if scen in faults.SLIP_SCENARIOS:
        row.update(metrics.slip_metrics(out, wins))
    else:
        row['slip_false_frac'] = float(out.slip.mean()) if len(out) else np.nan
    if not want_series:
        return row, None
    return row, {'out': out, 'ref': ref, 'pos': series, 'wins': wins}


def plot_run(bag, scen, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, ref, ser, wins = data['out'], data['ref'], data['pos'], data['wins']
    t0 = out.t.iloc[0]
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    if ref is not None:
        vr = ref['vel']
        ax[0].plot(vr.t - t0, vr.v, lw=0.8, color='0.5', label='GNSS (эталон)')
    ax[0].plot(out.t - t0, out.v, lw=0.8, color='C0', label='оценка')
    ax[0].set_ylabel('v, м/с')
    ax[0].legend(loc='upper right')
    if ref is not None:
        m_i, m_e = metrics.match(out, ref['vel'].t.to_numpy())
        ax[1].plot(ref['vel'].t.to_numpy()[m_i] - t0, out.v.to_numpy()[m_e] - ref['vel'].v.to_numpy()[m_i], lw=0.6)
    ax[1].set_ylabel('ошибка v, м/с')
    ax[1].set_ylim(-1.5, 1.5)
    if ser is not None and len(ser):
        ax[2].plot(ser.t - t0, ser.along, lw=0.8, label='along')
        ax[2].plot(ser.t - t0, ser.cross, lw=0.8, label='cross')
        ax[2].plot(ser.t - t0, ser.du, lw=0.8, label='up')
        ax[2].legend(loc='upper left')
    ax[2].set_ylabel('ошибка положения, м')
    ax[3].plot(out.t - t0, out.slip.astype(int), lw=0.8, color='C3', label='флаг проскальзывания')
    ax[3].plot(out.t - t0, out.wheels_ok, lw=0.8, color='C2', label='тележек в работе')
    ax[3].legend(loc='upper right')
    ax[3].set_xlabel('t, с')
    for a0, a1 in wins:
        for a in ax:
            a.axvspan(a0 - t0, a1 - t0, color='orange', alpha=0.2)
    fig.suptitle(f'{bag} — {scen}')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--scenarios', default=','.join(faults.SCENARIOS))
    ap.add_argument('--gnss-window', type=float, default=5.0)
    ap.add_argument('--set', action='append', default=[], help='параметр оценщика key=value')
    ap.add_argument('--plots', action='store_true', help='графики по каждому прогону/сценарию')
    ap.add_argument('-j', type=int, default=8)
    a = ap.parse_args()
    sp = splits()
    bags = a.only or (sp['val'] if a.split == 'val' else sp['train'] if a.split == 'train' else sp['val'] + sp['train'])
    scens = a.scenarios.split(',')
    overrides = {k: parse_val(v) for k, v in (s.split('=', 1) for s in a.set)}
    outdir = ROOT / 'reports' / 'bench' / a.tag
    (outdir / 'plots').mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rows = []
    with ProcessPoolExecutor(a.j) as ex:
        futs = {ex.submit(job, b, s, overrides, a.gnss_window, a.plots): (b, s) for b in bags for s in scens}
        for f in as_completed(futs):
            b, s = futs[f]
            try:
                row, data = f.result()
            except Exception as e:  # noqa: BLE001 — стенд не должен падать целиком из-за одного прогона
                row, data = {'bag': b, 'scenario': s, 'error': repr(e)}, None
            rows.append(row)
            if data is not None:
                plot_run(b, s, data, outdir / 'plots' / f'{b}_{s}.png')
    df = pd.DataFrame(rows).sort_values(['scenario', 'bag'])
    df.to_csv(outdir / 'per_run.csv', index=False)
    cols = [k for k, _ in KEY if k in df.columns]
    agg = df.groupby('scenario')[cols].median().reindex([s for s in scens if s in set(df.scenario)])
    agg_mean = df.groupby('scenario')[cols].mean().reindex(agg.index)
    agg.to_csv(outdir / 'summary_median.csv')
    agg_mean.to_csv(outdir / 'summary_mean.csv')
    names = dict(KEY)
    md = [f'# Стенд валидации: `{a.tag}`\n',
          f'- прогонов: {len(bags)} ({a.split}); сценариев: {len(scens)}; GNSS — только первые {a.gnss_window:g} с',
          f'- параметры: {json.dumps(overrides, ensure_ascii=False) if overrides else "по умолчанию"}',
          f'- время счёта: {time.time() - t_start:.0f} с\n',
          '## Медиана по прогонам\n', agg.rename(columns=names).to_markdown(floatfmt='.3f'),
          '\n## Среднее по прогонам\n', agg_mean.rename(columns=names).to_markdown(floatfmt='.3f')]
    if 'error' in df.columns and df.error.notna().any():
        md += ['\n## Ошибки\n', df[df.error.notna()][['bag', 'scenario', 'error']].to_markdown(index=False)]
    (outdir / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

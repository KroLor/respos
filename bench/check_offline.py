"""Офлайн-прогон ядра на bag стенда организаторов (extracted_check/, эталон check-code/_ref.parquet — их
/localization/kinematic_state). Те же метрики, что у hackathon_solution_checker (сопоставление по stamp).

PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/check_offline.py [--set k=v ...]
"""
import sys

import numpy as np
import pandas as pd

import common
from common import ROOT, events, make_estimator, run_estimator


def main():
    ov = {}
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--set':
            k, v = args[i + 1].split('=')
            ov[k] = type(getattr(__import__('tram_backup_odometry.estimator', fromlist=['Params']).Params, k))(eval(v))
    common.EX = ROOT / 'extracted_check'
    out, _ = run_estimator(make_estimator(ov), events(common.load_run('30618_88aea4d9')))
    R = pd.read_parquet(ROOT / 'check-code' / '_ref.parquet').sort_values('th').reset_index(drop=True)
    P = out[out.pos_valid].reset_index(drop=True)
    tp = P.t.to_numpy()
    j = np.clip(np.searchsorted(tp, R.th.to_numpy()), 1, len(tp) - 1)
    j = np.where(np.abs(tp[j - 1] - R.th) <= np.abs(tp[j] - R.th), j - 1, j)
    ok = np.abs(tp[j] - R.th.to_numpy()) <= 0.05
    Rm, Pm = R[ok].reset_index(drop=True), P.iloc[j[ok]].reset_index(drop=True)
    dx, dy, dz = Pm.e - Rm.x, Pm.n - Rm.y, Pm.u - Rm.z
    v_ref = np.hypot(Rm.vx, Rm.vy)
    hd = np.arctan2(np.gradient(Rm.y.rolling(25, center=True, min_periods=1).mean()),
                    np.gradient(Rm.x.rolling(25, center=True, min_periods=1).mean()))
    hd = pd.Series(np.where(v_ref > 0.5, hd, np.nan)).ffill().bfill().to_numpy()
    along = dx * np.cos(hd) + dy * np.sin(hd)
    cross = -dx * np.sin(hd) + dy * np.cos(hd)
    d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    ev = Pm.v - Rm.vx
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))  # noqa: E731
    print(f'n={ok.sum()} скорость RMSE {rms(ev):.3f} max {np.abs(ev).max():.3f} | 3D RMSE {rms(d3):.2f} max {d3.max():.2f} | '
          f'x {rms(dx):.2f} y {rms(dy):.2f} z {rms(dz):.2f} | along RMSE {rms(along):.2f} max {np.abs(along).max():.2f} '
          f'| cross RMSE {rms(cross):.2f} max {np.abs(cross).max():.2f} | в конце {d3.iloc[-1]:.2f}')
    T = pd.DataFrame({'t': Rm.th - Rm.th.iloc[0], 'along': along, 'cross': cross, 'd3': d3, 'dz': dz, 'v': Rm.vx})
    T.to_csv(ROOT / 'reports' / 'check' / 'offline_errors.csv', index=False)
    if '-v' in args:
        print(T.iloc[::1500].round(2).to_string(index=False))


if __name__ == '__main__':
    main()

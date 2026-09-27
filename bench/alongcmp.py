"""Ошибка вдоль пути по времени для двух карт: python alongcmp.py <bag> <scenario> <каталог data старой карты>"""
import sys
import numpy as np
import pandas as pd
import common
import faults
from common import events, load_run, make_estimator, reference, run_estimator


def along_err(out, ref):
    p = ref['pos']
    j = np.clip(np.searchsorted(p.t.to_numpy(), out.t.to_numpy()), 0, len(p) - 1)
    dx, dy = out.e.to_numpy() - p.e.to_numpy()[j], out.n.to_numpy() - p.n.to_numpy()[j]
    return dx, dy


b, scen, old = sys.argv[1], sys.argv[2], sys.argv[3]
run = load_run(b); ref = reference(run); fr, w = faults.apply(run, b, scen)
res = {}
for tag, path in (('new', common.MAP_JSON), ('old', common.ROOT / old / 'pathgraph.json')):
    common.MAP_JSON = path
    est = make_estimator(); out, _ = run_estimator(est, events(fr))
    res[tag] = (out, est.n_stop_corr)
print('окна сбоя:', [(round(a - res['new'][0].t.iloc[0]), round(c - res['new'][0].t.iloc[0])) for a, c in w])
print(ref['pos'].columns.tolist())
import metrics
S = {}
for tag in ('new', 'old'):
    out, nst = res[tag]
    pm, ser = metrics.position_metrics(out, ref)
    S[tag] = ser.set_index('t').along
    print(tag, 'остановок-коррекций', nst, 'along RMSE', round(pm['along_rmse'], 2))
D = pd.DataFrame(S).dropna()
D['t'] = D.index - D.index[0]
D['diff'] = D.new - D.old
print(D.iloc[::400].round(2).to_string(index=False))

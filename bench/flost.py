"""Где копится ошибка пути в сценарии: python flost.py <bag> <scenario>"""
import sys
import numpy as np
import pandas as pd
import faults
from common import events, load_run, make_estimator, reference, run_estimator
b, scen = sys.argv[1], sys.argv[2]
run = load_run(b); ref = reference(run); fr, w = faults.apply(run, b, scen)
est = make_estimator(); out, _ = run_estimator(est, events(fr))
vr = ref['vel']
t0 = out.t.iloc[0]
o = out.copy()
o['vref'] = np.interp(o.t, vr.t, vr.v)
o['ev'] = o.v - o.vref
dt = np.diff(o.t, prepend=o.t.iloc[0])
o['ds_err'] = np.cumsum(o.ev * dt)
g = o.groupby(((o.t - t0) // 30).astype(int)).agg(t=('t', 'first'), ev=('ev', 'mean'), evmax=('ev', lambda x: x.abs().max()),
                                                   ds=('ds_err', 'last'), v=('vref', 'mean'), wheels=('wheels_ok', 'min'), slip=('slip', 'mean'))
g['t'] = g.t - t0
print(g.round(3).to_string())

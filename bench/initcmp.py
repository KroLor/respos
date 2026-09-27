"""Выставка на двух картах: python initcmp.py <bag> <scenario> <каталог data старой карты>"""
import sys
import common
import faults
from common import events, load_run, make_estimator
b, scen, old = sys.argv[1], sys.argv[2], sys.argv[3]
run = load_run(b); fr, _ = faults.apply(run, b, scen)
ev = events(fr)
for tag, path in (('new', common.MAP_JSON), ('old', common.ROOT / old / 'pathgraph.json')):
    common.MAP_JSON = path
    est = make_estimator()
    t0 = None
    for _, kind, stamp, val in ev:
        t0 = t0 or stamp
        if kind == 'cmd':
            est.on_cmd(stamp, int(val))
        elif kind in ('front', 'rear'):
            est.on_wheel(kind, stamp, float(val))
        else:
            est.on_fix(kind, stamp, *val)
        if est.gnss_done:
            o = est.output(stamp)
            print(tag, f't={stamp - t0:.2f} mode={est.mode} anchor={est.anchor} s={est.s:.2f} v={est.x[0]:.2f} out=({o.e:.2f},{o.n:.2f})')
            break

"""Коррекции по остановкам на двух картах: python stopcmp.py <bag> <scenario> <каталог data старой карты>"""
import sys
import common
import faults
from common import events, load_run, make_estimator, run_estimator
from tram_backup_odometry import estimator as E
b, scen, old = sys.argv[1], sys.argv[2], sys.argv[3]
run = load_run(b); fr, _ = faults.apply(run, b, scen)
orig = E.Estimator._stop_logic
T0 = [None]


def sl(self, stamp):
    T0[0] = T0[0] or stamp
    n0 = self.n_stop_corr
    a0 = list(self.anchor) if self.mode == 'map' and self.anchor is not None else None
    orig(self, stamp)
    if self.n_stop_corr > n0:
        print(f'  t={stamp - T0[0]:7.1f} коррекция {self.last_corr:+.2f} м: {a0[:2]} -> {self.anchor[:2]} sig={self.sigma_along():.2f}')


E.Estimator._stop_logic = sl
for tag, path in (('new', common.MAP_JSON), ('old', common.ROOT / old / 'pathgraph.json')):
    common.MAP_JSON = path
    T0[0] = None
    print(tag)
    run_estimator(make_estimator(), events(fr))

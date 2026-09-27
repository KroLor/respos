"""Трассировка выбора ветки на стрелках: python forkdbg.py <bag>"""
import sys
import numpy as np
from common import events, load_run, make_estimator, run_estimator
from tram_backup_odometry import estimator as E
from tram_backup_odometry.pathmap import branch_fit

orig_fork = E.Estimator._fork_logic
orig_adv = E.Estimator._advance_map


def adv(self):
    e0 = self.anchor[0] if self.anchor is not None else None
    f0 = self.fork
    orig_adv(self)
    if self.fork is not None and self.fork is not f0 and self.fork['k'] == 0:
        print(f"стрелка: ребро {e0} -> {self.anchor[0]}, s0={self.fork['s0']:.1f}, ветки {list(self.fork['br'])}")


def fork(self, force=False):
    f = self.fork
    n0 = self.n_fork_switch
    orig_fork(self, force)
    if f is not None and (f['k'] % 2 == 0 or self.n_fork_switch > n0) and self.s - f['s0'] >= self.p.fork_min_m:
        step = self.map.fork_step
        sa = np.array([q[0] for q in self.sv]); va = np.array([q[1] for q in self.sv])
        v = np.interp(f['s0'] + step * np.arange(f['k'] + 1), sa, va)
        fit = {b: tuple(round(x, 1) for x in branch_fit(v, *pr)) for b, pr in f['br'].items()}
        print(f"  d={self.s - f['s0']:.0f} v={self.x[0]:.2f} cur={f['cur']} fit={fit}" + ("  ПЕРЕУСТАНОВКА" if self.n_fork_switch > n0 else ''))


E.Estimator._fork_logic = fork
E.Estimator._advance_map = adv
import common
import faults
run = common.load_run(sys.argv[1]) if len(sys.argv) < 3 or sys.argv[2].startswith('sc:') else None
if run is None:
    common.EX = common.ROOT / sys.argv[2]
    run = common.load_run(sys.argv[1])
elif len(sys.argv) > 2:
    run, _ = faults.apply(run, sys.argv[1], sys.argv[2][3:])
run_estimator(make_estimator(), events(run))

"""Эхо-сеть (ESN) — обучаемая часть модели продольной динамики.

Роль: поправка к физической модели ускорения, обученная на многошаговый прогноз «без колёс» (замкнутый
контур, горизонт 30 с). Резервуар работает на сетке 10 Гц и получает:
  * внешние воздействия — позицию контроллера (режим, знак), уклон пути, изменение позиции за 1 с;
  * пока колёса достоверны — измеренный остаток модели (ускорение по колёсам − физика) и флаг «колёса есть».
Когда колёса недостоверны (пропадание, проскальзывание, залипание), остаток обнуляется и флаг = 0: сеть
продолжает прогноз, опираясь на память о текущем прогоне («память до провала»).
Контекст прогона — онлайн-МНК (с забыванием) множителей физики: тяга, торможение, выбег, смещение.
Выход: множители к тяговой и тормозной составляющей и добавка к ускорению (вид E3), либо только добавка.
Ансамбль из нескольких резервуаров — ускорение усредняется.
"""
from __future__ import annotations

import math

import numpy as np

RELEASE = 99
DT = 0.1
LAG_R = 5


class _Member:
    def __init__(self, W, Win, w, leak):
        self.W, self.Win, self.w, self.leak = np.asarray(W), np.asarray(Win), np.asarray(w), float(leak)
        self.x = np.zeros(self.W.shape[0])

    def step(self, inp):
        self.x = (1.0 - self.leak) * self.x + self.leak * np.tanh(self.W @ self.x + self.Win @ inp)


class ResidualESN:
    @classmethod
    def from_file(cls, path: str) -> 'ResidualESN':
        import json
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def __init__(self, params: dict):
        self.kind = params['kind']                         # 'E1' | 'E2' | 'E3'
        self.members = [_Member(m['W'], m['Win'], m['w'], m['leak']) for m in params['members']]
        self.phys = np.asarray(params['phys'], float)      # множители C1: тяга, торм., выбег, уклон, log τ, смещение
        self.lam = float(params.get('rls_lambda', 0.9995))
        self.base_tau = float(params.get('base_tau', 0.3))
        self.lag_r = int(params.get('lag_r', 5))
        self.reset()

    def reset(self):
        for m in self.members:
            m.x[:] = 0.0
        self.th = np.array([1.0, 1.0, 1.0, 0.0])
        self.P = np.eye(4) * 0.02
        self.corr = (0.0, 0.0, 0.0)            # (множитель тяги, множитель торможения, добавка) — среднее ансамбля
        self.u_hist = []                       # позиции за последнюю секунду (сетка 10 Гц)

    # ------------------------------------------------------------------ шаг сетки 10 Гц
    def grid_step(self, u_eff: int, v: float, a_f: float, grade: float, trusted: bool, r_in: float = 0.0,
                  lag_comps=None):
        """trusted — колёсам сейчас верим (флаг); r_in — измеренный остаток (0, если его нет, например на стоянке);
        lag_comps — (aT, aB, a0, a_уклон, a_изм) физики после инерционного звена в момент, к которому относится
        остаток (для онлайн-МНК контекста), или None."""
        rel = u_eff == RELEASE
        uu = -8.0 if rel else float(u_eff)
        self.u_hist.append(uu)
        if len(self.u_hist) > 11:
            self.u_hist.pop(0)
        up = self.u_hist[0]
        exo = np.array([uu / 15.0, float(rel), float(uu > 0), float(uu < 0), 20.0 * grade, (uu - up) / 15.0])
        flag = bool(trusted)
        inp = np.r_[exo, 2.0 * float(np.clip(r_in, -1.5, 1.5)) if flag else 0.0, 1.0 if flag else 0.0]
        if flag and lag_comps is not None:
            self._rls(lag_comps, r_in)
        vs = v / 15.0
        dyn = np.array([vs, vs * vs, a_f, 1.0])
        ctx = self.th - np.array([1.0, 1.0, 1.0, 0.0])
        kT = kB = add = 0.0
        for m in self.members:
            m.step(inp)
            f = np.r_[m.x, exo, dyn, ctx] if self.kind in ('E2', 'E3') else np.r_[m.x, exo, dyn]
            if self.kind == 'E3':
                kT += float(m.w[0] @ f)
                kB += float(m.w[1] @ f)
                add += float(m.w[2] @ f)
            else:
                add += float(m.w[0] @ f)
        k = len(self.members)
        self.corr = (kT / k, kB / k, add / k)

    def _rls(self, comps, r_meas):
        """Контекст прогона: y = a_изм − a_уклон = gT·aT + gB·aB + g0·a0 + c (по данным с колёсами)."""
        aT, aB, a0, ag, a_meas = comps
        z = np.array([aT, aB, a0, 1.0])
        e = (a_meas - ag) - z @ self.th
        Pz = self.P @ z
        gain = Pz / (self.lam + z @ Pz)
        self.th = self.th + gain * e
        self.P = (self.P - np.outer(gain, Pz)) / self.lam
        tr = np.trace(self.P)
        if tr > 1.0:
            self.P *= 1.0 / tr

    # ------------------------------------------------------------------ применение
    def steady_accel(self, aT: float, aB: float, a0: float, grade_term: float, active: bool = True) -> float:
        """Установившееся ускорение физики с множителями C1 и (active) поправкой сети к тяге/торможению.
        Сеть обучена для режима «колёсам не верим» — только в нём её поправка и применяется."""
        pT, pB, p0, pk, _, pc = self.phys
        kT, kB, _ = self.corr if active else (0.0, 0.0, 0.0)
        return (1 + pT + kT) * aT + (1 + pB + kB) * aB + (1 + p0) * a0 + pc + (1 + pk) * grade_term

    @property
    def lag_tau(self) -> float:
        return 0.3 * math.exp(self.phys[4])

    @property
    def additive(self) -> float:
        return self.corr[2]

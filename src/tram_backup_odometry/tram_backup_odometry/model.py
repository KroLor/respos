"""Модель тягового привода и продольной динамики (удельные силы, м/с²).

a_ss(u, v, i, κ) = a_drive(u(t − τ_d), v) − k_i · i − k_κ · |κ|   — установившееся ускорение
T_a · ȧ + a = a_ss                                            — инерционность привода (звено 1-го порядка)
  a_drive — таблично-аппроксимирующая характеристика «позиция контроллера × скорость», идентифицированная
            офлайн по записям (включает тягу/торможение привода и основное сопротивление движению);
  τ_d     — транспортная задержка привода (команда → ускорение), T_a — постоянная нарастания силы;
  i       — уклон пути из карты (dz/ds), k_i — эффективный коэффициент уклона (≈ g с учётом инерции
            вращающихся масс и сглаживания карты);
  κ       — кривизна пути из карты (1/R), k_κ — сопротивление на кривых (по данным w_r ≈ 360/R Н/кН).
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

RELEASE = 99      # эффективная позиция: −8 при отпуске тормоза


class DriveModel:
    def __init__(self, params: dict):
        self.u_grid = np.asarray(params['u_grid'], float)
        self.v_grid = np.asarray(params['v_grid'], float)
        self.table = np.asarray(params['a_table'], float)          # [len(u_grid), len(v_grid)]
        self.k_grade = float(params.get('k_grade', 9.81))
        self.k_curve = float(params.get('k_curve', 0.0))
        self.delay = float(params.get('cmd_delay_s', 0.0))
        self.lag_tau = float(params.get('lag_tau_s', 0.0))
        self.sigma = float(params.get('a_sigma', 0.25))
        self.release = np.asarray(params.get('a_release_m8', self.table[int(-self.u_grid[0])]), float)
        allrows = np.vstack([self.table, self.release[None, :]])
        self.env_max = allrows.max(axis=0)       # огибающая: максимум тяги по всем позициям, м/с²
        self.env_min = allrows.min(axis=0)       # максимум торможения
        self._u0 = int(self.u_grid[0])
        self._hist: deque = deque()          # (stamp, u_eff)
        self._prev = 0                       # последняя позиция, отличная от −8

    @classmethod
    def from_file(cls, path: str) -> 'DriveModel':
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def reset(self):
        self._hist.clear()
        self._prev = 0

    def push_cmd(self, stamp: float, u: int):
        """Позиция −8 по данным двузначна: при углублении торможения (со стороны −7) — обычная тормозная
        ступень, при отпуске (со стороны −9…−15) — режим, близкий к выбегу. Кодируется как RELEASE."""
        u = int(u)
        if self._hist and stamp < self._hist[-1][0]:
            # запоздавшая команда (очередь в начале bag) — вставка по времени
            i = len(self._hist)
            while i > 0 and self._hist[i - 1][0] > stamp:
                i -= 1
            prev = self._hist[i - 1][1] if i > 0 else 0
            if u == -8 and prev != RELEASE and prev < -8:
                u = RELEASE
            elif u == -8 and prev == RELEASE:
                u = RELEASE
            self._hist.insert(i, (stamp, u))
            return
        if u != -8:
            self._prev = u
        elif self._prev < -8:
            u = RELEASE
        self._hist.append((stamp, u))
        while len(self._hist) > 400:
            self._hist.popleft()

    def u_at(self, t: float) -> int:
        """Позиция контроллера, действовавшая в момент t − τ."""
        tq = t - self.delay
        u = None
        for ts, uu in reversed(self._hist):
            if ts <= tq:
                u = uu
                break
        if u is None:
            u = self._hist[0][1] if self._hist else 0
        # отбрасываем историю, которая уже не понадобится
        while len(self._hist) > 2 and self._hist[1][0] <= tq - 1.0:
            self._hist.popleft()
        return u

    def accel(self, u: int, v: float, grade: float, curv: float = 0.0) -> float:
        if u == RELEASE:
            row = self.release
        else:
            row = self.table[int(np.clip(round(u), self.u_grid[0], self.u_grid[-1])) - self._u0]
        a = float(np.interp(v, self.v_grid, row))
        return a - self.k_grade * grade - self.k_curve * abs(curv)

    def envelope(self, v: float, grade: float, curv: float = 0.0) -> tuple[float, float]:
        """Физически достижимый диапазон ускорения при скорости v (любая позиция контроллера)."""
        g = self.k_grade * grade + self.k_curve * abs(curv)
        return float(np.interp(v, self.v_grid, self.env_min)) - g, float(np.interp(v, self.v_grid, self.env_max)) - g


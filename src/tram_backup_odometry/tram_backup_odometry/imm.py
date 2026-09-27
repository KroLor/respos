"""IMM из трёх UKF: норма / боксование / юз (общий режим обеих тележек).

Состояние каждого режима x = [v, a_f]: скорость и выход инерционного звена привода. Прогноз — сигма-точки
через нелинейную модель привода (таблица, звено, отсечки «не катится назад», ESN), шум процесса — на a_f.
Измерение (скорость тележки z):
  норма     — z = v + n,                      правдоподобие N(z; v̂, S);
  боксование— z = v + n + δ, δ ≥ 0 (полунорм.) → колесо — верхняя граница скорости: v ≤ z (усечение);
  юз        — z = v + n − δ                   → нижняя граница: v ≥ z.
Правдоподобия боксования/юза — скошенное нормальное распределение. Вероятности перехода зависят от режима
управления: боксование вероятно при тяге, юз — при торможении (по времени, а не по числу сообщений).
"""
from __future__ import annotations

import math

import numpy as np

N_M = 3
NORMAL, SLIP, SLIDE = 0, 1, 2
_SQ2 = math.sqrt(2.0)


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / _SQ2))


class ImmUkf:
    def __init__(self, q_accel=0.2, q_v=0.02, t_enter_act=200.0, t_enter_other=5000.0, t_exit=1.5,
                 slip_frac=0.25, env_factor=10.0, s_floor=0.12, eps_out=0.01, v_max=25.0):
        self.q_accel, self.q_v = q_accel, q_v
        self.t_enter_act, self.t_enter_other, self.t_exit = t_enter_act, t_enter_other, t_exit
        self.slip_frac, self.env_factor = slip_frac, env_factor
        self.s_floor, self.eps_out, self.v_max = s_floor, eps_out, v_max
        n, kappa = 2, 1.0
        self.lam = kappa
        self.wm = np.r_[self.lam / (n + self.lam), np.full(2 * n, 1.0 / (2 * (n + self.lam)))]
        self.reset(0.0, 0.0, 1.0)

    def reset(self, v, a, pv):
        self.m = np.tile(np.array([v, a], float), (N_M, 1))
        self.P = np.tile(np.diag([pv, 0.3 ** 2]), (N_M, 1, 1))
        self.mu = np.array([0.98, 0.01, 0.01])

    # ------------------------------------------------------------------ прогноз
    def predict(self, h, step_fn):
        """step_fn(mode, v, a_f) -> (v', a_f') — один подшаг нелинейной модели."""
        for i in range(N_M):
            m, P = self.m[i], self.P[i]
            try:
                L = np.linalg.cholesky((2 + self.lam) * (P + 1e-9 * np.eye(2)))
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.maximum(np.diag(P), 1e-9) * (2 + self.lam)))
            pts = [m, m + L[:, 0], m + L[:, 1], m - L[:, 0], m - L[:, 1]]
            Y = np.array([step_fn(i, p[0], p[1]) for p in pts])
            mean = self.wm @ Y
            d = Y - mean
            cov = (d.T * self.wm) @ d
            cov += np.diag([self.q_v ** 2 * h, self.q_accel ** 2 * h])
            self.m[i], self.P[i] = mean, 0.5 * (cov + cov.T)

    # ------------------------------------------------------------------ коррекция
    def update(self, z, R, dt, u_sign, env=''):
        """Одно измерение скорости тележки. dt — время с прошлого измерения (для вероятностей перехода)."""
        self._mix(dt, u_sign)
        lik = np.zeros(N_M)
        for i in range(N_M):
            m0, P = self.m[i], self.P[i]
            S = max(P[0, 0], 1e-8) + R
            y = z - m0[0]
            if i == NORMAL:
                # нижняя граница неопределённости (невязки < ~0.35 м/с — не повод менять режим) и компонента
                # одиночных выбросов (равномерно на 0…v_max): выброс не переключает режим и не корректирует оценку
                Se = max(S, self.s_floor ** 2)
                ln = math.exp(-0.5 * y * y / Se) / math.sqrt(2 * math.pi * Se)
                lo = self.eps_out / self.v_max
                lik[i] = (1 - self.eps_out) * ln + lo
                r_out = lo / lik[i] if lik[i] > 0 else 1.0
                K = P[:, 0] / S * (1.0 - r_out)
                IKH = np.eye(2) - np.outer(K, [1.0, 0.0])
                self.m[i] = m0 + K * y
                self.P[i] = IKH @ P @ IKH.T + R * np.outer(K, K)
            else:
                sd = self.slip_frac * max(m0[0], 2.0)
                om = math.sqrt(S + sd * sd)
                al = sd / math.sqrt(S) * (1.0 if i == SLIP else -1.0)
                lik[i] = 2.0 / om * _phi(y / om) * _Phi(al * y / om)
                self._truncate(i, z, upper=(i == SLIP))
        if env == 'hi':
            lik[SLIP] *= self.env_factor
            lik[NORMAL] /= self.env_factor
        elif env == 'lo':
            lik[SLIDE] *= self.env_factor
            lik[NORMAL] /= self.env_factor
        post = self.mu * np.maximum(lik, 1e-300)
        tot = post.sum()
        self.mu = post / tot if tot > 0 and np.isfinite(tot) else np.array([0.98, 0.01, 0.01])
        self.mu = np.maximum(self.mu, 1e-6)
        self.mu /= self.mu.sum()
        return lik

    def _truncate(self, i, b, upper):
        """Одностороннее ограничение v ≤ b (upper) или v ≥ b; a_f обновляется по регрессии на v."""
        m, P = self.m[i], self.P[i]
        s = math.sqrt(max(P[0, 0], 1e-10))
        a = (b - m[0]) / s
        if upper:
            Z = max(_Phi(a), 1e-12)
            r = _phi(a) / Z
            ev = m[0] - s * r
            var = P[0, 0] * max(1.0 - a * r - r * r, 1e-4)
        else:
            Z = max(1.0 - _Phi(a), 1e-12)
            r = _phi(a) / Z
            ev = m[0] + s * r
            var = P[0, 0] * max(1.0 + a * r - r * r, 1e-4)
        g = P[1, 0] / max(P[0, 0], 1e-10)
        m_new = np.array([ev, m[1] + g * (ev - m[0])])
        P_new = np.array([[var, g * var], [g * var, P[1, 1] - g * P[1, 0] + g * g * var]])
        self.m[i], self.P[i] = m_new, P_new

    def _mix(self, dt, u_sign):
        """Смешивание IMM с вероятностями перехода за dt (зависят от режима управления)."""
        dt = min(max(dt, 1e-3), 2.0)
        p = lambda T: 1.0 - math.exp(-dt / T)  # noqa: E731
        e_slip = p(self.t_enter_act if u_sign > 0 else self.t_enter_other)
        e_slide = p(self.t_enter_act if u_sign < 0 else self.t_enter_other)
        x = p(self.t_exit)
        T = np.array([[1 - e_slip - e_slide, e_slip, e_slide],
                      [x, 1 - x - 1e-4, 1e-4],
                      [x, 1e-4, 1 - x - 1e-4]])
        c = self.mu @ T
        W = (T * self.mu[:, None]) / np.maximum(c[None, :], 1e-300)     # W[i, j] = P(было i | стало j)
        m_mix = W.T @ self.m
        P_mix = np.zeros_like(self.P)
        for j in range(N_M):
            for i in range(N_M):
                d = self.m[i] - m_mix[j]
                P_mix[j] += W[i, j] * (self.P[i] + np.outer(d, d))
        self.m, self.P, self.mu = m_mix, P_mix, c

    # ------------------------------------------------------------------ результат
    def combined(self):
        m = self.mu @ self.m
        P = np.zeros((2, 2))
        for i in range(N_M):
            d = self.m[i] - m
            P += self.mu[i] * (self.P[i] + np.outer(d, d))
        return m, P

    def set_speed(self, v, pv=None):
        """Принудительная скорость во всех режимах (ZUPT, ресинхронизация)."""
        for i in range(N_M):
            self.m[i, 0] = v
            if pv is not None:
                self.P[i] = np.diag([pv, self.P[i][1, 1]])

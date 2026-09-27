"""Оценщик скорости и положения трамвая (без ROS; используется нодой и офлайн-стендом).

Скорость: фильтр Калмана, состояние x = [v, b], где v — продольная скорость (м/с), b — медленное смещение
модельного ускорения. Прогноз: v̇ = a_f + b, где a_f — выход инерционного звена от a_model(u(t−τ), v, i(s)).
Коррекция: скорости тележек (км/ч → м/с) с отбраковкой недостоверных измерений (гейт, физическая огибающая
ускорения, залипание, скачки, расхождение тележек; откат и гистерезис эпизодов проскальзывания).
Положение: s = ∫v dt, затем s → точка на карте пути (pathgraph) с коррекцией на местах остановок, либо
прямолинейное счисление от точки выставки. GNSS используется только для начальной выставки в первые секунды.
Подробно — docs/MODEL.md.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geo import EnuFrame, MgrsFrame
from .imm import NORMAL, SLIDE, SLIP, ImmUkf
from .model import RELEASE, DriveModel
from .pathmap import PathMap, branch_fit

MODE_PRE, MODE_MAP, MODE_DR, MODE_REL = 'pre_init', 'map', 'dead_reckoning', 'relative'
MODE_APPROACH = 'approach'          # путь выезда на линию, затем карта


@dataclass
class Params:
    wheel_scale: float = 1.0 / 3.6        # (м/с) / (единица показаний колёс)
    distance_scale: float = 1.0            # поправка масштаба пути к скорости
    v_max: float = 25.0
    bogie_timeout: float = 0.5             # с, после — показание тележки считается устаревшим
    r_wheel: float = 0.05                  # СКО шума колеса, м/с
    q_accel: float = 0.2                   # СКО ошибки модельного ускорения, м/с² (спектральная плотность)
    q_bias: float = 0.05                   # случайное блуждание смещения, м/с²/√с
    bias_limit: float = 0.0                # смещение модели отключено: с ESN и без него оно ухудшало положение (стенд)
    bias_decay_s: float = 5.0              # затухание смещения при отсутствии колёс
    gate_sigma: float = 3.0
    gate_min: float = 0.35                 # м/с — меньшие невязки никогда не отбраковываются
    a_phys_max: float = 3.0                # м/с² — физически невозможная скорость изменения показаний
    slip_accel_margin: float = 0.5         # м/с² — выход ускорения колеса за физическую огибающую (боксование/юз)
    reaccept_s: float = 3.0                # мин. время отбраковки обеих тележек до ресинхронизации
    reaccept_max_s: float = 8.0            # ресинхронизация безусловно
    reaccept_quick_s: float = 1.0          # согласованные тележки без физического подтверждения проскальзывания
    slip_end_tol: float = 0.5              # м/с² — согласие ускорения колеса с моделью (конец проскальзывания)
    resync_dy: float = 0.3                 # м/с за 1 с — «стабильная» невязка (ошибка модели, не проскальзывание)
    frozen_n: int = 5                      # одинаковых показаний подряд в движении — залипание датчика
    noise_tau: float = 5.0                 # с, постоянная оценки шума колёс
    bogie_mismatch: float = 0.5            # м/с — расхождение тележек
    zero_speed: float = 0.02               # м/с
    substep: float = 0.05
    max_gap: float = 5.0                   # с, дольше — не интегрируем (разрыв записи)
    reset_back_jump: float = 10.0          # с, скачок времени назад → новый прогон (перезапуск bag), сброс
    stale_s: float = 0.3                   # с, сообщения старее состояния не обрабатываются фильтром
    # выставка
    init_window: float = 2.0               # с после первого фикса
    init_timeout: float = 15.0             # с после первого входа без GNSS → относительная одометрия
    # tf антенн в base_link (организаторы): base_link — ось вращения первой тележки на уровне рельса
    master_x: float = -9.873
    rover_x: float = 2.563
    antenna_z: float = 3.0
    grade_back: float = 9.873              # м: уклон берётся под антенной master (≈ середина вагона)
    antenna_base: float = 12.436           # м, rover впереди master по ходу (rover_x − master_x)
    base_tol: float = 1.5
    init_scatter_max: float = 0.2          # м: rover с разбросом больше (и > 3× master) для выставки не берётся
    map_match_dist: float = 6.0
    use_map: bool = True
    use_esn: bool = True                   # ESN в модели динамики (если есть config/esn.json)
    filter: str = 'kf'                     # 'kf' — фильтр Калмана с эвристической отбраковкой; 'imm' — IMM из 3 UKF
    imm_t_enter: float = 200.0             # с: характерное время до боксования при тяге / юза при торможении
    imm_t_exit: float = 1.5                # с: характерная длительность эпизода
    imm_slip_frac: float = 0.25            # масштаб величины проскальзывания (доля скорости)
    imm_s_floor: float = 0.0               # м/с: нижняя граница неопределённости режима «норма» для правдоподобия
    imm_eps_out: float = 0.01              # доля одиночных выбросов в режиме «норма»
    origin: str = 'mgrs'                   # 'mgrs' (как у судьи) | 'first_fix' (ENU от первого фикса) | 'fixed'
    utm_zone: int = 37
    mgrs_east0: float = 300000.0           # угол квадрата MGRS 37U CB: x = E − 300000, y = N − 6100000
    mgrs_north0: float = 6100000.0
    origin_lat: float = 55.804767
    origin_lon: float = 37.419565
    origin_alt: float = 150.0
    # начальная позиция без GNSS (в системе `origin`, по умолчанию MGRS): используется, если за init_timeout
    # не пришло ни одного фикса; NaN — не задана (тогда относительная одометрия от стартовой точки)
    init_x: float = float('nan')
    init_y: float = float('nan')
    init_z: float = float('nan')
    init_yaw: float = float('nan')         # рад, от оси x против часовой; NaN — по направлению ребра карты
    # физические параметры (пересчёт удельной силы привода в момент на валу двигателя; на оценку v, s не влияют)
    mass_kg: float = 42000.0               # масса вагона с загрузкой (по умолчанию ориентировочная, задаётся)
    wheel_radius_m: float = 0.33
    gear_ratio: float = 7.0
    drive_efficiency: float = 0.95
    n_motors: int = 1                      # 1 — суммарный момент всех двигателей
    rot_mass_factor: float = 0.26          # γ: по данным k_i = g/(1+γ) ≈ 7.8 → γ ≈ 0.26
    adhesion_mu: float = 0.3               # ограничение по сцеплению: |a_модели| ≤ μ·g
    noise_warn: float = 0.3                # м/с: шум колёс выше — диагностика «аномально высокая дисперсия»
    pos_sigma0: float = 0.5
    pos_sigma_per_m: float = 0.004         # относит. ошибка пути (разброс масштаба колёс по прогонам ≈0.25 %)
    # коррекция вдоль пути по местам остановок (карта)
    stop_correction: bool = True
    stop_confirm_s: float = 1.0            # с стоянки до коррекции
    stop_sigma: float = 1.5                # м, разброс места остановки
    stop_window_min: float = 10.0
    stop_window_max: float = 30.0          # ~длина вагона: стоянка в очереди за другим трамваем не притягивается
    sigma_model_var: bool = False          # учитывать в σ пути неопределённость участков без колёс
    stop_window_model_k: float = 0.0       # расширение окна после участков «только модель» (σ пути без колёс)
    scale_adapt: bool = True
    scale_alpha: float = 0.3
    scale_min_dist: float = 300.0
    scale_limit: float = 0.015
    # выбор ветки на стрелке по профилю скорости после неё (data/branch_profiles.json)
    fork_select: bool = True
    fork_min_m: float = 30.0               # м после стрелки: раньше не решаем (мало данных)
    fork_llr: float = 6.0                  # перевес логарифма правдоподобия другой ветки для переустановки
    fork_fit_max: float = 3.0              # и средний z² её профиля не больше (движение ей действительно соответствует)


@dataclass
class Output:
    stamp: float
    v: float
    a: float
    s: float
    e: float
    n: float
    u: float
    yaw: float
    mode: str
    var_v: float
    sigma_along: float
    sigma_cross: float
    slip: bool
    slip_kind: str
    slip_ratio: float
    wheels_ok: int
    sensor_fault: str = ''
    wheel_noise: float = 0.0
    pos_valid: bool = True
    a_drive: float = 0.0                   # удельная сила привода (тяга > 0 / торможение < 0) без сопротивлений, м/с²
    torque_nm: float = 0.0                 # момент на валу двигателя (по физ. параметрам), Н·м
    shaft_rpm: float = 0.0                 # частота вращения вала двигателя, об/мин
    map_edge: int = -1                     # ребро карты пути и координата вдоль него (представление положения по pathgraph)
    map_s: float = float('nan')
    lat: float = float('nan')
    lon: float = float('nan')
    h: float = float('nan')


@dataclass
class _Bogie:
    z: float = float('nan')
    t: float = -1e18
    t_acc: float = -1e18             # время последнего принятого измерения
    z_acc: float = float('nan')
    rej_since: float | None = None
    last_kind: str = ''
    last_ratio: float = 0.0
    same: int = 0
    y_hist: deque = field(default_factory=deque)
    acc_hist: deque = field(default_factory=deque)
    raw_hist: deque = field(default_factory=deque)
    cons_since: float | None = None
    strong: bool = False             # эпизод подтверждён физически (огибающая/расхождение тележек)


@dataclass
class _Init:
    t_first_input: float | None = None
    t_first_fix: float | None = None
    fixes: dict = field(default_factory=lambda: {'master': [], 'rover': []})


class Estimator:
    def __init__(self, params: Params, model: DriveModel, pathmap: PathMap | None = None, esn=None):
        self.p = params
        self.model = model
        self.map = pathmap if params.use_map else None
        self.esn = esn if params.use_esn else None       # обучаемая часть модели динамики (esn.py)
        self.imm = (ImmUkf(q_accel=params.q_accel, t_enter_act=params.imm_t_enter, t_exit=params.imm_t_exit,
                           slip_frac=params.imm_slip_frac, s_floor=params.imm_s_floor, eps_out=params.imm_eps_out)
                    if params.filter == 'imm' else None)
        self.reset()

    # ================================================================== состояние
    def reset(self):
        self.model.reset()
        if getattr(self, 'esn', None) is not None:
            self.esn.reset()
        self.esn_next = None
        if getattr(self, 'imm', None) is not None:
            self.imm.reset(0.0, 0.0, 1.0)
        self.imm_last = None
        self.imm_bad_since = None
        self.esn_buf: deque = deque(maxlen=12)
        self.base_lagc = np.zeros(3)
        self.base_glag = 0.0
        self.t: float | None = None
        self.x = np.zeros(2)
        self.P = np.diag([1.0, 0.1 ** 2])
        self.a_last = 0.0
        self.a_f = 0.0
        self.s = 0.0
        self.bog = {'front': _Bogie(), 'rear': _Bogie()}
        self.mode = MODE_PRE
        self.frame: EnuFrame | None = None
        self.init = _Init()
        self.gnss_done = False
        self.prov = None                # (e, n, u, курс, s) — положение до завершения выставки
        self.prov_pair = False
        self.anchor = None              # map: [eid, s_edge, overflow, s_total]; dr: (e0, n0, u0, psi, s_total)
        self.s_at_fix: list = []        # (stamp, s) для компенсации пути, пройденного после фикса
        self.slip_until = -1e18
        self.slip_kind = ''
        self.fault_until = -1e18
        self.fault_kind = ''
        self.noise_var = 0.0
        self.scale_adj = 1.0
        self.var_s_model = 0.0             # дисперсия пути от неопределённости скорости (растёт без колёс)
        self.zbuf: deque = deque()
        self.v_init = False
        self.v_hist: deque = deque()
        self.stop_t0: float | None = None
        self.stop_done = False
        self.sig_s0 = self.p.pos_sigma0
        self.s_corr = 0.0
        self.n_stop_corr = 0
        self.fork = None                   # активная стрелка: {'br': профили веток, 's0': путь в узле, 'cur': ветка, 'k': бин}
        self.sv: deque = deque(maxlen=400)  # (путь, скорость) с шагом ≥ 0.5 м — для сравнения с профилями веток
        self.n_fork_switch = 0
        self.last_corr = 0.0
        self.n_rejected = 0
        self.n_accepted = 0
        self.n_stale = 0
        self._stale_out = False

    # ================================================================== вход
    def _time(self, stamp: float) -> bool:
        """Обработка времени входа; False — сообщение устарело и игнорируется."""
        if self.init.t_first_input is None:
            self.init.t_first_input = stamp
        if self.t is None:
            self.t = stamp
            return True
        if stamp < self.t - self.p.reset_back_jump:
            self.reset()
            self.init.t_first_input = stamp
            self.t = stamp
            return True
        if stamp < self.t - self.p.stale_s:
            # в начале записи bag топики приходят пачкой со «старыми» stamp (очередь рекордера):
            # фильтр их не использует, но выход с этим stamp публикуется (текущая оценка)
            self.n_stale += 1
            self._stale_out = True
            return False
        self._propagate(stamp)
        return True

    def on_cmd(self, stamp: float, position: int) -> Output | None:
        if not math.isfinite(stamp):
            return None
        pos = int(position) if -15 <= int(position) <= 15 else 0
        self.model.push_cmd(stamp, pos)
        self._stale_out = False
        if not self._time(stamp):
            return self.output(stamp) if self._stale_out else None
        self._check_init(stamp)
        self._stop_logic(stamp)
        return self.output(stamp)

    def on_wheel(self, side: str, stamp: float, value: float) -> Output | None:
        if not math.isfinite(stamp):
            return None
        self._stale_out = False
        if not self._time(stamp):
            return self.output(stamp) if self._stale_out else None
        self._wheel_update(side, stamp, value)
        self._check_init(stamp)
        self._stop_logic(stamp)
        return self.output(stamp)

    def on_fix(self, antenna: str, stamp: float, lat: float, lon: float, alt: float, status: int):
        """GNSS принимается только до завершения выставки."""
        if self.gnss_done or antenna not in ('master', 'rover'):
            return
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)) or status < 0:
            return
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return
        if self.init.t_first_fix is None:
            self.init.t_first_fix = stamp
        self.init.fixes[antenna].append((stamp, lat, lon, alt, int(status)))
        self.s_at_fix.append((stamp, self.s))
        self._provisional(antenna)

    # ================================================================== динамика
    def _grade(self) -> float:
        """Уклон под серединой вагона: точка на grade_back м позади base_link (там же, где он калибровался)."""
        return self._map_at_back()[0]

    def _curv(self) -> float:
        """Кривизна пути под серединой вагона (для сопротивления на кривых)."""
        return self._map_at_back()[1]

    def _map_at_back(self):
        if self.mode == MODE_MAP and self.anchor is not None and self.anchor[2] <= 0.0:
            eid, s_edge, _ = self.map.advance(self.anchor[0], self.anchor[1], -self.p.grade_back)
            ed = self.map.edges[eid]
            i = min(int(s_edge / self.map.step), len(ed.grade) - 1)
            return float(ed.grade[i]), float(ed.curv[i])
        return 0.0, 0.0

    def _propagate(self, t: float):
        dt = t - self.t
        if dt <= 0.0:
            return
        if dt > self.p.max_gap:
            self.t = t
            return
        wheels_fresh = any(t - b.t_acc < self.p.bogie_timeout for b in self.bog.values())
        grade, curv = self._map_at_back()
        n = max(1, int(math.ceil(dt / self.p.substep)))
        h = dt / n
        v, b = self.x
        P = self.P
        if self.imm is not None:
            self._propagate_imm(t, n, h, wheels_fresh, grade, curv)
            return
        trusted = wheels_fresh and t >= self.slip_until and t >= self.fault_until
        for k in range(n):
            tk = self.t + (k + 1) * h
            u = self.model.u_at(tk)
            if self.esn is not None:
                if self.esn_next is None:
                    self.esn_next = tk
                while tk >= self.esn_next:          # резервуар ESN — на сетке 10 Гц
                    self._esn_grid(self.esn_next, u, v, grade, trusted, curv)
                    self.esn_next += self.esn_dt
                tv = self.model.accel(u, v, 0.0)
                us = -8 if u == RELEASE else u
                a_ss = self.esn.steady_accel(tv * (us > 0), tv * (us < 0), tv * (us == 0), -self.model.k_grade * grade,
                                             active=not trusted) - self.model.k_curve * abs(curv)
                tau = self.esn.lag_tau
            else:
                a_ss = self.model.accel(u, v, grade, curv)
                tau = self.model.lag_tau
            lim = self.p.adhesion_mu * 9.80665
            a_ss = min(max(a_ss, -lim), lim)           # ограничение по сцеплению колеса с рельсом
            self.a_f = self.a_f + (a_ss - self.a_f) * (1.0 - math.exp(-h / tau)) if tau > 0 else a_ss
            if v <= 0.0 and self.a_f < 0.0:
                self.a_f = 0.0                  # стоящий трамвай тормоза не разгоняют назад
            a = self.a_f + b + (self.esn.additive if (self.esn is not None and not trusted) else 0.0)
            v_new = v + a * h
            if v_new < 0.0:
                v_new = 0.0
                a = -v / h
            self.s += 0.5 * (v + v_new) * h * self.p.distance_scale
            v = v_new
            if not wheels_fresh:
                b *= math.exp(-h / self.p.bias_decay_s)
            F = np.array([[1.0, h], [0.0, 1.0]])
            Q = np.diag([self.p.q_accel ** 2 * h, self.p.q_bias ** 2 * h])
            P = F @ P @ F.T + Q
            if not wheels_fresh:        # только модель: ошибка скорости коррелирована во времени
                self.var_s_model = (math.sqrt(self.var_s_model) + math.sqrt(max(P[0, 0], 0.0)) * h) ** 2
        self.x = np.array([v, b])
        self.P = P
        self.a_last = a
        self.t = t
        if self.mode == MODE_MAP:
            self._advance_map()
        elif self.mode == MODE_APPROACH:
            k, s0, sref = self.approach_state
            st = s0 + (self.s - sref)
            tl = len(self.map.approach[k]['e']) - 1.0
            if st >= tl:
                t = self.map.approach[k]
                eid, s_edge, over = self.map.advance(t['edge'], t['s'], st - tl)
                self.anchor = [eid, s_edge, over, self.s]
                self.mode = MODE_MAP

    def _accel_model(self, u, v, grade, active, curv=0.0):
        """Установившееся ускорение модели и постоянная звена; active — поправка ESN (колёсам не верим)."""
        if self.esn is not None:
            tv = self.model.accel(u, v, 0.0)
            us = -8 if u == RELEASE else u
            return (self.esn.steady_accel(tv * (us > 0), tv * (us < 0), tv * (us == 0), -self.model.k_grade * grade,
                                          active=active) - self.model.k_curve * abs(curv),
                    self.esn.lag_tau, self.esn.additive if active else 0.0)
        return self.model.accel(u, v, grade, curv), self.model.lag_tau, 0.0

    def _propagate_imm(self, t, n, h, wheels_fresh, grade, curv=0.0):
        imm = self.imm
        v = float(self.x[0])
        m, P = imm.combined()
        for k in range(n):
            tk = self.t + (k + 1) * h
            u = self.model.u_at(tk)
            trusted = wheels_fresh and imm.mu[NORMAL] > 0.5
            if self.esn is not None:
                if self.esn_next is None:
                    self.esn_next = tk
                while tk >= self.esn_next:
                    self._esn_grid(self.esn_next, u, v, grade, trusted, curv)
                    self.esn_next += self.esn_dt
            al_cache = {}

            def step(mode, vv, aa, u=u):
                active = (mode != NORMAL) or not wheels_fresh
                a_ss, tau, add = self._accel_model(u, max(vv, 0.0), grade, active, curv)
                lim = self.p.adhesion_mu * 9.80665
                a_ss = min(max(a_ss, -lim), lim)       # ограничение по сцеплению
                al = al_cache.get(tau)
                if al is None:
                    al = al_cache[tau] = (1.0 - math.exp(-h / tau)) if tau > 0 else 1.0
                aa = aa + (a_ss - aa) * al
                if vv <= 0.0 and aa < 0.0:
                    aa = 0.0
                return max(vv + (aa + add) * h, 0.0), aa

            imm.predict(h, step)
            m, P = imm.combined()
            v_new = max(float(m[0]), 0.0)
            self.s += 0.5 * (v + v_new) * h * self.p.distance_scale
            v = v_new
            if not wheels_fresh:
                self.var_s_model = (math.sqrt(self.var_s_model) + math.sqrt(max(P[0, 0], 0.0)) * h) ** 2
        self.a_last = (v - float(self.x[0])) / max(t - self.t, 1e-6)
        self.x = np.array([v, 0.0])
        self.P = np.diag([max(P[0, 0], 1e-8), self.P[1, 1]])
        self.a_f = float(m[1])
        self.t = t
        if self.mode == MODE_MAP:
            self._advance_map()
        elif self.mode == MODE_APPROACH:
            k, s0, sref = self.approach_state
            st = s0 + (self.s - sref)
            tl = len(self.map.approach[k]['e']) - 1.0
            if st >= tl:
                tr = self.map.approach[k]
                eid, s_edge, over = self.map.advance(tr['edge'], tr['s'], st - tl)
                self.anchor = [eid, s_edge, over, self.s]
                self.mode = MODE_MAP

    def _imm_wheel(self, bg, other, stamp, z, z_prev, t_prev, R, u):
        """Показание тележки в режиме IMM: предфильтры (скачок, расхождение тележек), затем IMM."""
        p, imm = self.p, self.imm
        v = float(self.x[0])
        y = z - v
        S = max(self.P[0, 0], 0.0) + R
        gate = max(p.gate_sigma * math.sqrt(S), p.gate_min)
        env = ''
        if R < 0.02 and bg.raw_hist and 0.25 < stamp - bg.raw_hist[0][0] < 1.0:
            t_a, z_a = bg.raw_hist[0]
            a_w = (z - z_a) / (stamp - t_a)
            a_lo, a_hi = self.model.envelope(v, *self._map_at_back())
            env = 'hi' if a_w > a_hi + p.slip_accel_margin else ('lo' if a_w < a_lo - p.slip_accel_margin else '')
        dt_raw = stamp - t_prev
        if 0.0 < dt_raw < 0.5 and math.isfinite(z_prev) and abs(z - z_prev) > p.a_phys_max * dt_raw + gate:
            self.fault_until, self.fault_kind = stamp + 0.5, 'spike'
            self.n_rejected += 1
            return
        bg.raw_hist.append((stamp, z))
        while len(bg.raw_hist) > 1 and bg.raw_hist[0][0] < stamp - 0.45:
            bg.raw_hist.popleft()
        bg.last_ratio = y / max(v, 1.0)
        if (stamp - other.t_acc < p.bogie_timeout and math.isfinite(other.z_acc)
                and abs(z - other.z_acc) > max(p.bogie_mismatch, 4.0 * math.sqrt(2 * R))
                and abs(y) > abs(other.z_acc - v)):
            self.slip_until, self.slip_kind = stamp + 0.5, ('slip' if z > other.z_acc else 'slide')
            self.n_rejected += 1
            return                           # одна тележка проскальзывает — берём другую
        dt = 0.1 if self.imm_last is None else stamp - self.imm_last
        self.imm_last = stamp
        us = 1 if u > 0 else (-1 if u < 0 else 0)
        imm.update(z, R, dt, us, env)
        # ресинхронизация: колесо быстрее модели без тяги / медленнее без торможения (не проскальзывание),
        # тележки согласованы, ускорение в огибающей — ошибка модели или команды; колёсам верим
        impossible = (y > 0 and u <= 0) or (y < 0 and u >= 0)
        agree = (math.isfinite(other.z) and abs(other.t - stamp) < 0.15
                 and abs(z - other.z) < max(0.3, 4.0 * math.sqrt(2 * R)))
        if imm.mu[NORMAL] < 0.05 and impossible and agree and not env and abs(y) > gate:
            if self.imm_bad_since is None:
                self.imm_bad_since = stamp
            elif stamp - self.imm_bad_since > p.reaccept_quick_s:
                imm.reset(z, self.a_f, R)
                self.imm_bad_since = None
        else:
            self.imm_bad_since = None
        mu = imm.mu
        if mu[SLIP] > 0.5 or mu[SLIDE] > 0.5:
            self.slip_until = stamp + 0.5
            self.slip_kind = 'slip' if mu[SLIP] >= mu[SLIDE] else 'slide'
        if mu[NORMAL] > 0.5:
            self.n_accepted += 1
            bg.t_acc, bg.z_acc = stamp, z
        else:
            self.n_rejected += 1
        # стоянка (ZUPT)
        self.zbuf.append((stamp, z))
        while self.zbuf and self.zbuf[0][0] < stamp - 1.0:
            self.zbuf.popleft()
        m, P = imm.combined()
        zth = min(3.0 * math.sqrt(R), 0.5)
        if u <= 0 and mu[NORMAL] > 0.5:
            still = z < max(zth, p.zero_speed) and m[0] < max(zth, 0.1)
            if not still and len(self.zbuf) >= 8:
                zm = sum(q[1] for q in self.zbuf) / len(self.zbuf)
                still = zm < 2.0 * math.sqrt(R / len(self.zbuf)) + 0.05 and m[0] < zth + 0.1
            if still:
                imm.set_speed(0.0)
                m, P = imm.combined()
        self.x = np.array([max(float(m[0]), 0.0), 0.0])
        self.P = np.diag([max(P[0, 0], 1e-8), self.P[1, 1]])
        self.a_f = float(m[1])

    esn_dt = 0.1

    def _esn_grid(self, tg: float, u: int, v: float, grade: float, trusted: bool, curv: float = 0.0):
        """Шаг ESN: базовая физика по компонентам (для измеренного остатка и контекста), затем резервуар."""
        tv = self.model.accel(u, v, 0.0)
        us = -8 if u == RELEASE else u
        comps = np.array([tv * (us > 0), tv * (us < 0), tv * (us == 0)])
        al = 1.0 - math.exp(-self.esn_dt / self.esn.base_tau)
        self.base_lagc += al * (comps - self.base_lagc)
        self.base_glag += al * (-self.model.k_grade * grade - self.model.k_curve * abs(curv) - self.base_glag)
        vm = None
        if trusted:
            zs = [b.z_acc for b in self.bog.values() if tg - b.t_acc < 0.15 and math.isfinite(b.z_acc)]
            vm = sum(zs) / len(zs) if zs else None
        self.esn_buf.append((self.base_lagc.copy(), self.base_glag, float(self.base_lagc.sum() + self.base_glag), vm, v))
        r_in, lag_comps = 0.0, None
        L = self.esn.lag_r
        if trusted and len(self.esn_buf) >= 2 * L + 1:
            new, old, mid = self.esn_buf[-1], self.esn_buf[-1 - 2 * L], self.esn_buf[-1 - L]
            if new[3] is not None and old[3] is not None and mid[4] > 0.3:
                a_c = (new[3] - old[3]) / (2 * L * self.esn_dt)       # центральная разность ±0.5 с
                r_in = a_c - mid[2]
                lag_comps = (mid[0][0], mid[0][1], mid[0][2], mid[1], a_c)
        self.esn.grid_step(u, v, self.a_f, grade, trusted, r_in, lag_comps)

    def _wheel_update(self, side: str, stamp: float, value: float):
        p = self.p
        bg = self.bog.get(side)
        if bg is None:
            return
        z = value * p.wheel_scale * self.scale_adj if value is not None and math.isfinite(value) else float('nan')
        # залипание: в реальных данных в движении не больше 4 одинаковых показаний подряд
        bg.same = bg.same + 1 if (z == bg.z and z > 1.0) else 0
        z_prev, t_prev = bg.z, bg.t
        bg.z, bg.t = z, stamp
        other = self.bog['rear' if side == 'front' else 'front']
        # адаптивный шум измерения: половина дисперсии разности тележек (в одной метке времени)
        if math.isfinite(z) and math.isfinite(other.z) and abs(other.t - stamp) < 0.05 and other.same < p.frozen_n:
            d2 = min((z - other.z) ** 2, 1.0)
            w = min(1.0, 0.1 / p.noise_tau)
            self.noise_var += w * (0.5 * d2 - self.noise_var)
        R = max(p.r_wheel ** 2, self.noise_var)
        u_eff = self.model.u_at(stamp)
        u = -8 if u_eff == RELEASE else u_eff        # знак режима: тяга > 0, торможение < 0
        v = self.x[0]
        kind = ''
        y = 0.0
        if not math.isfinite(z) or z < -0.5 or z > p.v_max:
            kind = 'invalid'
        elif bg.same >= p.frozen_n:
            kind = 'frozen'
        elif not self.v_init:
            # первое достоверное показание: нода могла стартовать на ходу — инициализация скорости
            z = max(z, 0.0)
            self.x[0] = z
            self.P = np.diag([R, self.P[1, 1]])
            self.v_init = True
            if self.imm is not None:
                self.imm.reset(z, 0.0, R)
        elif self.imm is not None:
            self._imm_wheel(bg, other, stamp, max(z, 0.0), z_prev, t_prev, R, u)
            return
        else:
            z = max(z, 0.0)
            y = z - v
            S = max(self.P[0, 0], 0.0) + R
            gate = max(p.gate_sigma * math.sqrt(S), p.gate_min)
            # ускорение колеса (на ~0.4 с) против физической огибающей (любая позиция контроллера)
            env = ''
            if R < 0.02 and bg.raw_hist and 0.25 < stamp - bg.raw_hist[0][0] < 1.0:
                t_a, z_a = bg.raw_hist[0]
                a_w = (z - z_a) / (stamp - t_a)
                a_lo, a_hi = self.model.envelope(v, *self._map_at_back())
                env = 'hi' if a_w > a_hi + p.slip_accel_margin else ('lo' if a_w < a_lo - p.slip_accel_margin else '')
                consistent = abs(a_w - (self.a_f + self.x[1])) < p.slip_end_tol
                if not consistent:
                    bg.cons_since = None
                elif bg.cons_since is None:
                    bg.cons_since = stamp
            dt_raw = stamp - t_prev
            if 0.0 < dt_raw < 0.5 and math.isfinite(z_prev) and abs(z - z_prev) > p.a_phys_max * dt_raw + gate:
                kind = 'spike'                       # одиночный скачок относительно предыдущего показания
            elif abs(y) > gate:
                kind = 'slip' if (y > 0 and u > 0) else ('slide' if (y < 0 and u < 0) else 'outlier')
                if (env == 'hi' and y > 0) or (env == 'lo' and y < 0):
                    self._rollback(stamp, bg)
                elif env:
                    bg.strong = True                 # восстановление после проскальзывания — тоже эпизод
            elif env == 'hi' and y > 0.05:
                # медленно нарастающее проскальзывание проходит гейт, но физически недостижимо
                kind = 'slip' if u > 0 else 'outlier'
                self._rollback(stamp, bg)
            elif env == 'lo' and y < -0.05:
                kind = 'slide' if u < 0 else 'outlier'
                self._rollback(stamp, bg)
            # гистерезис (только для физически подтверждённых эпизодов): после проскальзывания колесо
            # принимается, лишь когда его ускорение 0.5 с согласуется с моделью — иначе примется «хвост»
            if (not kind and bg.strong and bg.rej_since is not None
                    and (bg.cons_since is None or stamp - bg.cons_since < 0.5)
                    and stamp - bg.rej_since < p.reaccept_max_s):
                kind = bg.last_kind if bg.last_kind in ('slip', 'slide') else 'outlier'
            # расхождение тележек: недостоверна та, что дальше от прогноза
            if (not kind and stamp - other.t_acc < p.bogie_timeout and math.isfinite(other.z_acc)
                    and abs(z - other.z_acc) > max(p.bogie_mismatch, 4.0 * math.sqrt(2 * R))
                    and abs(y) > abs(other.z_acc - v)):
                kind = 'slip' if z > other.z_acc else 'slide'
                self._rollback(stamp, bg)
            bg.last_ratio = y / max(v, 1.0)
            if kind != 'spike':
                bg.raw_hist.append((stamp, z))
                while len(bg.raw_hist) > 1 and bg.raw_hist[0][0] < stamp - 0.45:
                    bg.raw_hist.popleft()

        if kind:
            self.n_rejected += 1
            bg.last_kind = kind
            if bg.rej_since is None:
                bg.rej_since = stamp
                bg.y_hist.clear()
            bg.y_hist.append((stamp, y))
            while len(bg.y_hist) > 2 and bg.y_hist[0][0] < stamp - 1.0:
                bg.y_hist.popleft()
            if kind in ('slip', 'slide'):
                self.slip_until = stamp + 0.5
                self.slip_kind = kind
            if kind in ('invalid', 'frozen', 'spike'):
                self.fault_until = stamp + 0.5
                self.fault_kind = kind
                return
            other_ok = stamp - other.t_acc < p.bogie_timeout and other.rej_since is None
            dur = stamp - bg.rej_since
            # невязка стабильна (не меняется за последнюю секунду) — это ошибка модели, а не проскальзывание
            stable = bg.y_hist[-1][0] - bg.y_hist[0][0] > 0.8 and abs(bg.y_hist[-1][1] - bg.y_hist[0][1]) < p.resync_dy
            # тележки согласованы между собой, ускорение колёс физически достижимо (эпизод не подтверждён
            # огибающей) — вероятнее ошибка модели/команды контроллера, чем проскальзывание: быстрая ресинхронизация
            agree = (math.isfinite(other.z) and abs(other.t - stamp) < 0.15
                     and abs(z - other.z) < max(0.3, 4.0 * math.sqrt(2 * R)))
            # «outlier» = колесо быстрее модели без тяги или медленнее без торможения: проскальзыванием это
            # быть не может физически (нет момента, раскручивающего/блокирующего колесо)
            # второй тележки нет вовсе (отказ датчика) — сверить не с чем; без этого одна оставшаяся тележка
            # отбраковывается секундами, пока модель уводит скорость (front_lost: до 60 м пути)
            other_absent = not math.isfinite(other.z) or stamp - other.t > p.bogie_timeout
            quick = kind == 'outlier' and not bg.strong and (agree or other_absent) and dur > p.reaccept_quick_s
            if other_ok or not ((dur > p.reaccept_s and stable) or quick or dur > p.reaccept_max_s):
                return
            # обе тележки долго и стабильно расходятся с моделью — ресинхронизация по колёсам
            if y ** 2 > self.P[0, 0]:
                self.P = np.diag([y ** 2, self.P[1, 1]])
        else:
            bg.rej_since = None
            bg.last_kind = ''
            bg.strong = False
        self.n_accepted += 1
        H = np.array([1.0, 0.0])
        S = self.P[0, 0] + R
        K = self.P[:, 0] / S
        if stamp < self.slip_until:
            K[1] = 0.0                      # смещение модели не подстраиваем по подозрительным данным
        self.x = self.x + K * (z - self.x[0])
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + R * np.outer(K, K)       # форма Джозефа — P остаётся SPD
        self.P = 0.5 * (self.P + self.P.T)
        self.x[1] = float(np.clip(self.x[1], -p.bias_limit, p.bias_limit))
        # стоянка (ZUPT): показания за 1 с в среднем в пределах шума, тяги нет — скорость обнуляется
        self.zbuf.append((stamp, z))
        while self.zbuf and self.zbuf[0][0] < stamp - 1.0:
            self.zbuf.popleft()
        zth = min(3.0 * math.sqrt(R), 0.5)
        if u <= 0:
            if z < max(zth, p.zero_speed) and self.x[0] < max(zth, 0.1):
                self.x[0] = 0.0
            elif len(self.zbuf) >= 8:
                zm = sum(q[1] for q in self.zbuf) / len(self.zbuf)
                if zm < 2.0 * math.sqrt(R / len(self.zbuf)) + 0.05 and self.x[0] < zth + 0.1:
                    self.x[0] = 0.0
        self.x[0] = max(self.x[0], 0.0)
        bg.t_acc, bg.z_acc = stamp, z
        self.v_hist.append((stamp, float(self.x[0])))
        while len(self.v_hist) > 2 and self.v_hist[0][0] < stamp - 1.0:
            self.v_hist.popleft()
        bg.acc_hist.append((stamp, z))
        while len(bg.acc_hist) > 1 and bg.acc_hist[0][0] < stamp - 0.45:
            bg.acc_hist.popleft()

    def _rollback(self, stamp: float, bg: _Bogie):
        """Эпизод проскальзывания подтверждён физически: первые его показания уже прошли гейт и сместили
        оценку. Скорость восстанавливается по оценке 0.35–0.6 с назад и модельному ускорению."""
        first = not bg.strong
        bg.strong = True
        if not first or not self.v_hist:
            return
        old = [q for q in self.v_hist if stamp - 0.6 <= q[0] <= stamp - 0.35]
        if not old:
            return
        t0, v0 = old[0]
        v_rb = max(v0 + (self.a_f + self.x[1]) * (stamp - t0), 0.0)
        if abs(v_rb - self.x[0]) > 0.1:
            self.x[0] = v_rb

    # ================================================================== выставка
    def _check_init(self, stamp: float):
        if self.gnss_done:
            return
        ini = self.init
        if ini.t_first_fix is None:
            if stamp - ini.t_first_input > self.p.init_timeout:
                self.gnss_done = True
                if math.isfinite(self.p.init_x) and math.isfinite(self.p.init_y):
                    self._manual_init()
                else:
                    self.mode = MODE_REL
            return
        allf = ini.fixes['master'] + ini.fixes['rover']
        span = max(f[0] for f in allf) - ini.t_first_fix
        moving = self.x[0] > 1.0
        # по меткам самих фиксов (а не колёс): в начале bag топики сдвинуты друг относительно друга
        if ((span >= self.p.init_window and len(allf) >= 6) or (moving and len(allf) >= 2)
                or stamp - ini.t_first_fix > self.p.init_window + 5.0):
            self._finalize_init()

    def _manual_init(self):
        """Выставка по заданной начальной позиции (последняя известная до отказа навигации / стартовая точка)."""
        p = self.p
        self._make_frame()
        yaw = p.init_yaw if math.isfinite(p.init_yaw) else None
        m = None
        if self.map is not None:
            self.map.set_frame(self.frame)
            m = self.map.match(p.init_x, p.init_y, yaw, 4.0 * p.map_match_dist)
        if m is not None:
            eid, s_edge, _ = m
            self.anchor = [eid, s_edge, 0.0, self.s]
            self.mode = MODE_MAP
        else:
            z0 = p.init_z if math.isfinite(p.init_z) else 0.0
            self.anchor = (p.init_x, p.init_y, z0, yaw if yaw is not None else 0.0, self.s)
            self.mode = MODE_DR
        self.s_init = self.s_corr = self.s
        self.var_s_model = 0.0

    def _make_frame(self):
        p = self.p
        if p.origin == 'mgrs':
            self.frame = MgrsFrame(p.utm_zone, p.mgrs_east0, p.mgrs_north0)
        elif p.origin == 'fixed':
            self.frame = EnuFrame(p.origin_lat, p.origin_lon, p.origin_alt)
        else:
            fm, fr = self.init.fixes['master'], self.init.fixes['rover']
            f0 = fm[0] if fm else fr[0]
            self.frame = EnuFrame(f0[1], f0[2], f0[3])

    def _provisional(self, antenna: str):
        """Положение base_link до завершения выставки — по последней паре антенн (или одной антенне)."""
        if self.frame is None:
            self._make_frame()
        p = self.p
        last = {a: (self.init.fixes[a][-1] if self.init.fixes[a] else None) for a in ('master', 'rover')}
        pos = {a: (f[0], *self.frame.to_enu(f[1], f[2], f[3])) for a, f in last.items() if f is not None}
        m, r = pos.get('master'), pos.get('rover')
        if (m is not None and r is not None and abs(m[0] - r[0]) < 0.06
                and abs(math.hypot(r[1] - m[1], r[2] - m[2]) - p.antenna_base) < p.base_tol):
            k = -p.master_x / p.antenna_base
            self.prov = (float(m[1] + k * (r[1] - m[1])), float(m[2] + k * (r[2] - m[2])),
                         float(m[3] + k * (r[3] - m[3]) - p.antenna_z),
                         math.atan2(r[2] - m[2], r[1] - m[1]), self.s)
        elif self.prov is None or not self.prov_pair:
            a = pos[antenna]
            self.prov = (float(a[1]), float(a[2]), float(a[3]) - p.antenna_z, 0.0, self.s)
            return
        self.prov_pair = True

    def _finalize_init(self):
        p = self.p
        self.gnss_done = True
        fm, fr = self.init.fixes['master'], self.init.fixes['rover']
        if self.frame is None:
            self._make_frame()
        fr_ = self.frame

        def enu(lst):
            if not lst:
                return np.zeros((0, 5))
            a = np.array(lst, float)
            e, n, u = fr_.to_enu(a[:, 1], a[:, 2], a[:, 3])
            return np.c_[a[:, 0], e, n, u, a[:, 4]]

        M, Rv = enu(fm), enu(fr)
        # курс: вектор master → rover по парам с близкими метками
        psi = None
        if len(M) and len(Rv):
            j = np.searchsorted(Rv[:, 0], M[:, 0])
            angs, good = [], []
            for i in range(len(M)):
                for jj in (j[i] - 1, j[i]):
                    if 0 <= jj < len(Rv) and abs(Rv[jj, 0] - M[i, 0]) < 0.06:
                        d = Rv[jj, 1:3] - M[i, 1:3]
                        if abs(np.hypot(*d) - p.antenna_base) < p.base_tol:
                            angs.append(math.atan2(d[1], d[0]))
                            good.append(M[i, 4] == 2 and Rv[jj, 4] == 2)
                        break
            if angs:
                a = np.array(angs)
                g = np.array(good)
                if g.sum() >= 3:
                    a = a[g]
                psi = math.atan2(np.median(np.sin(a)), np.median(np.cos(a)))
        # положение антенны master (по последним точкам окна)
        s_now = self.s

        def pick(A):
            if not len(A):
                return None
            A2 = A[A[:, 4] == 2]
            A = A2 if len(A2) >= 3 else A
            moving = self.x[0] > 0.3
            B = A[-1:] if moving else A
            return B[:, 0].max(), np.median(B[:, 1]), np.median(B[:, 2]), np.median(B[:, 3]), bool(A[0, 4] == 2)

        def scatter(A):                          # разброс фиксов антенны на стоянке (многолучёвость и т. п.)
            if len(A) < 3 or self.x[0] > 0.3:
                return 0.0
            return float(math.hypot(np.std(A[:, 1]), np.std(A[:, 2])))

        pm, pr = pick(M), pick(Rv)
        if pm is None and pr is None:
            self.mode = MODE_REL
            return
        # антенна для привязки: решение status=2 предпочтительнее; у 30639 master в начале часто без
        # решения (смещён на метры), а rover — с решением. Обе антенны едут по тем же рельсам.
        # Но статус не гарантирует точность: если rover на стоянке «гуляет» заметно сильнее master
        # (bag стенда 88aea4d9: разброс 0.87 м против 0.02, база 11.0 м вместо 12.44) — берём master.
        use_rover = pr is not None and pr[4] and (pm is None or not pm[4])
        if use_rover and pm is not None and scatter(Rv) > max(p.init_scatter_max, 3.0 * scatter(M)):
            use_rover = False
        t_f, e0, n0, u0 = (pr if use_rover else pm)[:4]
        # выход — base_link (ось первой тележки, уровень рельса): смещение от антенны вдоль пути вперёд, м
        fwd = -(p.rover_x if use_rover else p.master_x)
        u0 -= p.antenna_z
        if psi is None and not use_rover and len(M) >= 2 and np.hypot(M[-1, 1] - M[0, 1], M[-1, 2] - M[0, 2]) > 3.0:
            psi = math.atan2(M[-1, 2] - M[0, 2], M[-1, 1] - M[0, 1])
        if psi is not None:                     # курс известен — сразу точка base_link
            e0, n0 = e0 + fwd * math.cos(psi), n0 + fwd * math.sin(psi)
            fwd = 0.0
        # путь, пройденный после использованного фикса
        s_f = s_now
        for ts, ss in self.s_at_fix:
            if ts <= t_f:
                s_f = ss
        ds = s_now - s_f
        self.s_at_fix = []
        m = None
        if self.map is not None:
            self.map.set_frame(self.frame)
            m = self.map.match(e0, n0, psi, p.map_match_dist)
            ap = self.map.match_approach(e0, n0, psi, p.map_match_dist)
            if ap is not None and (m is None or ap[2] < m[2]):
                # старт вне линии (разворотное кольцо, отстойный путь) — сначала по известному пути выезда
                k, s_tr, _ = ap
                self.approach_state = (k, s_tr + ds + fwd, s_now)
                self.mode = MODE_APPROACH
                self.s_init = s_now
                self.s_corr = s_now
                return
            if m is None and psi is not None:
                # GNSS прогона бывает смещён на 10–20 м параллельно пути — ищем путь с тем же курсом шире
                m = self.map.match(e0, n0, psi, 4.0 * p.map_match_dist)
            if m is None and psi is None:
                # курс неизвестен: ближайший путь (рёбра однонаправленные — направление задаёт ребро)
                m = self.map.match(e0, n0, None, 2.5 * p.map_match_dist)
        if m is not None:
            eid, s_edge, _ = m
            eid, s_edge, over = self.map.advance(eid, s_edge, ds + fwd)
            self.anchor = [eid, s_edge, over, s_now]
            self.mode = MODE_MAP
        elif psi is not None:
            self.anchor = (e0 + ds * math.cos(psi), n0 + ds * math.sin(psi), u0, psi, s_now)
            self.mode = MODE_DR
        else:
            # ни карты, ни курса: прямолинейное счисление с неизвестным курсом бессмысленно —
            # относительная одометрия от стартовой точки (x = пройденный путь)
            self.frame = None
            self.mode = MODE_REL
        self.s_init = s_now
        self.s_corr = s_now
        self.var_s_model = 0.0

    def _advance_map(self):
        eid, s_edge, over, s_ref = self.anchor
        ds = self.s - s_ref
        if ds <= 0.0:
            return
        if over > 0.0:
            self.anchor = [eid, s_edge, over + ds, self.s]
        else:
            e_old = eid
            eid, s_edge, over = self.map.advance(eid, s_edge, ds)
            self.anchor = [eid, s_edge, over, self.s]
            if self.p.fork_select and self.map.fork_of:
                if not self.sv or self.s - self.sv[-1][0] >= 0.5:
                    self.sv.append((self.s, float(self.x[0])))
                if eid != e_old and eid in self.map.fork_of:
                    self.fork = {'br': self.map.fork_of[eid], 's0': self.s - s_edge, 'cur': eid, 'k': 0}
                if self.fork is not None:
                    self._fork_logic()

    def _fork_logic(self, force: bool = False):
        """Стрелка: какая ветка лучше объясняет скорость после узла (профили по train-прогонам).
        По умолчанию — ветка с наибольшей плотностью проездов; переустановка при перевесе fork_llr."""
        f, step = self.fork, self.map.fork_step
        d = self.s - f['s0']
        n_bins = min(len(v) for v, _ in f['br'].values())
        k = min(int(d / step), n_bins - 1)
        if k <= f['k'] and not force:
            return
        f['k'] = k
        if d >= (n_bins - 1) * step:
            self.fork = None                   # профиль кончился — решение окончательное
        if (d < self.p.fork_min_m and not force) or len(self.sv) < 2:
            return
        sa = np.array([q[0] for q in self.sv])
        va = np.array([q[1] for q in self.sv])
        grid = f['s0'] + step * np.arange(k + 1)
        if grid[0] < sa[0]:
            return
        v = np.interp(grid, sa, va)
        # сравниваются только бины, где профили веток действительно различаются (разница медиан ≥ разброса):
        # на первых метрах после стрелки профили почти одинаковы, и случайные отличия там решать не должны
        if 'mask' not in f:
            mu = np.array([pr[0] for pr in f['br'].values()])
            sd = np.array([pr[1] for pr in f['br'].values()])
            f['mask'] = (mu.max(0) - mu.min(0) >= sd.max(0)) & (mu.min(0) >= 0.0)
        msk = f['mask'][:k + 1]
        if not msk.any():
            return
        fit = {b: branch_fit(v[msk], pr[0][:k + 1][msk], pr[1][:k + 1][msk]) for b, pr in f['br'].items()}
        best = max(fit, key=lambda b: fit[b][0])
        # вложенная стрелка сразу после переустановки: родительское решение уже подтвердило ход по этому пути,
        # достаточно лучшей ветки (иначе положение на время набора перевеса уходит по ветке «по умолчанию»)
        llr = 0.0 if force else self.p.fork_llr
        if best == f['cur'] or fit[best][0] - fit[f['cur']][0] <= llr or fit[best][1] > self.p.fork_fit_max:
            return
        eid, s_edge, over = self.map.advance(best, 0.0, d)
        self.anchor = [eid, s_edge, over, self.s]
        f['cur'] = best
        self.n_fork_switch += 1
        # новая ветка сама может вести через стрелку с профилями — сразу начинаем её разбор
        e, off = best, 0.0
        while e != eid and e is not None:
            off += self.map.edges[e].length
            e = self.map.edges[e].succ
            if e in self.map.fork_of:
                # решение принимается сразу по уже накопленной скорости — иначе положение на время
                # ожидания уходит по ветке «по умолчанию» (другое направление)
                self.fork = {'br': self.map.fork_of[e], 's0': f['s0'] + off, 'cur': e, 'k': 0}
                self._fork_logic(force=True)
                break

    # ================================================================== коррекция по остановкам
    def sigma_along(self) -> float:
        return math.sqrt(self.sig_s0 ** 2 + (self.p.pos_sigma_per_m * (self.s - self.s_corr)) ** 2
                         + (self.var_s_model if self.p.sigma_model_var else 0.0))

    def _stop_logic(self, stamp: float):
        p = self.p
        if self.mode != MODE_MAP or not p.stop_correction or self.anchor[2] > 0.0:
            return
        if self.x[0] > (0.01 if self.imm is not None else 0.0):   # IMM после смешивания даёт ~1e-6 на стоянке
            self.stop_t0, self.stop_done = None, False
            return
        if self.stop_t0 is None:
            self.stop_t0 = stamp
        if self.stop_done or stamp - self.stop_t0 < p.stop_confirm_s:
            return
        self.stop_done = True
        sig = self.sigma_along()
        # окно ≤ длины вагона в норме; шире — только если путь накопил неопределённость без колёс
        win = min(max(3.0 * sig, p.stop_window_min), p.stop_window_max + p.stop_window_model_k * math.sqrt(self.var_s_model))
        cand = self.map.stops_near(self.anchor[0], self.anchor[1], win)
        if not cand:
            return
        d = cand[0][0]
        # две остановки близко — неоднозначно, не корректируем
        if len(cand) > 1 and abs(cand[1][0]) < 2.0 * abs(d) + 2.0 * p.stop_sigma:
            return
        K = sig ** 2 / (sig ** 2 + p.stop_sigma ** 2)
        eid, s_edge, over = self.map.advance(self.anchor[0], self.anchor[1], K * d)
        self.anchor = [eid, s_edge, over, self.s]
        dist = self.s - self.s_corr
        if p.scale_adapt and dist > p.scale_min_dist:
            r = d / dist
            self.scale_adj = min(max(self.scale_adj * (1.0 + p.scale_alpha * r), 1.0 - p.scale_limit), 1.0 + p.scale_limit)
        self.sig_s0 = math.sqrt(1.0 - K) * sig
        self.var_s_model = 0.0
        self.s_corr = self.s
        self.n_stop_corr += 1
        self.last_corr = K * d

    # ================================================================== выход
    def output(self, stamp: float) -> Output:
        p = self.p
        v = float(self.x[0])
        s = self.s
        sig_c = 0.5
        if self.mode == MODE_MAP:
            mp = self.map.pose(self.anchor[0], self.anchor[1], self.anchor[2])
            e, n, u, yaw = mp.e, mp.n, mp.u, mp.hdg
        elif self.mode == MODE_APPROACH:
            k, s0, sref = self.approach_state
            e, n, u, yaw = self.map.approach_pose(k, s0 + (s - sref))
        elif self.mode == MODE_DR:
            e0, n0, u0, psi, s0 = self.anchor
            e, n, u, yaw = e0 + (s - s0) * math.cos(psi), n0 + (s - s0) * math.sin(psi), u0, psi
            sig_c = 0.5 + 0.1 * (s - s0)
        elif self.mode == MODE_PRE and self.prov is not None:
            e, n, u, yaw, s0 = self.prov
            e, n = e + (s - s0) * math.cos(yaw), n + (s - s0) * math.sin(yaw)
            sig_c = 3.0
        else:
            e, n, u, yaw = s, 0.0, 0.0, 0.0
        pos_valid = self.mode != MODE_PRE or self.prov is not None
        sig_a = self.sigma_along()
        lat = lon = h = float('nan')
        if self.frame is not None:
            lat, lon, h = (float(x) for x in self.frame.to_geodetic(e, n, u))
        ratio = max((b.last_ratio for b in self.bog.values() if stamp - b.t < p.bogie_timeout), key=abs, default=0.0)
        wheels_ok = sum(1 for b in self.bog.values() if stamp - b.t_acc < p.bogie_timeout)
        slip = stamp < self.slip_until
        # удельная сила привода = характеристика при текущей позиции − характеристика выбега (сопротивление)
        u_now = self.model.u_at(stamp) if self.model._hist else 0
        a_drive = self.model.accel(u_now, v, 0.0) - self.model.accel(0, v, 0.0)
        eta = p.drive_efficiency if a_drive >= 0 else 1.0 / p.drive_efficiency
        torque = a_drive * p.mass_kg * (1.0 + p.rot_mass_factor) * p.wheel_radius_m / (p.gear_ratio * eta * max(p.n_motors, 1))
        rpm = v / p.wheel_radius_m * p.gear_ratio * 60.0 / (2 * math.pi)
        fault = self.fault_kind if stamp < self.fault_until else ''
        noise = math.sqrt(max(self.noise_var, p.r_wheel ** 2))
        if not fault and noise > p.noise_warn:
            fault = 'high_variance'                 # аномально высокая дисперсия показаний колёс
        edge, s_edge = (int(self.anchor[0]), float(self.anchor[1])) if self.mode == MODE_MAP else (-1, float('nan'))
        return Output(stamp=stamp, v=v, a=float(self.a_last), s=s, e=float(e), n=float(n), u=float(u),
                      yaw=float(yaw), mode=self.mode, var_v=float(self.P[0, 0]), sigma_along=sig_a,
                      sigma_cross=sig_c, slip=slip, slip_kind=self.slip_kind if slip else '',
                      slip_ratio=float(ratio), wheels_ok=wheels_ok,
                      sensor_fault=fault, wheel_noise=noise, pos_valid=pos_valid,
                      a_drive=float(a_drive), torque_nm=float(torque), shaft_rpm=float(rpm),
                      map_edge=edge, map_s=s_edge, lat=lat, lon=lon, h=h)

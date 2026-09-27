"""Привязка к карте по GNSS: начальная выставка и коррекция по редким точкам.

Организаторы: GNSS в первые секунды проверочного прогона гарантирован; в середине маршрута
изредка бывают точки, их можно использовать для коррекции.

1. Начало координат — первая точка опорного приёмника (reference_antenna, по умолчанию master;
   если его точек нет дольше INIT_WAIT — другого). Так же, по-видимому, строит систему судья.
2. Курс — по двум антеннам (база master→rover 12,4 м вдоль трамвая, работает и на стоянке),
   иначе — по смещению опорной антенны.
3. Выставка: проекция точки на кольцо карты с учётом курса (пути встречных направлений идут
   в ~5 м друг от друга) → u₀ = u − s, где s — пройденный путь. Далее u(t) = u₀ + s(t).
   Нет карты или точка дальше MAX_INIT_DISTANCE от неё — прямая от этой точки по курсу.
4. Коррекция: проекция новой точки в окне вокруг ожидаемого места; u₀ поправляется фильтром
   Калмана (вес — неопределённость пути против точности решения GNSS); выбросы отбрасываются.
Модуль не зависит от ROS.
"""
import math
from collections import deque
from typing import Callable, List, Optional, Tuple

from tram_odometry.geo import LocalEnu
from tram_odometry.projection import ProjectedPose, StraightLineProjector
from tram_odometry.route_map import LocalRoute

INIT_WAIT = 1.0                 # с: ждать точку опорного приёмника / курс по двум антеннам
BASELINE_LENGTH = 12.4          # м: расстояние master → rover (калибровка по данным)
BASELINE_TOLERANCE = 1.5        # м: допуск на длину базы вверх
BASELINE_MIN_LENGTH = 10.0      # м: вниз — трамвай сочленённый, на кривой кольца хорда между
                                #   антеннами короче (в данных до 10,5 м, курс при этом верен до ~20°)
BASELINE_HEADING_OFFSET = math.radians(-0.2)   # курс = направление master→rover + это
PAIR_TOLERANCE = 0.1            # с: точки двух антенн считаются одновременными
PAIR_HISTORY = 30               # точек: история каждой антенны для поиска пары (потоки сдвинуты до ~0,5 с)
MOTION_HEADING_MIN = 3.0        # м: смещение для курса по движению
MAX_INIT_DISTANCE = 30.0        # м: дальше от карты — выставка по карте не делается
RAY_START_DISTANCE = 3.0        # м: стоит дальше от карты — ищем карту лучом по курсу
RAY_LENGTH = 100.0              # м: длина луча вперёд по курсу
RAY_HIT = 1.5                   # м: точка луча ближе к карте — попадание
CORRECTION_MAX_LATERAL = 5.0    # м: точка дальше от линии карты — не для коррекции
CORRECTION_MIN_WINDOW = 50.0    # м: окно поиска проекции вокруг ожидаемого места
PATH_DRIFT = 0.006              # доля пути: рост неопределённости после привязки
GNSS_SIGMA_RTK = 0.1            # м: точность точки с RTK-решением (статус 2)
GNSS_SIGMA = 1.5                # м: без RTK
MAP_CROSS_SIGMA = 0.5           # м: поперёк пути на карте (точность карты и колеи)
MAP_Z_SIGMA = 1.0               # м: высота по карте
MAP_YAW_SIGMA = 0.03            # рад: курс по карте
STOP_MIN_SIGMA = 3.0            # м: привязка к остановке — только если путь уже неточен
STOP_EXTRA_SIGMA = 1.0          # м: к разбросу места остановки (очередь, точность остановки)
STOP_GATE = 3.0                 # остановка дальше этого числа σ — не наша (светофор и т. п.)
STOP_MAX_INNOVATION = 20.0      # м: поправка по остановке не больше — иначе трамвай стоит не у платформы (очередь, светофор)


class RouteProjector:
    """Движение по кольцу карты: u = u₀ + пройденный путь."""

    def __init__(self, route: LocalRoute, u0: float, cross_extra_var: float = 0.0) -> None:
        self.route = route
        self.u0 = u0
        # Добавка поперёк пути: трамвай мог стоять на пути, которого нет на карте (параллельный
        # путь у конечной, ~5 м сбоку) — пока привязка не подтвердит, что он на пути карты
        self.cross_extra_var = cross_extra_var

    def project(self, distance: float) -> ProjectedPose:
        x, y, z, yaw = self.route.pose(self.u0 + distance)
        return ProjectedPose(x=x, y=y, z=z, yaw=yaw, cross_var=MAP_CROSS_SIGMA ** 2 + self.cross_extra_var,
                             z_var=MAP_Z_SIGMA ** 2, yaw_var=MAP_YAW_SIGMA ** 2)


class GnssAnchor:

    def __init__(self, route_points: Optional[List[Tuple[float, float, float]]],
                 reference_antenna: str = 'master', correction: bool = True,
                 correction_interval: float = 1.0,
                 log: Optional[Callable[[str, str], None]] = None,
                 stops: Optional[List[Tuple[float, float, float, float]]] = None) -> None:
        self.route_points = route_points
        self.stops = stops or []
        self.stop_u = []                # остановки на кольце: (u, разброс места, м)
        self.stop_updates = 0
        self.reference = reference_antenna if reference_antenna in ('master', 'rover') else 'master'
        self.correction = correction
        self.correction_interval = correction_interval
        self._log = log or (lambda level, text: None)
        self.state = 'ожидание GNSS'
        self.enu = None                 # локальная система: начало — первая точка опорного приёмника
        self.route = None
        self.projector = None           # появляется после выставки
        self.heading = None
        self.heading_source = '-'
        self.u0 = None
        self.init_distance = None       # расстояние до карты при выставке, м
        self.init_offset = None         # расстояние от точки GNSS до выставленного места на карте, м
        self.corrections = 0
        self.rejected = 0
        self._anchor_var = 0.0          # дисперсия u₀ после последней привязки, м²
        self._anchor_s = 0.0            # путь на момент последней привязки, м
        self._first_stamp = None
        self._history = {'master': deque(maxlen=PAIR_HISTORY), 'rover': deque(maxlen=PAIR_HISTORY)}
        self._motion_ref = None         # точка опорной антенны для курса по движению
        self._last_correction = None

    @property
    def on_route(self) -> bool:
        return isinstance(self.projector, RouteProjector)

    def along_var(self, distance: float) -> float:
        """Дисперсия положения вдоль пути после привязки к карте, м²."""
        return self._anchor_var + (PATH_DRIFT * (distance - self._anchor_s)) ** 2

    def on_fix(self, source: str, stamp: float, lat: float, lon: float, alt: float,
               status: int, distance: float) -> None:
        """Точка GNSS; distance — пройденный путь на момент точки."""
        if self._first_stamp is None:
            self._first_stamp = stamp
        if self.enu is None:
            waited = stamp - self._first_stamp > INIT_WAIT
            if source != self.reference and not waited:
                return
            if source != self.reference:
                self._log('warn', f'Точек GNSS {self.reference} нет — опорный приёмник {source}')
                self.reference = source
            self.enu = LocalEnu(lat, lon, alt)
            if self.route_points:
                self.route = LocalRoute(self.route_points, self.enu)
                for lat_s, lon_s, heading_s, spread in self.stops:
                    sx, sy, _ = self.enu.to_enu(lat_s, lon_s, alt)
                    match = self.route.project(sx, sy, heading_s)
                    if match is not None and match[1] <= 5.0:
                        self.stop_u.append((match[0], spread))
        east, north, _ = self.enu.to_enu(lat, lon, alt)
        self._update_heading(source, stamp, east, north)
        if source != self.reference:
            return
        if self.projector is None:
            self._initialize(stamp, east, north, distance)
        elif self.correction and self.on_route:
            self._correct(stamp, east, north, status, distance)

    def _update_heading(self, source, stamp, east, north):
        # Две антенны: пара точек с метками ближе PAIR_TOLERANCE (ищется в истории другой антенны:
        # потоки master и rover сдвинуты до ~0,5 с), длина базы правдоподобна
        other = min(self._history['rover' if source == 'master' else 'master'],
                    key=lambda h: abs(h[0] - stamp), default=None)
        if other is not None and abs(other[0] - stamp) <= PAIR_TOLERANCE:
            (mx, my), (rx, ry) = (((east, north), other[1:]) if source == 'master'
                                  else (other[1:], (east, north)))
            if BASELINE_MIN_LENGTH <= math.hypot(rx - mx, ry - my) <= BASELINE_LENGTH + BASELINE_TOLERANCE:
                self.heading = math.atan2(ry - my, rx - mx) + BASELINE_HEADING_OFFSET
                self.heading_source = 'две антенны'
        self._history[source].append((stamp, east, north))
        # По движению опорной антенны — если курса по двум антеннам нет
        if source == self.reference and self.heading_source != 'две антенны':
            if self._motion_ref is None:
                self._motion_ref = (east, north)
            elif math.hypot(east - self._motion_ref[0], north - self._motion_ref[1]) >= MOTION_HEADING_MIN:
                self.heading = math.atan2(north - self._motion_ref[1], east - self._motion_ref[0])
                self.heading_source = 'по движению'
                self._motion_ref = (east, north)

    def _initialize(self, stamp, east, north, distance):
        if self.heading is None and stamp - self._first_stamp <= INIT_WAIT:
            return              # ждём курс по двум антеннам
        match = self.route.project(east, north, self.heading) if self.route is not None else None
        if match is not None and match[1] > RAY_START_DISTANCE and self.heading is not None:
            # Трамвай стоит до начала известного карте пути (места стоянки на конечных разные):
            # идём лучом вперёд по курсу до линии карты; пройденное по лучу вычитается
            cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
            for k in range(1, int(RAY_LENGTH) + 1):
                rx, ry = east + k * cos_h, north + k * sin_h
                hit = self.route.project(rx, ry, self.heading)
                if hit is not None and hit[1] <= RAY_HIT:
                    # Остаток по курсу от точки луча до её проекции (если луч «недолетел» до начала)
                    px, py = self.route.pose(hit[0])[:2]
                    along = (px - rx) * cos_h + (py - ry) * sin_h
                    match = (self.route.wrap(hit[0] - k - along), hit[1])
                    self._log('info', f'Трамвай стоит в {k} м до начала пути на карте — выставка лучом по курсу')
                    break
        if match is not None and match[1] <= MAX_INIT_DISTANCE:
            self.u0 = self.route.wrap(match[0] - distance)
            self.init_distance = match[1]
            # Начальная ошибка — расстояние от точки GNSS до места на карте, которое будет
            # опубликовано (при выставке лучом оно больше, чем match[1] у точки попадания луча)
            px, py = self.route.pose(match[0])[:2]
            self.init_offset = math.hypot(east - px, north - py)
            self.projector = RouteProjector(self.route, self.u0, self.init_offset ** 2)
            self._anchor_var, self._anchor_s = MAP_CROSS_SIGMA ** 2 + self.init_offset ** 2, distance
            self.state = 'по карте'
            self._log('info', f'Выставка по карте: u = {match[0]:.1f} м из {self.route.length:.0f}, '
                              f'до карты {match[1]:.2f} м, курс: {self.heading_source}')
        else:
            yaw = self.heading if self.heading is not None else 0.0
            self.projector = StraightLineProjector(origin=(east, north, 0.0), yaw=yaw,
                                                   distance0=distance)
            self.state = 'вне карты' if self.route is not None else 'без карты'
            self._log('warn', f'Выставка без карты ({self.state}): прямая от точки GNSS, '
                              f'курс: {self.heading_source}')

    def on_standstill(self, distance: float) -> None:
        """Трамвай стоит: если рядом известная остановка, уточнить место на кольце по ней.

        Остановки (из обучающих прогонов) — места стабильной стоянки с разбросом ~1 м; привязка
        делается, только когда неопределённость пути уже больше STOP_MIN_SIGMA, и только к
        остановке в пределах STOP_GATE σ (иначе это, например, светофор)."""
        if not self.on_route or not self.stop_u:
            return
        prior_var = self.along_var(distance)
        if prior_var < STOP_MIN_SIGMA ** 2:
            return
        u_pred = self.route.wrap(self.u0 + distance)
        u_stop, spread = min(self.stop_u, key=lambda s: abs(self.route.delta(s[0], u_pred)))
        innovation = self.route.delta(u_stop, u_pred)
        stop_var = (spread + STOP_EXTRA_SIGMA) ** 2
        if abs(innovation) > min(STOP_GATE * math.sqrt(prior_var + stop_var), STOP_MAX_INNOVATION):
            return
        gain = prior_var / (prior_var + stop_var)
        self.u0 = self.route.wrap(self.u0 + gain * innovation)
        self.projector.u0 = self.u0
        self.projector.cross_extra_var = 0.0        # у платформы — на пути карты
        self._anchor_var, self._anchor_s = (1.0 - gain) * prior_var, distance
        self.stop_updates += 1

    def _correct(self, stamp, east, north, status, distance):
        if self._last_correction is not None and stamp - self._last_correction < self.correction_interval:
            return
        route = self.route
        u_pred = route.wrap(self.u0 + distance)
        prior_var = self.along_var(distance)
        window = max(CORRECTION_MIN_WINDOW, 4.0 * math.sqrt(prior_var))
        match = route.project(east, north, heading=route.pose(u_pred)[3], max_heading_error=math.pi / 4,
                              near_u=u_pred, window=window)
        if match is None or match[1] > CORRECTION_MAX_LATERAL:
            self.rejected += 1
            return
        innovation = route.delta(match[0], u_pred)
        # Точка в стороне от линии карты (соседний путь) задаёт место на кольце неточно
        gnss_var = (GNSS_SIGMA_RTK if status == 2 else GNSS_SIGMA) ** 2 + match[1] ** 2
        if abs(innovation) > 5.0 * math.sqrt(prior_var + gnss_var) and abs(innovation) > 10.0:
            self.rejected += 1
            return
        gain = prior_var / (prior_var + gnss_var)
        self.u0 = route.wrap(self.u0 + gain * innovation)
        self.projector.u0 = self.u0
        self.projector.cross_extra_var = match[1] ** 2
        # Смещение точки от линии карты систематическое (соседний путь): повторные точки
        # не делают место на кольце точнее него
        self._anchor_var, self._anchor_s = max((1.0 - gain) * prior_var, match[1] ** 2), distance
        self._last_correction = stamp
        self.corrections += 1

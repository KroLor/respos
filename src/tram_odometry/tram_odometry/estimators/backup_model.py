"""Оценщик backup_model: модель движения из пакета tram_backup_odometry.

Расчётное ядро пакета (фильтр Калмана по скоростям тележек с моделью привода, эхо-сеть,
отбраковка проскальзываний, выставка по GNSS, карта пути со стрелками и остановками)
вызывается строго по его контракту (docs/INTEGRATION.md): значения как в топиках (скорость
колёс в км/ч) в порядке прихода, время — header.stamp; только сообщения, чьи метки и значения
прошли предобработку ноды (мусорная метка иначе сбрасывает ядро). Положение ядро выдаёт само —
точка base_link в координатах судьи (плоские MGRS).

Сверх ядра здесь — коррекция по редким точкам GNSS в середине маршрута (организаторы
разрешили их использовать): после выставки ядро GNSS не слушает, а точка приёмника
сравнивается с положением той же антенны на карте. Невязка вдоль пути уточняет место
на карте фильтром Калмана (как коррекция на остановках в самом ядре); если несколько
точек подряд лежат в стороне от пути, но на соседней ветке карты, — трамвай
переставляется на эту ветку (ошибка выбора ветки на стрелке).
"""
import math
from collections import deque

from tram_odometry.estimators.base import Estimate, Estimator

# Режимы ядра, в которых положение привязано к системе координат результата
POSITION_MODES = ('map', 'approach', 'dead_reckoning', 'pre_init')
MODE_MAP = 'map'

# Коррекция по GNSS в середине маршрута
GNSS_MIN_STATUS = 2              # только точки с решением (status ≥ 2)
# м: ошибка точки вдоль пути относительно карты (геометрия карты, свес антенны над рельсом на кривых,
# метки времени) — как точность места остановки в самом ядре (stop_sigma)
GNSS_ALONG_SIGMA = 1.5
GNSS_ALONG_GATE = 4.0            # σ: невязка больше — выброс
GNSS_ALONG_GATE_MIN = 10.0       # м: меньшие невязки не отбраковываются
GNSS_CROSS_GATE = 3.0            # м: точка дальше от пути — не для коррекции вдоль него
BRANCH_MATCH_DIST = 2.0          # м: точка так близко к другой ветке — довод за перестановку
BRANCH_CONFIRM = 5               # столько точек подряд должны указать на одну ветку
BRANCH_WINDOW = 3.0              # с: и уложиться в это время
HISTORY_SEC = 5.0                # с: история пути для точки GNSS с запаздывающей меткой


def _installed_share():
    """Каталог данных установленного пакета модели; None — без ROS (данные рядом с исходниками)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('tram_backup_odometry')
    except Exception:
        return None


class BackupModelEstimator(Estimator):

    name = 'backup_model'
    raw_inputs = True
    provides_position = True

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # Пакет модели — отдельный пакет; без него оценщик не создаётся и ядро
        # ноды переходит на запасной wheel_baseline
        from tram_backup_odometry.core import build_estimator, default_files
        self._core = build_estimator(**default_files(params.get('backup_model_share')
                                                     or _installed_share()))
        self._gnss_correction = bool(params.get('gnss_correction', True))
        self._last = None                   # последний выход ядра
        self._history = deque()             # (метка, путь) — путь на момент точки GNSS
        self._branch_votes = deque()        # (метка, ребро) — точки GNSS на другой ветке
        self.resets = 0
        self.gnss_corrections = 0
        self.gnss_rejected = 0
        self.branch_switches = 0

    # --- Сырые входы: ровно контракт ядра ---

    def on_raw_wheel(self, bogie, stamp, speed_kmh):
        self._store(self._guard(self._core.on_wheel, bogie, stamp, float(speed_kmh)))

    def on_raw_driver_cmd(self, stamp, position):
        self._store(self._guard(self._core.on_cmd, stamp, int(position)))

    def on_raw_gnss_fix(self, source, stamp, status, latitude, longitude, altitude):
        if not self._core.gnss_done:
            self._guard(self._core.on_fix, source, stamp, latitude, longitude, altitude, int(status))
        elif self._gnss_correction:
            self._guard(self._correct_by_gnss, source, stamp, int(status),
                        latitude, longitude, altitude)

    def _guard(self, method, *args):
        """Ошибка внутри ядра: сброс его состояния (как в ноде пакета) и сбой для supervisor."""
        try:
            return method(*args)
        except Exception:
            self._core.reset()
            self._last = None
            self._history.clear()
            self.resets += 1
            raise

    def _store(self, output) -> None:
        if output is None:
            return
        if self._last is None or output.stamp >= self._last.stamp:
            self._last = output
            if self._history and output.stamp < self._history[-1][0]:
                self._history.clear()       # время пошло назад — история больше не годится
            self._history.append((output.stamp, output.s))
            while output.stamp - self._history[0][0] > HISTORY_SEC:
                self._history.popleft()

    # --- Коррекция по GNSS после выставки ---

    def _distance_at(self, stamp):
        """Путь ядра на момент stamp по истории выходов (None — вне истории)."""
        h = self._history
        if not h or stamp < h[0][0] - 0.2:
            return None
        if stamp >= h[-1][0]:
            return h[-1][1]
        for (t0, s0), (t1, s1) in zip(h, list(h)[1:]):
            if t0 <= stamp <= t1:
                w = (stamp - t0) / (t1 - t0) if t1 > t0 else 0.0
                return s0 + w * (s1 - s0)
        return h[0][1]

    def _correct_by_gnss(self, source, stamp, status, lat, lon, alt):
        core = self._core
        anchor = core.anchor
        if (core.mode != MODE_MAP or anchor is None or anchor[2] > 0.0 or core.frame is None
                or source not in ('master', 'rover')):
            return
        if status < GNSS_MIN_STATUS or not all(math.isfinite(v) for v in (lat, lon, alt)):
            self.gnss_rejected += 1
            return
        s_fix = self._distance_at(stamp)
        if s_fix is None:
            self.gnss_rejected += 1
            return
        p, pathmap = core.p, core.map
        antenna_x = p.master_x if source == 'master' else p.rover_x
        # Где по оценке была эта антенна в момент точки: base_link сейчас → назад на путь,
        # пройденный после точки, → на смещение антенны вдоль вагона
        back = core.s - s_fix
        eid, s_edge, over = pathmap.advance(anchor[0], anchor[1], antenna_x - back)
        if over > 0.0:
            return
        pose = pathmap.pose(eid, s_edge)
        e, n, _ = core.frame.to_enu(lat, lon, alt)
        c, s = math.cos(pose.hdg), math.sin(pose.hdg)
        along = (e - pose.e) * c + (n - pose.n) * s
        cross = -(e - pose.e) * s + (n - pose.n) * c
        if abs(cross) > GNSS_CROSS_GATE:
            self._vote_branch(stamp, e, n, pose.hdg, antenna_x, back, eid)
            return
        self._branch_votes.clear()
        sigma = core.sigma_along()
        innovation_var = sigma ** 2 + GNSS_ALONG_SIGMA ** 2
        if abs(along) > max(GNSS_ALONG_GATE * math.sqrt(innovation_var), GNSS_ALONG_GATE_MIN):
            self.gnss_rejected += 1
            return
        gain = sigma ** 2 / innovation_var
        # Сдвиг места на карте и уменьшение неопределённости — как коррекция на остановке в ядре
        eid, s_edge, over = pathmap.advance(anchor[0], anchor[1], gain * along)
        core.anchor = [eid, s_edge, over, core.s]
        core.sig_s0 = math.sqrt(1.0 - gain) * sigma
        core.s_corr = core.s
        core.var_s_model = 0.0
        self.gnss_corrections += 1

    def _vote_branch(self, stamp, e, n, hdg, antenna_x, back, current_edge):
        """Точка в стороне от пути: если несколько подряд ложатся на другую ветку — перестановка."""
        core = self._core
        match = core.map.match(e, n, hdg, BRANCH_MATCH_DIST)
        if match is None or match[0] == current_edge:
            self.gnss_rejected += 1
            return
        votes = self._branch_votes
        votes.append((stamp, match[0]))
        while votes and stamp - votes[0][0] > BRANCH_WINDOW:
            votes.popleft()
        if len(votes) < BRANCH_CONFIRM or len({edge for _, edge in votes}) != 1:
            return
        # Все последние точки — на одной ветке: base_link на ней = антенна + путь после точки − смещение
        eid, s_edge, over = core.map.advance(match[0], match[1], back - antenna_x)
        core.anchor = [eid, s_edge, over, core.s]
        core.fork = None
        core.s_corr = core.s
        core.sig_s0 = GNSS_ALONG_SIGMA
        core.var_s_model = 0.0
        votes.clear()
        self.branch_switches += 1

    # --- Оценка ---

    def estimate(self, stamp: float) -> Estimate:
        o = self._last
        if o is None:
            raise RuntimeError('модель ещё не выдала оценку')
        # Публикация идёт по метке того же входного сообщения (dt = 0); иначе — экстраполяция
        ds = o.v * max(0.0, stamp - o.stamp)
        position = yaw = None
        if o.pos_valid and o.mode in POSITION_MODES:
            position = (o.e + ds * math.cos(o.yaw), o.n + ds * math.sin(o.yaw), o.u)
            yaw = o.yaw
        return Estimate(
            velocity=o.v, distance=o.s + ds, acceleration=o.a,
            velocity_var=o.var_v, distance_var=o.sigma_along ** 2,
            position=position, yaw=yaw,
            slip_detected=bool(o.slip), slip_ratio=o.slip_ratio,
            position_cross_var=o.sigma_cross ** 2, position_z_var=1.0)

    def diagnostics(self) -> dict:
        core, o = self._core, self._last
        values = {
            'модель: режим': core.mode,
            'модель: выставка по GNSS': 'да' if core.gnss_done else 'нет',
            'модель: коррекций по остановкам': core.n_stop_corr,
            'модель: переустановок ветки на стрелках': core.n_fork_switch,
            'модель: поправка масштаба колёс': f'{core.scale_adj:.5f}',
            'модель: сбросов после ошибки': self.resets,
            'GNSS в пути: коррекций': self.gnss_corrections,
            'GNSS в пути: отклонено точек': self.gnss_rejected,
            'GNSS в пути: перестановок на другую ветку': self.branch_switches,
        }
        if o is not None:
            values.update({
                'модель: σ вдоль пути, м': f'{o.sigma_along:.2f}',
                'модель: работающих тележек': o.wheels_ok,
                'модель: сбой датчика': o.sensor_fault or '-',
                'модель: вид проскальзывания': o.slip_kind or '-',
            })
        return values

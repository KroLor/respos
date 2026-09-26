import math
from pathlib import Path

from tram_odometry.gnss_anchor import GnssAnchor, RouteProjector
from tram_odometry.geo import WGS84_A, WGS84_E2, LocalEnu
from tram_odometry.projection import StraightLineProjector
from tram_odometry.route_map import LocalRoute, load_route_csv

LAT0, LON0, ALT0 = 55.80, 37.40, 150.0
# Метров в градусе на широте LAT0 — радиусы кривизны эллипсоида WGS84
_W = 1.0 - WGS84_E2 * math.sin(math.radians(LAT0)) ** 2
M_PER_DEG_LAT = math.radians(1.0) * WGS84_A * (1.0 - WGS84_E2) / _W ** 1.5
M_PER_DEG_LON = math.radians(1.0) * WGS84_A / math.sqrt(_W) * math.cos(math.radians(LAT0))
MAP_FILE = Path(__file__).resolve().parents[1] / 'config' / 'route_map.csv'


def geodetic(x, y):
    """Локальные метры (восток, север) от (LAT0, LON0) → широта, долгота, высота."""
    return LAT0 + y / M_PER_DEG_LAT, LON0 + x / M_PER_DEG_LON, ALT0


def test_ring():
    """Кольцо: путь «туда» на восток по y = 0, путь «обратно» на запад по y = 5 (как встречные пути)."""
    pts = [(float(x), 0.0) for x in range(0, 1000)]
    pts += [(1000.0, float(y)) for y in range(0, 5)]
    pts += [(float(x), 5.0) for x in range(1000, 0, -1)]
    pts += [(0.0, float(y)) for y in range(5, 0, -1)]
    return [geodetic(x, y) for x, y in pts]


def local_route():
    return LocalRoute(test_ring(), LocalEnu(LAT0, LON0, ALT0))


def test_pose_along_the_ring():
    route = local_route()
    assert abs(route.length - 2010.0) < 0.5
    x, y, _, yaw = route.pose(500.0)
    assert abs(x - 500.0) < 0.1 and abs(y) < 0.1 and abs(yaw) < 0.01
    x, y, _, yaw = route.pose(1505.0)                      # путь «обратно»
    assert abs(x - 500.0) < 0.2 and abs(y - 5.0) < 0.1 and abs(abs(yaw) - math.pi) < 0.01
    assert abs(route.pose(2010.0 + 500.0)[0] - 500.0) < 0.1   # кольцо замкнуто


def test_projection_respects_heading_and_window():
    route = local_route()
    u, d = route.project(500.0, 4.0)                       # без курса — ближайший путь («обратно»)
    assert abs(u - 1505.0) < 0.5 and abs(d - 1.0) < 0.05
    u, d = route.project(500.0, 4.0, heading=0.0)          # курс на восток — путь «туда»
    assert abs(u - 500.0) < 0.5 and abs(d - 4.0) < 0.05
    u, d = route.project(520.0, 0.3, near_u=510.0, window=50.0)
    assert abs(u - 520.0) < 0.5
    wrap_gap = route.length - 2000.0                        # от u = 2005 до u = 5 через конец кольца
    assert abs(route.delta(5.0, 2005.0) - wrap_gap) < 1e-9 and abs(route.delta(2005.0, 5.0) + wrap_gap) < 1e-9


def fix(anchor, source, stamp, x, y, distance=0.0, status=2):
    lat, lon, alt = geodetic(x, y)
    anchor.on_fix(source, stamp, lat, lon, alt, status, distance)


def test_initialization_picks_track_by_two_antenna_heading():
    anchor = GnssAnchor(test_ring())
    # Трамвай стоит на пути «обратно» (едет на запад); точка ближе к пути «туда» (2,4 м против 2,6)
    for k in range(5):
        t = 0.1 * k
        fix(anchor, 'master', t, 600.0, 2.4)
        fix(anchor, 'rover', t + 0.01, 600.0 - 12.4, 2.4)  # rover впереди по ходу движения
    assert anchor.on_route and anchor.heading_source == 'две антенны'
    assert abs(anchor.u0 - 1405.0) < 1.0                    # путь «обратно»: u = 1005 + (1000 − 600)
    # Начало координат — первая точка master: положение по карте в её системе
    x, y, _, yaw = anchor.projector.route.pose(anchor.u0)
    assert abs(x) < 0.5 and abs(y - 2.6) < 0.2 and abs(abs(yaw) - math.pi) < 0.05


def test_two_antenna_heading_with_shifted_streams():
    # Как в данных: сообщения rover приходят со сдвигом меток на 0,3 с относительно master
    anchor = GnssAnchor(test_ring())
    for k in range(8):
        fix(anchor, 'rover', 0.1 * k, 600.0 - 12.4, 2.4)
        fix(anchor, 'master', 0.3 + 0.1 * k, 600.0, 2.4)
    assert anchor.heading_source == 'две антенны' and anchor.on_route
    assert abs(anchor.u0 - 1405.0) < 1.0


def test_initialization_before_map_start_uses_heading_ray():
    # Карта — только путь на восток от x = 100; трамвай стоит на его продолжении в x = 80
    points = [geodetic(float(x), 0.0) for x in range(100, 600)] + [geodetic(600.0, 50.0), geodetic(100.0, 50.0)]
    anchor = GnssAnchor(points)
    for k in range(3):
        fix(anchor, 'master', 0.1 * k, 80.0, 0.0)
        fix(anchor, 'rover', 0.1 * k + 0.01, 92.4, 0.0)
    assert anchor.on_route
    assert abs(anchor.route.delta(anchor.u0, 0.0) + 20.0) < 1.0     # на 20 м до начала карты


def test_initialization_waits_for_heading_then_uses_nearest():
    anchor = GnssAnchor(test_ring())
    fix(anchor, 'master', 0.0, 300.0, 0.2)
    assert anchor.projector is None                        # ждём курс по второй антенне
    fix(anchor, 'master', 1.5, 300.0, 0.2)                 # второй антенны нет — ближайший путь
    assert anchor.on_route and abs(anchor.u0 - 300.0) < 0.5


def test_correction_pulls_position_and_rejects_outliers():
    anchor = GnssAnchor(test_ring(), correction_interval=0.0)
    for k in range(5):
        fix(anchor, 'master', 0.1 * k, 500.0, 0.0)
        fix(anchor, 'rover', 0.1 * k + 0.01, 512.4, 0.0)
    assert abs(anchor.u0 - 500.0) < 0.5
    before = anchor.corrections
    # Прошли по одометрии 400 м (σ ≈ 2,4 м), а GNSS видит трамвай на 5 м дальше — поправка
    fix(anchor, 'master', 40.0, 905.0, 0.0, distance=400.0)
    assert anchor.corrections == before + 1 and abs(anchor.u0 - 505.0) < 0.5
    var_after = anchor.along_var(400.0)
    fix(anchor, 'master', 41.0, 600.0, 0.0, distance=410.0)   # точка на 315 м позади — выброс
    assert anchor.rejected == 1 and abs(anchor.u0 - 505.0) < 0.5
    assert var_after < 0.05


def test_without_map_straight_line_from_fix_along_heading():
    anchor = GnssAnchor(None)
    fix(anchor, 'master', 0.0, 0.0, 0.0)
    fix(anchor, 'rover', 0.01, 0.0, 12.4)                  # rover севернее — курс на север
    fix(anchor, 'master', 0.1, 0.0, 0.0, distance=5.0)
    assert isinstance(anchor.projector, StraightLineProjector) and anchor.state == 'без карты'
    pose = anchor.projector.project(15.0)                   # 10 м после выставки
    assert abs(pose.x) < 0.1 and abs(pose.y - 10.0) < 0.1


def test_packaged_route_map():
    points = load_route_csv(str(MAP_FILE))
    route = LocalRoute(points, LocalEnu(*points[0]))
    assert 10_900 < route.length < 11_100                  # кольцо Щукинская — Таллинская ~11 км
    projector = RouteProjector(route, 0.0)
    pose = projector.project(route.length / 2)
    assert pose.cross_var < 1.0 and pose.yaw_var < 0.01

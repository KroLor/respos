#!/usr/bin/env python3
"""Сборка карты маршрута (замкнутое кольцо) из GNSS обучающих прогонов.

Трамвай ходит между двумя конечными (разворотные кольца у Щукинской и у Таллинской).
Прогон начинается на стоянке у одной конечной перед разворотной петлёй, проходит петлю
и идёт до другой конечной. Два прогона разных направлений вместе дают полное кольцо:
петля A → путь A→B → петля B → путь B→A. Карта — это кольцо в порядке движения.

Что делает скрипт:
1. Читает координаты приёмника master из всех bag (sqlite + CDR, ROS не нужен).
2. Находит две конечные — места, где чаще всего начинаются и заканчиваются прогоны.
3. Для каждого направления выбирает лучший прогон трамвая 30618: доля RTK-решений,
   нет пропусков GNSS длиннее 1 с, длина пути близка к медиане направления.
4. Склеивает их в кольцо, прореживает с шагом 1 м (убирает дрожание на стоянках),
   сглаживает высоту.
5. Проверяет карту на всех прогонах: расстояние их точек GNSS до линии карты.

Запуск: python3 tools/build_route_map.py <папка data датасета> src/tram_odometry/config/route_map.csv
Только стандартная библиотека Python.
"""
import argparse
import bisect
import math
import sqlite3
import statistics
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'tram_odometry'))
from tram_odometry.geo import LocalEnu  # noqa: E402

VEHICLE = '30618'          # трамвай скрытой проверки — карта по его прогонам
TERMINAL_RADIUS = 150.0    # м: начала и концы прогонов ближе этого — одна конечная
STEP = 1.0                 # м: шаг точек карты
ALT_SMOOTH = 25            # точек (м): окно сглаживания высоты


def read_master_fixes(bag: Path):
    """Точки GNSS master: (метка, широта, долгота, высота, статус)."""
    con = sqlite3.connect(next(bag.glob('*.db3')))
    tid = {name: i for i, name in con.execute('select id, name from topics')}
    if '/sensing/gnss/master/fix' not in tid:
        return []
    out = []
    for (raw,) in con.execute('select data from messages where topic_id=? order by timestamp',
                              (tid['/sensing/gnss/master/fix'],)):
        sec, nsec, flen = struct.unpack_from('<iII', raw, 4)
        off = 16 + flen
        status = struct.unpack_from('<b', raw, off)[0]
        off = 4 + ((off + 4 - 4 + 7) // 8) * 8          # NavSatStatus (4 байта), выравнивание 8
        lat, lon, alt = struct.unpack_from('<3d', raw, off)
        if status >= 0 and all(math.isfinite(v) for v in (lat, lon, alt)) and abs(lat) > 1:
            out.append((sec + nsec * 1e-9, lat, lon, alt, status))
    return sorted(out)


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


MAX_SPEED = 25.0            # м/с: переход между точками GNSS быстрее — выброс
MAX_TURN = math.radians(120)  # поворот линии больше этого на шаге ~1 м — артефакт (задним ходом не ездит)


def remove_jumps(fixes, local):
    """Отбросить точки GNSS, переход к которым означает скорость выше MAX_SPEED."""
    kept_f, kept_l = [fixes[0]], [local[0]]
    for f, p in zip(fixes[1:], local[1:]):
        dt = f[0] - kept_f[-1][0]
        if dt > 0 and math.dist(p[:2], kept_l[-1][:2]) <= MAX_SPEED * dt + 0.5:
            kept_f.append(f)
            kept_l.append(p)
    return kept_l


def remove_reversals(points):
    """Убрать развороты линии: трамвай не ездит задним ходом, значит это «пила» выбросов GNSS."""
    out = points[:2]
    for p in points[2:]:
        a = math.atan2(out[-1][1] - out[-2][1], out[-1][0] - out[-2][0])
        b = math.atan2(p[1] - out[-1][1], p[0] - out[-1][0])
        if abs((b - a + math.pi) % (2 * math.pi) - math.pi) <= MAX_TURN:
            out.append(p)
    return out


def resample(xyz, step):
    """Точки через step метров по ломаной; стоячие точки (ближе step/2) отбрасываются."""
    pts = [xyz[0]]
    for p in xyz[1:]:
        if math.dist(p[:2], pts[-1][:2]) >= step / 2:
            pts.append(p)
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a[:2], b[:2])
        pos = step - carry
        while pos <= seg:
            t = pos / seg
            out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
            pos += step
        carry = seg - (pos - step)
    return out


def smooth_altitude(points, window):
    half = window // 2
    n = len(points)
    alts = [p[2] for p in points]
    out = []
    for i, p in enumerate(points):
        # Кольцо: окно замыкается через начало
        vals = [alts[(i + k) % n] for k in range(-half, half + 1)]
        out.append((p[0], p[1], sum(vals) / len(vals)))
    return out


def distance_to_polyline(x, y, xs, ys, grid, cell):
    """Расстояние от точки до ломаной (кольца); grid — индекс сегментов по клеткам."""
    best = math.inf
    gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for i in grid.get((gx + dx, gy + dy), ()):
                j = (i + 1) % len(xs)
                ax, ay, bx, by = xs[i], ys[i], xs[j], ys[j]
                vx, vy = bx - ax, by - ay
                ll = vx * vx + vy * vy
                t = 0.0 if ll == 0 else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / ll))
                best = min(best, math.hypot(ax + t * vx - x, ay + t * vy - y))
    return best


MAX_EXTENSION = 60.0        # м: продление начала/конца — только в пределах мест стоянки


def extend_ends(core, others, join=1.5):
    """Продлить ломаную core: начало — по прогону, стартовавшему раньше всех на том же пути,
    конец — по прогону, доехавшему дальше всех. Места стоянки на конечных у прогонов разные
    (до ~30 м), и карта должна покрывать все.

    Для каждого прогона ищется первая (последняя) точка ближе join метров к началу (концу) core;
    всё, что до (после) неё, — кандидат на продление; берётся самый длинный.
    """
    head, tail = [], []
    start, end = core[0][:2], core[-1][:2]
    for pts in others:
        i = next((k for k, p in enumerate(pts) if math.dist(p[:2], start) < join), None)
        if i is not None:
            length = path_length([p[:2] for p in pts[:i + 1]])
            if path_length([p[:2] for p in head]) < length <= MAX_EXTENSION:
                head = pts[:i]
        j = next((k for k in range(len(pts) - 1, -1, -1) if math.dist(pts[k][:2], end) < join), None)
        if j is not None:
            length = path_length([p[:2] for p in pts[j:]])
            if path_length([p[:2] for p in tail]) < length <= MAX_EXTENSION:
                tail = pts[j + 1:]
    return head + core + tail, path_length([p[:2] for p in head]), path_length([p[:2] for p in tail])


def validate_lengths(route, local, rtk_names, chunk=500.0):
    """Длина карты против пути прогонов: на отрезках по chunk метров пути прогона сравнить
    прирост координаты на карте Δu с путём по GNSS Δs. Лишняя или недостающая длина карты
    (например, «пила» выбросов GNSS в исходном прогоне) даёт постоянную ошибку положения."""
    worst = []
    for name in rtk_names:
        # Путь прогона по GNSS тоже чистится от «пилы», иначе его длина завышена
        pts = remove_reversals(resample(local[name], STEP))
        first = route.project(pts[0][0], pts[0][1])
        if first is None or first[1] > 5.0:
            continue
        u_prev, s_acc, s_prev, last = first[0], 0.0, 0.0, pts[0]
        for p in pts[1:]:
            step = math.dist(p[:2], last[:2])
            s_acc += step
            heading = math.atan2(p[1] - last[1], p[0] - last[0]) if step > 0.5 else None
            last = p
            if s_acc - s_prev < chunk or heading is None:
                continue
            ds = s_acc - s_prev
            match = route.project(p[0], p[1], heading=heading, near_u=u_prev + ds, window=100.0)
            if match is None or match[1] > 3.0:
                break                                   # прогон ушёл с маршрута
            du = route.delta(match[0], u_prev)
            worst.append((abs(du - ds), du - ds, name, u_prev))
            u_prev, s_prev = match[0], s_acc
    worst.sort(reverse=True)
    errs = sorted(w[0] for w in worst)
    if errs:
        print(f'Проверка длины на {len(worst)} отрезках по {chunk:.0f} м: |Δu − Δs| медиана '
              f'{errs[len(errs) // 2]:.2f} м, p95 {errs[int(0.95 * len(errs))]:.2f} м, макс {errs[-1]:.2f} м')
        for _, diff, name, u in worst[:5]:
            print(f'   {name}: отрезок с u ≈ {u:.0f} м — карта {"длиннее" if diff > 0 else "короче"} на {abs(diff):.1f} м')
    return errs


STOP_MOVE = 0.5            # м: за STOP_MIN_TIME трамвай сдвинулся меньше — стоит
STOP_MIN_TIME = 5.0        # с
STOP_CLUSTER_GAP = 8.0     # м: стоянки ближе этого вдоль кольца — одно место
STOP_MIN_RUNS = 8          # место считается остановкой, если там стояли в стольких прогонах
STOP_MAX_SPREAD = 3.0      # м: и разброс мест стоянки не больше этого


def find_stops(route, runs, enu, names):
    """Места, где трамваи стабильно останавливаются (платформы): (lat, lon, курс, u, σ, прогонов)."""
    episodes = []           # (u, прогон, x, y, курс)
    for name in names:
        fx = runs[name]
        # Выбросы GNSS убираются так же, как в remove_jumps, но с сохранением меток времени
        kept, stamps = [enu.to_enu(*fx[0][1:4])], [fx[0][0]]
        for f, p in zip(fx[1:], [enu.to_enu(*f[1:4]) for f in fx[1:]]):
            if math.dist(p[:2], kept[-1][:2]) <= MAX_SPEED * max(f[0] - stamps[-1], 1e-3) + 0.5:
                kept.append(p)
                stamps.append(f[0])
        i = 0
        while i < len(kept):
            j = i
            while j + 1 < len(kept) and math.dist(kept[j + 1][:2], kept[i][:2]) < STOP_MOVE:
                j += 1
            if stamps[j] - stamps[i] >= STOP_MIN_TIME:
                back = next((k for k in range(i, -1, -1) if math.dist(kept[k][:2], kept[i][:2]) > 5.0), None)
                if back is not None:
                    heading = math.atan2(kept[i][1] - kept[back][1], kept[i][0] - kept[back][0])
                    x = sum(p[0] for p in kept[i:j + 1]) / (j + 1 - i)
                    y = sum(p[1] for p in kept[i:j + 1]) / (j + 1 - i)
                    match = route.project(x, y, heading)
                    if match is not None and match[1] <= 3.0:
                        episodes.append((match[0], name, x, y, heading))
            i = j + 1
    episodes.sort()
    stops, group = [], []
    for ep in episodes + [None]:
        if ep is not None and (not group or ep[0] - group[-1][0] <= STOP_CLUSTER_GAP):
            group.append(ep)
            continue
        if len({g[1] for g in group}) >= STOP_MIN_RUNS:
            us = [g[0] for g in group]
            mean = sum(us) / len(us)
            spread = math.sqrt(sum((u - mean) ** 2 for u in us) / len(us))
            if spread <= STOP_MAX_SPREAD:
                x, y = sum(g[2] for g in group) / len(group), sum(g[3] for g in group) / len(group)
                heading = math.atan2(sum(math.sin(g[4]) for g in group), sum(math.cos(g[4]) for g in group))
                stops.append((x, y, heading, mean, spread, len({g[1] for g in group})))
        group = [ep] if ep is not None else []
    return stops


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('data', type=Path, help='папка data датасета (122 bag)')
    parser.add_argument('output', type=Path, help='файл карты .csv')
    args = parser.parse_args()

    runs = {}
    for bag in sorted(args.data.iterdir()):
        fixes = read_master_fixes(bag)
        if len(fixes) >= 100:
            runs[bag.name] = fixes
    enu = LocalEnu(*next(iter(runs.values()))[0][1:4])
    # Локальные координаты без выбросов GNSS (скачков быстрее MAX_SPEED)
    local = {name: remove_jumps(fx, [enu.to_enu(*f[1:4]) for f in fx]) for name, fx in runs.items()}

    # Конечные — два самых частых места начала и конца прогонов
    ends = [(name, key, local[name][idx][:2]) for name in runs for key, idx in (('start', 0), ('end', -1))]
    clusters = []
    for name, key, p in ends:
        for c in clusters:
            if math.dist(p, c['p']) < TERMINAL_RADIUS:
                c['n'] += 1
                break
        else:
            clusters.append({'p': p, 'n': 1})
    clusters.sort(key=lambda c: -c['n'])
    terminals = [c['p'] for c in clusters[:2]]

    def terminal_of(p):
        for k, t in enumerate(terminals):
            if math.dist(p, t) < TERMINAL_RADIUS:
                return k
        return None

    # Лучший прогон 30618 в каждом направлении; начало и конец продлеваются по другим прогонам
    chosen, directions = {}, {}
    for a, b in ((0, 1), (1, 0)):
        cands = [n for n in runs if n.startswith(VEHICLE)
                 and terminal_of(local[n][0][:2]) == a and terminal_of(local[n][-1][:2]) == b]
        lengths = {n: path_length([p[:2] for p in local[n]]) for n in cands}
        med = statistics.median(lengths.values())

        def quality(n):
            fx = runs[n]
            rtk = sum(1 for f in fx if f[4] == 2) / len(fx)
            max_gap = max(b[0] - a[0] for a, b in zip(fx, fx[1:]))
            return (max_gap <= 1.0, abs(lengths[n] - med) < 0.03 * med, rtk)
        chosen[(a, b)] = max(cands, key=quality)
        n = chosen[(a, b)]
        # Для продления — все прогоны направления: места стоянки у конечных разные, а точности
        # без RTK (~1 м) для нескольких десятков метров у стоянки достаточно
        others = [local[m] for m in cands if m != n]
        directions[(a, b)], head, tail = extend_ends(local[n], others)
        print(f'Направление {a}→{b}: {len(cands)} прогонов, медиана пути {med:.0f} м; '
              f'выбран {n}: путь {lengths[n]:.0f} м, RTK {100 * quality(n)[2]:.0f} %; '
              f'продлено: начало {head:.0f} м, конец {tail:.0f} м')

    # Развороты убираются в каждом направлении отдельно: на стыке направлений у конечной
    # (петля между записями не записана) разворот настоящий
    for key in directions:
        dense = resample(directions[key], STEP)
        directions[key] = remove_reversals(dense)
        print(f'Направление {key[0]}→{key[1]}: убрано разворотов линии (выбросы GNSS) — '
              f'{len(dense) - len(directions[key])} точек')
    ring = directions[(0, 1)] + directions[(1, 0)]
    points = smooth_altitude(resample(ring, STEP), ALT_SMOOTH)
    length = path_length([p[:2] for p in points]) + math.dist(points[-1][:2], points[0][:2])
    print(f'Кольцо: {len(points)} точек, длина {length:.0f} м; '
          f'стык колец: {math.dist(ring[-1][:2], ring[0][:2]):.1f} м и '
          f'{math.dist(directions[(0, 1)][-1][:2], directions[(1, 0)][0][:2]):.1f} м')

    # Проверка: расстояние точек всех прогонов до карты
    cell = 20.0
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    grid = {}
    for i in range(len(points)):
        j = (i + 1) % len(points)
        for x, y in ((xs[i], ys[i]), (xs[j], ys[j])):
            grid.setdefault((int(math.floor(x / cell)), int(math.floor(y / cell))), set()).add(i)
    dists, per_run = [], []
    for name, pts in local.items():
        d = [distance_to_polyline(x, y, xs, ys, grid, cell) for x, y, _ in pts[::10]]
        dists += d
        per_run.append((sum(1 for v in d if v > 3.0) / len(d), name))
    dists.sort()
    q = lambda p: dists[min(len(dists) - 1, int(p * len(dists)))]
    print(f'Проверка на {len(local)} прогонах ({len(dists)} точек): расстояние до карты '
          f'медиана {q(0.5):.2f} м, p95 {q(0.95):.2f} м, p99 {q(0.99):.2f} м; дальше 3 м — '
          f'{100 * sum(1 for v in dists if v > 3.0) / len(dists):.1f} % точек')
    for share, name in sorted(per_run, reverse=True)[:5]:
        if share > 0.01:
            print(f'   {name}: {100 * share:.0f} % точек дальше 3 м от карты')

    back = LocalEnu(*runs[next(iter(runs))][0][1:4])
    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Карта маршрута трамвая: замкнутое кольцо в порядке движения, шаг 1 м\n')
        f.write(f'# Собрана tools/build_route_map.py из прогонов {chosen[(0, 1)]} и {chosen[(1, 0)]} '
                f'(приёмник master)\n')
        f.write(f'# Длина кольца {length:.0f} м; проверка на {len(local)} прогонах: медиана '
                f'расстояния до карты {q(0.5):.2f} м, p95 {q(0.95):.2f} м\n')
        f.write('u,lat,lon,alt\n')
        u = 0.0
        prev = None
        for p in points:
            if prev is not None:
                u += math.dist(p[:2], prev[:2])
            lat, lon, alt = enu_to_geodetic(back, p)
            f.write(f'{u:.2f},{lat:.8f},{lon:.8f},{alt:.2f}\n')
            prev = p
    print(f'Карта записана: {args.output}')

    from tram_odometry.route_map import LocalRoute, load_route_csv
    route = LocalRoute(load_route_csv(str(args.output)), enu)
    rtk_names = [n for n in runs if n.startswith(VEHICLE)
                 and sum(1 for f in runs[n] if f[4] == 2) / len(runs[n]) > 0.9]
    validate_lengths(route, local, rtk_names)

    # Остановки: места, где трамваи стабильно стоят (для уточнения пути во время работы)
    stops = find_stops(route, runs, enu, [n for n in runs if n.startswith(VEHICLE)])
    stops_file = args.output.with_name('route_stops.csv')
    lines = ['# Остановки: места стабильной стоянки трамваев (платформы), из обучающих прогонов 30618',
             f'# Стоянка >= {STOP_MIN_TIME:.0f} с; место — если стояли в >= {STOP_MIN_RUNS} прогонах '
             f'с разбросом <= {STOP_MAX_SPREAD:.0f} м вдоль пути',
             'lat,lon,heading,spread,runs']
    for x, y, heading, u, spread, count in stops:
        lat, lon, _ = enu_to_geodetic(enu, (x, y, 0.0))
        lines.append(f'{lat:.8f},{lon:.8f},{heading:.4f},{spread:.2f},{count}')
    with open(stops_file, 'w', encoding='utf-8') as f:
        f.write(chr(10).join(lines) + chr(10))
    spreads = sorted(st[4] for st in stops)
    print(f'Остановок: {len(stops)} (разброс места: медиана {spreads[len(spreads) // 2]:.2f} м, '
          f'макс {spreads[-1]:.2f} м); средний интервал {route.length / max(len(stops), 1):.0f} м; '
          f'записаны: {stops_file}')


def enu_to_geodetic(enu: LocalEnu, point):
    """Обратное преобразование ENU → широта/долгота/высота (итерациями, точность < 1 мм)."""
    lat0, lon0, alt0 = enu.origin
    lat, lon, alt = lat0, lon0, alt0 + point[2]
    for _ in range(5):
        e, n, u = enu.to_enu(lat, lon, alt)
        m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        m_per_deg_lon = 111412.84 * math.cos(math.radians(lat))
        lat += (point[1] - n) / m_per_deg_lat
        lon += (point[0] - e) / m_per_deg_lon
        alt += point[2] - u
    return lat, lon, alt


if __name__ == '__main__':
    main()

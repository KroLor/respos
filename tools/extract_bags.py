"""Выгрузка всех rosbag2-прогонов из data/ в parquet (без установки ROS).

Результат — extracted/<bag_id>/<name>.parquet, по одному файлу на топик:
    front.parquet, rear.parquet          — t_bag, t_hdr, velocity
    cmd.parquet                          — t_bag, t_hdr, position
    master_fix.parquet, rover_fix.parquet — t_bag, t_hdr, lat, lon, alt, status, cov_e, cov_n, cov_u
    master_vel.parquet, rover_vel.parquet — t_bag, t_hdr, vx, vy, vz, wz
Плюс extracted/summary.csv со сводкой по прогонам.

t_bag — время записи в bag, t_hdr — header.stamp; оба в наносекундах (int64).

Запуск:  python tools/extract_bags.py [--data data] [--out extracted] [--only 30618_01f73500 ...]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

ROOT = Path(__file__).resolve().parent.parent
MSG_DIR = ROOT / 'tram_vehicle_msgs' / 'msg'

TOPICS = {
    '/vehicle/front_bogie_velocity': 'front',
    '/vehicle/rear_bogie_velocity': 'rear',
    '/vehicle/driver_position_cmd': 'cmd',
    '/sensing/gnss/master/fix': 'master_fix',
    '/sensing/gnss/rover/fix': 'rover_fix',
    '/sensing/gnss/master/vel': 'master_vel',
    '/sensing/gnss/rover/vel': 'rover_vel',
}


def make_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for f in MSG_DIR.glob('*.msg'):
        types.update(get_types_from_msg(f.read_text(encoding='utf-8'),
                                        f'tram_vehicle_msgs/msg/{f.stem}'))
    ts.register(types)
    return ts


def stamp_ns(h) -> int:
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def row(name: str, t: int, m) -> dict:
    r = {'t_bag': t, 't_hdr': stamp_ns(m.header)}
    if name in ('front', 'rear'):
        r['velocity'] = m.velocity
    elif name == 'cmd':
        r['position'] = m.position
    elif name.endswith('_fix'):
        c = m.position_covariance
        r.update(lat=m.latitude, lon=m.longitude, alt=m.altitude,
                 status=m.status.status, cov_e=c[0], cov_n=c[4], cov_u=c[8])
    else:  # *_vel
        v, w = m.twist.linear, m.twist.angular
        r.update(vx=v.x, vy=v.y, vz=v.z, wz=w.z)
    return r


def extract(bag: Path, out_root: Path) -> dict:
    ts = make_typestore()
    rows = {n: [] for n in TOPICS.values()}
    with AnyReader([bag], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, t, raw in reader.messages(connections=conns):
            name = TOPICS[conn.topic]
            rows[name].append(row(name, t, reader.deserialize(raw, conn.msgtype)))
        duration_s = reader.duration / 1e9

    out = out_root / bag.name
    out.mkdir(parents=True, exist_ok=True)
    summary = {'bag': bag.name, 'vehicle': bag.name.split('_')[0], 'duration_s': duration_s}
    for name, rs in rows.items():
        df = pd.DataFrame(rs)
        df.to_parquet(out / f'{name}.parquet', index=False)
        summary[f'n_{name}'] = len(df)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data')
    ap.add_argument('--out', type=Path, default=ROOT / 'extracted')
    ap.add_argument('--only', nargs='*', help='список bag_id; по умолчанию все')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    args = ap.parse_args()

    bags = sorted(p for p in args.data.iterdir() if (p / 'metadata.yaml').exists())
    if args.only:
        bags = [b for b in bags if b.name in set(args.only)]

    summaries = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(extract, b, args.out): b for b in bags}
        for i, f in enumerate(as_completed(futs), 1):
            b = futs[f]
            try:
                summaries.append(f.result())
                print(f'[{i}/{len(bags)}] {b.name}', flush=True)
            except Exception as e:  # битый прогон не должен останавливать выгрузку
                print(f'[{i}/{len(bags)}] {b.name} ОШИБКА: {e!r}', file=sys.stderr, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).sort_values('bag').to_csv(args.out / 'summary.csv', index=False)
    print(f'Готово: {len(summaries)}/{len(bags)} → {args.out}')


if __name__ == '__main__':
    main()

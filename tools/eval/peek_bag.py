"""Быстрый осмотр bag без ROS (sqlite): по каждому топику — число сообщений, частота, промежуток,
наибольший пропуск, запаздывание header.stamp относительно времени записи, диапазон значений.

  python tools/eval/peek_bag.py <каталог bag>
"""
import sqlite3
import struct
import sys
from pathlib import Path

bag = Path(sys.argv[1])
db = next(bag.glob('*.db3'))
con = sqlite3.connect(db)
topics = {tid: (name, typ) for tid, name, typ in con.execute('select id, name, type from topics')}


def header(raw):
    # CDR: 4 байта encapsulation, затем Header: int32 sec, uint32 nsec, string frame_id
    sec, nsec, flen = struct.unpack_from('<iII', raw, 4)
    frame = raw[16:16 + flen - 1].decode(errors='replace')
    return sec + nsec * 1e-9, frame, 16 + flen


t0 = None
for tid, (name, typ) in sorted(topics.items(), key=lambda x: x[1][0]):
    rows = con.execute('select timestamp, data from messages where topic_id=? order by timestamp', (tid,)).fetchall()
    if not rows:
        print(name, 'EMPTY'); continue
    ts = [r[0] * 1e-9 for r in rows]
    t0 = t0 or ts[0]
    dur = ts[-1] - ts[0]
    hz = (len(ts) - 1) / dur if dur > 0 else 0
    gaps = max(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else 0
    lags = []
    for bt, raw in rows[:: max(1, len(rows) // 200)]:
        hs, frame, _ = header(raw)
        lags.append(bt * 1e-9 - hs)
    extra = ''
    if typ.endswith('VelocitySensor'):
        vals = []
        for _, raw in rows:
            _, _, off = header(raw)
            off = 4 + ((off - 4 + 7) // 8) * 8
            vals.append(struct.unpack_from('<d', raw, off)[0])
        extra = f' v=[{min(vals):.2f}..{max(vals):.2f}]'
    if typ.endswith('DriverControllerCommand'):
        vals = []
        for _, raw in rows:
            _, _, off = header(raw)
            vals.append(struct.unpack_from('<b', raw, off)[0])
        extra = f' pos=[{min(vals)}..{max(vals)}]'
    print(f'{name:34s} n={len(ts):6d} {hz:6.2f} Hz  span={ts[0]-t0:7.1f}..{ts[-1]-t0:7.1f}s  maxgap={gaps:6.2f}s  '
          f'bag-header lag med={sorted(lags)[len(lags)//2]*1e3:8.1f} ms frame="{frame}"{extra}')

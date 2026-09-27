"""Сверка выходов реальной ROS 2 ноды (записанных `ros2 bag record`) с офлайн-стендом и метрики по ним.

PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/compare_ros.py <bag_id> <путь к записанному bag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

import metrics
from common import ROOT, events, load_run, make_estimator, reference, run_estimator


def read_outputs(path: Path) -> pd.DataFrame:
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for f in (ROOT / 'tram_vehicle_msgs' / 'msg').glob('*.msg'):
        types.update(get_types_from_msg(f.read_text(encoding='utf-8'), f'tram_vehicle_msgs/msg/{f.stem}'))
    ts.register(types)
    v, p = [], []
    with AnyReader([path], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in ("/result/velocity", "/result/position")]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic == '/result/velocity':
                v.append((st, m.velocity, t))
            elif c.topic == '/result/position':
                q = m.pose.pose.position
                p.append((st, q.x, q.y, q.z, m.header.frame_id, m.child_frame_id, m.twist.twist.linear.x,
                          m.pose.covariance[0]))
    V = pd.DataFrame(v, columns=['t', 'v', 't_rec'])
    P = pd.DataFrame(p, columns=['t', 'e', 'n', 'u', 'frame', 'child', 'twist_x', 'cov_xx'])
    return V, P


def main():
    bag, rec = sys.argv[1], Path(sys.argv[2])
    V, P = read_outputs(rec)
    run = load_run(bag)
    ref = reference(run)
    off, _ = run_estimator(make_estimator(), events(run))
    m = V.merge(P[['t', 'e', 'n', 'u', 'twist_x']], on='t', how='inner')
    j = np.searchsorted(off.t.to_numpy(), m.t.to_numpy())
    j = np.clip(j, 0, len(off) - 1)
    same = np.abs(off.t.to_numpy()[j] - m.t.to_numpy()) < 1e-6
    d = m[same].reset_index(drop=True)
    o = off.iloc[j[same]].reset_index(drop=True)
    rate = len(V) / (V.t.iloc[-1] - V.t.iloc[0])
    gaps = np.diff(np.sort(V.t.to_numpy()))
    # метрики по выходу ноды: подставляем как «оценку»
    est = pd.DataFrame({'t': d.t, 'v': d.v, 'e': d.e, 'n': d.n, 'u': d.u})
    geo = ref['frame'].to_geodetic(est.e.to_numpy(), est.n.to_numpy(), est.u.to_numpy())
    est['lat'], est['lon'] = geo[0], geo[1]
    out = {
        'bag': bag, 'n_velocity_msgs': len(V), 'n_position_msgs': len(P), 'rate_hz': round(rate, 2),
        'max_gap_s': round(float(gaps.max()), 3), 'gaps_gt_100ms': int((gaps > 0.1).sum()),
        'offline_outputs': len(off), 'matched_to_offline': int(same.sum()),
        'max_abs_dv_vs_offline': float(np.max(np.abs(d.v - o.v))) if len(d) else None,
        'max_abs_dpos_vs_offline_m': float(np.max(np.hypot(d.e - o.e, d.n - o.n))) if len(d) else None,
        'frame_id': P.frame.iloc[0], 'child_frame_id': P.child.iloc[0],
        'twist_equals_velocity': bool(np.allclose(d.v, d.twist_x)),
    }
    out.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.speed_metrics(est, ref).items()})
    pm, _ = metrics.position_metrics(est, ref)
    out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in pm.items()})
    print(json.dumps(out, indent=1, ensure_ascii=False))
    dst = ROOT / 'reports' / 'ros_runs'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{bag}.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()

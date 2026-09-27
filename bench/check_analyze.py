"""Разбор прогона официального стенда: выход ноды против /localization/kinematic_state (эталон организаторов).

PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/check_analyze.py reports/check/out
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    t = {}
    for f in (ROOT / 'tram_vehicle_msgs' / 'msg').glob('*.msg'):
        t.update(get_types_from_msg(f.read_text(encoding='utf-8'), f'tram_vehicle_msgs/msg/{f.stem}'))
    ts.register(t)
    out = {'ref': [], 'pos': [], 'vel': [], 'diag': []}
    with AnyReader([Path(path)], default_typestore=ts) as r:
        for c, tb, raw in r.messages():
            m = r.deserialize(raw, c.msgtype)
            if c.topic == '/result/diagnostics':
                kv = {x.key: x.value for x in m.status[0].values}
                out['diag'].append((tb / 1e9, kv.get('mode'), kv.get('slip'), kv.get('stop_corrections'), kv.get('map_edge')))
                continue
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic == '/localization/kinematic_state':
                p = m.pose.pose.position
                out['ref'].append((st, p.x, p.y, p.z, m.twist.twist.linear.x))
            elif c.topic == '/result/position':
                p = m.pose.pose.position
                out['pos'].append((st, p.x, p.y, p.z))
            elif c.topic == '/result/velocity':
                out['vel'].append((st, m.velocity))
    R = pd.DataFrame(out['ref'], columns=['t', 'x', 'y', 'z', 'v']).sort_values('t')
    P = pd.DataFrame(out['pos'], columns=['t', 'x', 'y', 'z']).sort_values('t')
    V = pd.DataFrame(out['vel'], columns=['t', 'v']).sort_values('t')
    Dg = pd.DataFrame(out['diag'], columns=['tb', 'mode', 'slip', 'stops', 'edge'])
    return R, P, V, Dg


def nearest(A, t, tol=0.05):
    ta = A.t.to_numpy()
    j = np.clip(np.searchsorted(ta, t), 1, len(ta) - 1)
    j = np.where(np.abs(ta[j - 1] - t) <= np.abs(ta[j] - t), j - 1, j)
    ok = np.abs(ta[j] - t) <= tol
    return j, ok


def main():
    R, P, V, Dg = load(sys.argv[1])
    t0 = R.t.iloc[0]
    j, ok = nearest(P, R.t.to_numpy())
    Rm, Pm = R[ok].reset_index(drop=True), P.iloc[j[ok]].reset_index(drop=True)
    dx, dy, dz = Pm.x - Rm.x, Pm.y - Rm.y, Pm.z - Rm.z
    # вдоль/поперёк — по направлению эталона
    hd = np.arctan2(np.gradient(Rm.y.rolling(25, center=True, min_periods=1).mean()),
                    np.gradient(Rm.x.rolling(25, center=True, min_periods=1).mean()))
    mv = Rm.v.abs() > 0.5
    hd = pd.Series(np.where(mv, hd, np.nan)).ffill().bfill().to_numpy()
    along = dx * np.cos(hd) + dy * np.sin(hd)
    cross = -dx * np.sin(hd) + dy * np.cos(hd)
    d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    jv, okv = nearest(V, R.t.to_numpy())
    ev = V.v.to_numpy()[jv[okv]] - R.v.to_numpy()[okv]
    dist = float(np.sum(np.hypot(np.diff(R.x), np.diff(R.y))))
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))  # noqa: E731
    print(f'сопоставлено: положение {ok.sum()} / {len(R)}, скорость {okv.sum()}; путь эталона {dist:.0f} м')
    print(f'скорость: RMSE {rms(ev):.3f}, bias {np.mean(ev):+.4f}, max {np.max(np.abs(ev)):.3f}')
    print(f'положение: 3D RMSE {rms(d3):.2f} max {d3.max():.2f}; along RMSE {rms(along):.2f} mean {np.mean(along):+.2f} '
          f'max {np.max(np.abs(along)):.2f}; cross RMSE {rms(cross):.2f} max {np.max(np.abs(cross)):.2f}; '
          f'z RMSE {rms(dz):.2f}; в конце {d3.iloc[-1]:.2f} м = {100 * d3.iloc[-1] / dist:.3f} % пути')
    T = pd.DataFrame({'t': Rm.t - t0, 'along': along, 'cross': cross, 'd3': d3, 'v': Rm.v})
    print('по времени (каждые 60 с):')
    print(T.iloc[::3000].round(2).to_string(index=False))
    big = T[T.d3 > 15]
    if len(big):
        print('участки с ошибкой > 15 м: с', round(big.t.iloc[0], 1), 'по', round(big.t.iloc[-1], 1), 'с; max cross', round(big.cross.abs().max(), 1))
    T.to_csv(ROOT / 'reports' / 'check' / 'errors.csv', index=False)
    Dg.to_csv(ROOT / 'reports' / 'check' / 'diag.csv', index=False)


if __name__ == '__main__':
    main()

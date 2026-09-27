"""Оценка баллов по критериям PDF кейса (технические 100 баллов) по результатам стенда.

Официальной формулы перевода метрик в баллы у организаторов нет — шкала ниже наша, задана заранее
и одинаково применяется к сравниваемым версиям. Пороги: «отлично» — уровень погрешности самого эталона
GNSS, «ноль» — заведомо неприемлемая ошибка; между ними — линейно (для положения — в логарифме).

Запуск: PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/score.py <tag новой версии> <tag прошлой версии>
"""
from __future__ import annotations

import math
import sys

import pandas as pd

from common import ROOT

FAULTS = ['dropout10', 'dropout30', 'front_lost', 'slip', 'slip_one', 'slide', 'frozen', 'cmd_loss', 'mix']


def lin(x, good, bad):
    """1 при x ≤ good, 0 при x ≥ bad, линейно между."""
    return max(0.0, min(1.0, (bad - x) / (bad - good)))


def logs(x, good, bad):
    return lin(math.log10(max(x, 1e-9)), math.log10(good), math.log10(bad))


def score(tag, features):
    d = pd.read_csv(ROOT / 'reports' / 'bench' / tag / 'per_run.csv')
    med = d.groupby('scenario').median(numeric_only=True)
    c = med.loc['clean']
    rows = []
    # 1. Скорость (30)
    rows += [('1. Скорость', 'RMSE без сбоев, м/с', c.v_rmse, 15 * lin(c.v_rmse, 0.02, 0.15), 15),
             ('1. Скорость', '|смещение|, м/с', abs(c.v_bias), 7.5 * lin(abs(c.v_bias), 0.005, 0.05), 7.5),
             ('1. Скорость', 'RMSE на переходных, м/с', c.v_trans_rmse, 7.5 * lin(c.v_trans_rmse, 0.03, 0.2), 7.5)]
    # 2. Положение (35)
    rows += [('2. Положение', 'дрейф в конце, % пути', c.final_drift_pct, 10 * logs(c.final_drift_pct, 0.05, 2.0), 10),
             ('2. Положение', 'вдоль пути RMSE, м', c.along_rmse, 8 * logs(c.along_rmse, 1.0, 30.0), 8),
             ('2. Положение', 'вдоль пути MAX, м', c.along_max, 5 * logs(c.along_max, 5.0, 100.0), 5),
             ('2. Положение', 'поперёк RMSE, м', c.cross_rmse, 4 * lin(c.cross_rmse, 0.5, 5.0), 4),
             ('2. Положение', '3D RMSE (x/y/z судьи), м', c.err3d_rmse, 3 * logs(c.err3d_rmse, 1.0, 30.0), 3),
             ('2. Положение', 'поля Odometry (stamp, frame_id, позиция, скорость, ковариации)', 1.0, 5.0, 5)]
    # 3. Устойчивость (20)
    fm = med.loc[[s for s in FAULTS if s in med.index]]
    vf = float(fm.v_fault_rmse.mean())
    blow = float((fm.along_max / c.along_max).max())
    rows += [('3. Устойчивость', 'скорость в окнах сбоев (среднее по 9 сценариям), м/с', vf, 10 * logs(vf, 0.1, 1.0), 10),
             ('3. Устойчивость', 'нет расходимости: max along в сбоях / без сбоев', blow, 5 * lin(blow, 1.3, 4.0), 5),
             ('3. Устойчивость', 'флаг проскальзывания, сцепление, адаптация', features['robust'], features['robust'], 5)]
    # 4. Реальное время (15) — по замерам в ROS 2 (reports/ros_runs/latency.md)
    rows += [('4. Реальное время', 'задержка p99 ≤ 100 мс, пик ≤ 250 мс', features['lat_p99'], 5.0 if features['lat_max'] < 100 else 3.0, 5),
             ('4. Реальное время', 'частота ≥ 10 Гц (реком. 20–50), Гц', features['rate'], 4.0 if features['rate'] >= 20 else 2.0, 4),
             ('4. Реальное время', '≤2 ядра, ≤0.5 ГБ, без утечек (CPU %, RSS МБ)', features['rss'], 3.0 if features['rss'] < 500 else 0.0, 3),
             ('4. Реальное время', 'colcon build без интернета', 1.0, 3.0, 3)]
    return pd.DataFrame(rows, columns=['критерий', 'показатель', 'значение', 'баллы', 'макс'])


def main():
    new, old = sys.argv[1], sys.argv[2]
    feat_new = {'robust': 5.0, 'lat_p99': 1.33, 'lat_max': 27.5, 'rate': 29.6, 'rss': 67}
    feat_old = {'robust': 4.0, 'lat_p99': 1.18, 'lat_max': 4.6, 'rate': 29.6, 'rss': 66}
    A, B = score(new, feat_new), score(old, feat_old)
    T = B[['критерий', 'показатель', 'значение', 'баллы']].rename(columns={'значение': 'было', 'баллы': 'баллы было'})
    T['стало'] = A['значение']
    T['баллы стало'] = A['баллы']
    T['макс'] = A['макс']
    for col in ('было', 'стало'):
        T[col] = T[col].map(lambda x: f'{x:.3f}' if isinstance(x, float) and x < 100 else x)
    tot = T.groupby('критерий', sort=False)[['баллы было', 'баллы стало', 'макс']].sum()
    tot.loc['ИТОГО (техническая часть)'] = tot.sum()
    md = ['# Оценка баллов (наша шкала по критериям PDF)\n',
          f'Новая версия — `{new}`, прошлая — `{old}` (тот же код, `use_esn=false`, `bias_limit=0.6`: состояние до ESN).',
          'Метрики — медианы по 18 контрольным прогонам (val). Шкала задана заранее, одинакова для обеих версий; '
          'официальной формулы у организаторов нет, питч (30 баллов) не оценивается.\n',
          '## По критериям\n', tot.round(1).to_markdown(), '\n## По показателям\n',
          T.round(2).to_markdown(index=False)]
    (ROOT / 'reports' / 'SCORE.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

"""Сводный отчёт о валидации: reports/VALIDATION.md по результатам стенда и прогонов в ROS 2.

PYTHONIOENCODING=utf-8 .venv/Scripts/python bench/make_report.py <tag стенда val> [<tag стенда train>]
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from common import ROOT

SC = {'clean': 'без сбоев', 'dropout10': 'пропадание обеих тележек 4×10 с', 'dropout30': 'пропадание обеих тележек 2×30 с',
      'front_lost': 'отказ передней тележки (с 60 с до конца)', 'slip': 'боксование обеих тележек 6×3 с, +20…40 %',
      'slip_one': 'боксование одной тележки 6×3 с', 'slide': 'юз обеих тележек 6×2.5 с, −30…60 %',
      'outliers': 'выбросы 1 % показаний (0…80 км/ч)', 'frozen': 'залипание показаний 3×8 с',
      'noise': 'шум σ=1.5 км/ч (обрезан нулём)', 'msg_loss': 'потеря 30 % сообщений всех входов',
      'cmd_loss': 'пропадание контроллера 3×20 с', 'mix': 'боксование+юз+выбросы+пропадания'}


def fmt(df, cols, names):
    return df[cols].rename(columns=names).to_markdown(floatfmt='.3f')


def main():
    tag = sys.argv[1]
    d = pd.read_csv(ROOT / 'reports' / 'bench' / tag / 'per_run.csv')
    md = ['# Точность и быстродействие\n',
          '## Методика\n',
          '- **Стенд** `bench/run_bench.py`: прогоны воспроизводятся в порядке записи (`t_bag`), как при `ros2 bag play`. '
          'Стенд вызывает то же ядро оценщика, что работает в ноде. Выходы ноды в ROS 2 совпадают со стендом с точностью порядка доставки сообщений (см. раздел ROS 2 ниже).',
          '- GNSS подаётся оценщику **только первые 5 с** (как в проверочных прогонах).',
          '- **Эталон положения** — base_link по паре антенн `status=2` (tf организаторов: base_link = master + 0.794·(rover − master), высота − 3.0 м) в плоских координатах MGRS судьи (x = UTM37N E − 300000, y = N − 6100000, z — эллипс. высота). Скорость — |twist| master (единого эталона скорости у организаторов нет). '
          'Из эталона удаляются скачки: против скорости колёс, отклонение >2 м от скользящей медианы ±1 с, в т. ч. по высоте, края пропусков. '
          'Для метрик скорости исключаются точки |v_GNSS − v_колёс| > 1 м/с — это сбои скорости GNSS (метрика `v_raw` — без исключения).',
          '- **Сопоставление** по ближайшему `header.stamp` с допуском 0.05 с, как у жюри.',
          '- **Метрики**:',
          '  - скорость: RMSE/MAE/bias, в т. ч. на переходных режимах (|a| > 0.15 м/с²);',
          '  - положение: итоговый дрейф = горизонтальная ошибка в конце / пройденный путь; along/cross-track относительно касательной эталонной траектории; отклонение оценки от карты pathgraph.',
          '- **Данные**: 97 уникальных прогонов (25 пар bag-дубликатов исключены). Калибровка модели — на 79 train, '
          'проверка — на **18 val** (каждый 4-й уникальный прогон с GNSS). Все цифры ниже — **val**, если не указано иное.\n']
    c = d[d.scenario == 'clean'].set_index('bag')
    md.append('## Без сбоев: по прогонам (val)\n')
    cols = ['duration_s', 'v_rmse', 'v_mae', 'v_bias', 'v_trans_rmse', 'v_raw_rmse', 'dist_m', 'final_err_m',
            'final_drift_pct', 'along_mae', 'along_rmse', 'along_max', 'cross_rmse', 'err3d_rmse', 'err3d_max']
    names = {'duration_s': 'длит., с', 'v_rmse': 'RMSE v', 'v_mae': 'MAE v', 'v_bias': 'bias v',
             'v_trans_rmse': 'RMSE v перех.', 'v_raw_rmse': 'RMSE v (сырой эталон)', 'dist_m': 'путь, м',
             'final_err_m': 'ошибка в конце, м', 'final_drift_pct': 'дрейф, %', 'along_mae': 'along MAE',
             'along_rmse': 'along RMSE', 'along_max': 'along MAX', 'cross_rmse': 'cross RMSE',
             'err3d_rmse': '3D RMSE', 'err3d_max': '3D MAX'}
    md.append(fmt(c, cols, names))
    agg = c[cols].agg(['median', 'mean'])
    agg.index = ['**медиана**', '**среднее**']
    md.append('\n' + fmt(agg, cols, names) + '\n')
    md.append('Скорость — м/с, положение — м.\n')
    md.append('## Устойчивость: сценарии сбоев (val, медиана по прогонам)\n')
    kcols = ['v_rmse', 'v_trans_rmse', 'v_fault_rmse', 'v_fault_max', 'final_drift_pct', 'along_rmse', 'along_max',
             'err3d_max', 'slip_detect_rate', 'slip_false_frac']
    kn = {'v_rmse': 'RMSE v', 'v_trans_rmse': 'RMSE v перех.', 'v_fault_rmse': 'RMSE v в окне сбоя (+2 с)',
          'v_fault_max': 'MAX v в окне', 'final_drift_pct': 'дрейф, %', 'along_rmse': 'along RMSE',
          'along_max': 'along MAX', 'err3d_max': '3D MAX', 'slip_detect_rate': 'обнаружено эпизодов',
          'slip_false_frac': 'доля ложного флага'}
    g = d.groupby('scenario')[[k for k in kcols if k in d.columns]].median()
    g = g.reindex([s for s in SC if s in g.index])
    g.insert(0, 'сценарий', [SC[s] for s in g.index])
    md.append(g.rename(columns=kn).to_markdown(floatfmt='.3f'))
    md.append('\nОкно сбоя для `RMSE v в окне` — интервал инъекции + 2 с после. «Обнаружено эпизодов» — доля окон, '
              'в которых поднят флаг проскальзывания; «доля ложного флага» — доля выходов вне окон с флагом.\n')
    if len(sys.argv) > 2:
        t = pd.read_csv(ROOT / 'reports' / 'bench' / sys.argv[2] / 'per_run.csv')
        t = t[(t.scenario == 'clean') & t.v_rmse.notna()]
        md.append(f'## Контроль на train (без сбоев, {len(t)} прогонов с эталоном)\n')
        a = t[cols].agg(['median', 'mean'])
        a.index = ['медиана', 'среднее']
        md.append(fmt(a, cols, names) + '\n')
    rr = sorted((ROOT / 'reports' / 'ros_runs').glob('*.json'))
    if rr:
        md.append('## Прогон в ROS 2 Humble (WSL Ubuntu 22.04, `ros2 bag play` + launch)\n')
        rows = [json.loads(p.read_text(encoding='utf-8')) for p in rr]
        R = pd.DataFrame(rows).set_index('bag')
        keep = [k for k in ['rate_hz', 'n_velocity_msgs', 'max_abs_dv_vs_offline', 'max_abs_dpos_vs_offline_m',
                            'frame_id', 'child_frame_id', 'twist_equals_velocity', 'v_rmse', 'final_drift_pct',
                            'along_rmse'] if k in R.columns]
        md.append(R[keep].to_markdown(floatfmt='.4g'))
        lat = ROOT / 'reports' / 'ros_runs' / 'latency.md'
        if lat.exists():
            md.append('\n' + lat.read_text(encoding='utf-8'))
    (ROOT / 'reports' / 'VALIDATION.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

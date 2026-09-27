"""Глобальный тест перед коммитом (правило проекта: коммит — только после 100 % PASS).

Запуск из Windows (Python 3.10+, numpy; WSL2 Ubuntu-22.04 с ROS 2 Humble), из корня репозитория:
    python tools/dev/global_test.py [--skip-offline]
Шаги:
 1. Экспорт рабочей копии — ровно те файлы, что попадут в коммит
    (git ls-files -co --exclude-standard) — в чистый workspace WSL ~/jury_check/ws/src/respos.
 2. Сборка и тесты «как у жюри»: только ROS 2 Humble, без сети (jury_build.sh в unshare -n).
 3. Прогоны ноды ИЗ ЭТОЙ СБОРКИ на bag: чистый, испорченный (make_faulty_bag.py),
    реальные аномалии; метрики — eval_run.py.
 4. Офлайн-оценка ядра по всем bag с GNSS (offline_eval.py, Windows).
 5. Пороги «не хуже текущего» → таблица PASS/FAIL, код выхода 0 только при 100 % PASS.
Пороги (функция checks) ужесточать после каждого улучшения.
Отчёт последнего запуска: respos_last_global_test.txt во временном каталоге.
Пользователь WSL — переменная RESPOS_WSL_USER (по умолчанию feelyon).
"""
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
WSL = r'C:\Program Files\WSL\wsl.exe'
DISTRO = 'Ubuntu-22.04'
WSL_USER = os.environ.get('RESPOS_WSL_USER', 'feelyon')
REPO_WSL = f'/mnt/{REPO.drive[0].lower()}' + REPO.as_posix()[2:]      # H:\\... → /mnt/h/...
T = f'{REPO_WSL}/tools/dev'
MSGS = f'{REPO_WSL}/Резервное позиционирование/tram_vehicle_msgs'
ENV = dict(os.environ, WSL_UTF8='1', PYTHONIOENCODING='utf-8')

RUNS_SCRIPT = r'''
source /opt/ros/humble/setup.bash
export WS=$HOME/jury_check/ws
source $WS/install/setup.bash
D="$REPO_WSL/Резервное позиционирование/data"; E=$REPO_WSL/tools/eval; C=$HOME/respos_checks/global
rm -rf $C $HOME/respos_checks/faulty_bag; mkdir -p $C
python3 $E/make_faulty_bag.py "$D"/30618_af7496f0 $HOME/respos_checks/faulty_bag > $C/faulty_bag.log 2>&1
rm -rf $HOME/respos_checks/testlike_40ffd323
python3 $T/make_testlike_bag.py "$D"/30618_40ffd323 $HOME/respos_checks/testlike_40ffd323 5 > $C/testlike_bag.log 2>&1
( export ROS_DOMAIN_ID=101
  bash $T/run_bag.sh backup_model "$D"/30618_af7496f0 $C/clean 5
  bash $T/run_bag.sh backup_model $HOME/respos_checks/faulty_bag $C/faulty 5 ) > $C/runs1.log 2>&1 &
# Реальные аномалии и положение «как у жюри»: GNSS только первые 5 с, эталон — исходный bag
ROS_DOMAIN_ID=102 bash $T/run_bag.sh backup_model $HOME/respos_checks/testlike_40ffd323 $C/real 10 > $C/runs2.log 2>&1 &
wait; sleep 1
python3 $T/eval_run.py "$D"/30618_af7496f0 $C/clean/result --json $C/clean.json > $C/clean.txt 2>/dev/null
python3 $T/eval_run.py "$D"/30618_af7496f0 $C/faulty/result $C/clean/result --json $C/faulty.json > $C/faulty.txt 2>/dev/null
python3 $T/eval_run.py "$D"/30618_40ffd323 $C/real/result --json $C/real.json > $C/real.txt 2>/dev/null
echo "ERRORS=$(cat $C/*/node.log | grep -c -E 'ERROR|Traceback')"
python3 $E/fault_eval.py "$D"/30618_af7496f0 $HOME/respos_checks/faulty_bag --json $C/fault_offline.json > $C/fault_offline.txt 2>&1
echo "ORPHANS=$(pgrep -fc '[j]ury_check/ws/install|[r]os2 bag .*respos_checks')"
echo "===JSON==="
python3 -c 'import json, os; C = os.path.expanduser("~/respos_checks/global"); print(json.dumps({k: json.load(open(f"{C}/{k}.json")) for k in ("clean", "faulty", "real", "fault_offline")}))'
'''


def wsl(script, user=None, stdin=None, timeout=1800):
    cmd = [WSL, '-d', DISTRO] + (['-u', user] if user else []) + ['--exec', 'bash', '-c', script]
    r = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, env=ENV)
    return r.returncode, r.stdout.decode('utf-8', 'replace') + r.stderr.decode('utf-8', 'replace')


def export_tree():
    """Файлы будущего коммита → tar → чистый workspace в WSL."""
    names = subprocess.run(['git', '-C', str(REPO), 'ls-files', '-co', '--exclude-standard', '-z'],
                           capture_output=True, check=True).stdout.decode('utf-8').split('\0')
    names = [n for n in names if n]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for name in names:
            tar.add(str(REPO / name), arcname=name)
    code, out = wsl('rm -rf ~/jury_check && mkdir -p ~/jury_check/ws/src/respos && '
                    'tar -xf - -C ~/jury_check/ws/src/respos && echo EXPORTED', stdin=buf.getvalue())
    ok = code == 0 and 'EXPORTED' in out
    # Модель напарника: в ветке interface её нет — берём из ветки algorithm (как будет в main)
    model = model_archive()
    if model:
        code, out = wsl('tar -xf - -C ~/jury_check/ws/src/respos && echo MODEL', stdin=model)
        ok = ok and 'MODEL' in out
    # Пакет сообщений организаторов — оригинал из датасета (своей копии в репозитории нет)
    code, out = wsl(f'cp -r "{MSGS}" ~/jury_check/ws/src/ && echo MSGS')
    return len(names), ok and 'MSGS' in out


def model_archive():
    """tar пакета tram_backup_odometry из ветки algorithm, если его нет в рабочей копии."""
    if (REPO / 'src' / 'tram_backup_odometry').exists():
        return b''
    return subprocess.run(['git', '-C', str(REPO), 'archive', 'algorithm', 'src/tram_backup_odometry'],
                          capture_output=True, check=True).stdout


def offline_env(tmp):
    """Окружение офлайн-инструментов: пакет модели из ветки algorithm, если его нет в дереве."""
    env = dict(ENV)
    model = model_archive()
    if model:
        with tarfile.open(fileobj=io.BytesIO(model)) as tar:
            tar.extractall(tmp)
        env['PYTHONPATH'] = str(Path(tmp) / 'src' / 'tram_backup_odometry')
    return env


def checks(jury_out, runs, errors, orphans, offline):
    """Список (группа, проверка, значение, порог, PASS?)."""
    res = []

    def add(group, name, value, ok, limit):
        res.append((group, name, value, limit, bool(ok)))

    # Сборка «как у жюри»
    m = re.search(r'Summary: (\d+) tests, (\d+) errors, (\d+) failures', jury_out)
    add('жюри', 'сеть при сборке недоступна', 'да' if 'СЕТЬ: недоступна' in jury_out else 'нет',
        'СЕТЬ: недоступна' in jury_out, 'да')
    add('жюри', 'rosdep: зависимости есть', 'да' if 'All system dependencies have been satisfied'
        in jury_out else 'нет', 'All system dependencies have been satisfied' in jury_out, 'да')
    add('жюри', 'colcon build: 3 пакета (msgs, модель, нода)',
        'да' if 'Summary: 3 packages finished' in jury_out else 'нет',
        'Summary: 3 packages finished' in jury_out, 'да')
    tests = f'{m.group(1)} тестов, ошибок {int(m.group(2)) + int(m.group(3))}' if m else 'нет итога'
    add('жюри', 'colcon test', tests, m and int(m.group(1)) >= 67 and m.group(2) == '0'
        and m.group(3) == '0', '≥ 67, ошибок 0')
    add('жюри', 'JURY_STATUS', 'OK' if 'JURY_STATUS=0' in jury_out else 'FAIL', 'JURY_STATUS=0' in jury_out, 'OK')

    # Прогоны ноды из сборки «как у жюри»
    add('прогоны', 'ошибок в логах ноды', errors, errors == 0, '0')
    add('прогоны', 'зависших процессов', orphans, orphans == 0, '0')
    c, f, r = runs.get('clean', {}), runs.get('faulty', {}), runs.get('real', {})
    for tag, x, rmse_max in (('чистый af7496f0', c, 0.185), ('реальный 40ffd323', r, 0.185)):
        add('прогоны', f'{tag}: невозрастающих меток', x.get('back'), x.get('back') == 0, '0')
        add('прогоны', f'{tag}: выход / сообщения контроллера', f"{x.get('out_per_cmd', 0):.3f}",
            x.get('out_per_cmd', 0) >= 0.98, '≥ 0,98')
        add('прогоны', f'{tag}: частота, Гц', f"{x.get('rate', 0):.2f}", x.get('rate', 0) >= 19.5, '≥ 19,5')
        add('прогоны', f'{tag}: RMSE скорости, м/с', f"{x.get('speed_rmse') or 99:.3f}",
            (x.get('speed_rmse') or 99) <= rmse_max, f'≤ {rmse_max:.3f}')
        add('прогоны', f'{tag}: статусов ERROR в диагностике', x.get('diag_errors'),
            x.get('diag_errors') == 0, '0')
        share = x.get('slip_ratio_count', 0) / max(x.get('n_out', 1), 1)
        add('прогоны', f'{tag}: скольжение публикуется', f'{share:.3f}', share >= 0.9, '≥ 0,9 от выхода')
        add('прогоны', f'{tag}: скольжение |s| p99', f"{x.get('slip_ratio_p99', 99):.3f}",
            x.get('slip_ratio_p99', 99) <= 0.2, '≤ 0,2')
    # Положение «как у жюри»: нода, GNSS только первые 5 с, карта маршрута
    add('прогоны', 'реальный 40ffd323 (GNSS 5 с): средняя ошибка положения, м', f"{r.get('pos_mean', 999):.2f}",
        r.get('pos_mean', 999) <= 8.0, '≤ 8,0')
    add('прогоны', 'реальный 40ffd323 (GNSS 5 с): ошибка в конце, м', f"{r.get('pos_final', 999):.2f}",
        r.get('pos_final', 999) <= 20.0, '≤ 20,0')
    add('прогоны', 'испорченный: невозрастающих меток', f.get('back'), f.get('back') == 0, '0')
    add('прогоны', 'испорченный: выход / чистый', f"{f.get('out_vs_clean', 0):.3f}",
        f.get('out_vs_clean', 0) >= 0.95, '≥ 0,95')
    add('прогоны', 'испорченный: всплески видны в скольжении, макс |s|', f"{f.get('slip_ratio_max', 0):.2f}",
        f.get('slip_ratio_max', 0) >= 0.5, '≥ 0,5')
    add('прогоны', 'испорченный: путь против чистого, %', f"{f.get('path_diff_pct', 99):+.2f}",
        abs(f.get('path_diff_pct', 99)) <= 1.5, '|…| ≤ 1,5')
    # Окна сбоев — офлайн тем же ядром в порядке записи bag (детерминированно, tools/eval/fault_eval.py)
    fo = runs.get('fault_offline', {})
    add('сбои', 'оценки без сбоев оценщика', fo.get('failures'), fo.get('failures') == 0, '0')
    add('сбои', 'путь против чистого (офлайн), %', f"{fo.get('path_diff_pct', 99):+.2f}",
        abs(fo.get('path_diff_pct', 99)) <= 1.5, '|…| ≤ 1,5')
    for w in fo.get('faults', []):
        inside, after = w['inside_max'], w['after_max']
        # Известное ограничение модели: 20 с датчик задней тележки «залип» на нуле во время манёвров
        # у депо — на троганиях модель до ~3 с верит модели привода и занижает скорость (до 1 м/с)
        lim_in, lim_after = (1.1, 0.35) if w['name'] == 'rear залип на 0' else (0.25, 0.05)
        if w['name'] != 'колёса молчат':
            add('сбои', f"{w['name']}: внутри окна, м/с", f'{inside:.3f}' if inside is not None else '-',
                inside is not None and inside <= lim_in, f'≤ {lim_in}')
        add('сбои', f"{w['name']}: через 5 с, м/с", f'{after:.3f}' if after is not None else '-',
            after is not None and after <= lim_after, f'≤ {lim_after}')

    # Офлайн: копия стенда организаторов (bag 30618_88aea4d9, эталон судьи) и датасет
    if offline is not None:
        c, o = offline['check'], offline['dataset']
        add('стенд офлайн', 'сбоев модели', c['failures'], c['failures'] == 0 and c['active'] == 'backup_model', '0')
        add('стенд офлайн', 'скорость RMSE, м/с', f"{c['velocity_rmse']:.3f}", c['velocity_rmse'] <= 0.055, '≤ 0,055')
        add('стенд офлайн', 'положение 3D RMSE, м', f"{c['rmse_3d']:.3f}", c['rmse_3d'] <= 1.30, '≤ 1,30')
        add('стенд офлайн', 'положение 3D max, м', f"{c['max_3d']:.2f}", c['max_3d'] <= 12.0, '≤ 12,0')
        add('стенд офлайн', 'ошибка в конце, м', f"{c['end_3d']:.2f}", c['end_3d'] <= 1.0, '≤ 1,0')
        add('датасет', 'прогонов с эталоном', o['bags'], o['bags'] >= 78, '≥ 78')
        add('датасет', 'сбоев модели', o['model_failures'], o['model_failures'] == 0, '0')
        add('датасет', 'RMSE скорости (все точки), м/с', f"{o['speed_rmse']:.3f}", o['speed_rmse'] <= 0.155, '≤ 0,155')
        add('датасет', 'bias скорости, м/с', f"{o['speed_bias']:+.3f}", abs(o['speed_bias']) <= 0.010, '|…| ≤ 0,010')
        add('датасет', 'положение 3D RMSE: медиана по прогонам, м', f"{o['rmse_3d_median']:.2f}",
            o['rmse_3d_median'] <= 1.50, '≤ 1,50')
        add('датасет', 'положение 3D RMSE: p90, м', f"{o['rmse_3d_p90']:.2f}", o['rmse_3d_p90'] <= 3.60, '≤ 3,60')
        add('датасет', 'положение 3D max: медиана, м', f"{o['max_3d_median']:.2f}", o['max_3d_median'] <= 8.0, '≤ 8,0')
        add('датасет', 'ошибка в конце / путь: медиана, %', f"{o['end_pct_median']:.3f}",
            o['end_pct_median'] <= 0.030, '≤ 0,030')
        add('датасет', 'частота, мин по прогонам, Гц', f"{o['rate_min']:.1f}", o['rate_min'] >= 10.0, '≥ 10')
    return res


def main():
    started = time.time()
    lines = []

    def log(text=''):
        print(text, flush=True)
        lines.append(text)

    log(f'Глобальный тест: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    count, ok = export_tree()
    log(f'1. Экспорт рабочей копии: {count} файлов — {"OK" if ok else "ОШИБКА"}')
    if not ok:
        sys.exit(1)

    code, jury_out = wsl(f"unshare -n su - {WSL_USER} -c 'bash {T}/jury_build.sh'",
                         user='root', timeout=900)
    log('2. Сборка и тесты «как у жюри» (без сети):')
    for line in jury_out.splitlines():
        if line.strip() and 'systemd user session' not in line:
            log('   ' + line)

    log('3. Прогоны ноды из сборки «как у жюри» (чистый ×5, испорченный ×5, реальный ×10)...')
    code, runs_out = wsl(f'export REPO_WSL={REPO_WSL} T={T}\n' + RUNS_SCRIPT, timeout=1800)
    errors = int(re.search(r'ERRORS=(\d+)', runs_out).group(1)) if 'ERRORS=' in runs_out else -1
    orphans = int(re.search(r'ORPHANS=(\d+)', runs_out).group(1)) if 'ORPHANS=' in runs_out else -1
    runs = {}
    if '===JSON===' in runs_out:
        try:
            runs = json.loads(runs_out.split('===JSON===', 1)[1].strip().splitlines()[0])
        except (ValueError, IndexError):
            log('   не удалось разобрать метрики прогонов:\n' + runs_out[-2000:])

    offline = None
    if '--skip-offline' not in sys.argv:
        log('4. Офлайн: копия стенда организаторов и все bag датасета (tools/eval)...')
        with tempfile.TemporaryDirectory() as tmp:
            env = offline_env(tmp)
            results = {}
            for key, script in (('check', 'check_offline.py'), ('dataset', 'offline_eval.py')):
                out_json = Path(tmp) / f'{key}.json'
                r = subprocess.run([sys.executable, str(REPO / 'tools' / 'eval' / script), '--json', str(out_json)],
                                   capture_output=True, timeout=3600, env=env, cwd=str(REPO))
                for line in (r.stdout + r.stderr).decode('utf-8', 'replace').splitlines():
                    log('   ' + line)
                if out_json.exists():
                    data = json.loads(out_json.read_text(encoding='utf-8'))
                    results[key] = data.get('summary', data)
            offline = results if len(results) == 2 else None

    log('5. Пороги:')
    results = checks(jury_out, runs, errors, orphans, offline)
    width = max(len(f'{g}: {n}') for g, n, *_ in results)
    for group, name, value, limit, ok in results:
        log(f'   {"PASS" if ok else "FAIL"}  {f"{group}: {name}":<{width}}  {str(value):>22}  (порог {limit})')
    failed = [r for r in results if not r[4]]
    if offline is None:
        log('   ВНИМАНИЕ: офлайн-оценка пропущена — это не полный глобальный тест')
    verdict = 'ВСЁ PASS — можно коммитить' if not failed and offline is not None else \
        f'FAIL: {len(failed)} из {len(results)} проверок' if failed else 'PASS без офлайн-оценки — коммитить нельзя'
    log(f'ИТОГ: {verdict}  ({len(results) - len(failed)}/{len(results)} PASS, {time.time() - started:.0f} с)')
    (Path(tempfile.gettempdir()) / 'respos_last_global_test.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sys.exit(0 if not failed and offline is not None else 1)


if __name__ == '__main__':
    main()

# Резервная одометрия трамвая — ветка `interface`

Ветка разработки ROS 2 ноды `tram_odometry` и инструментов проверки.

**Итоговое решение для жюри — в ветке [`main`](https://github.com/KroLor/tram-backup-odometry/tree/main)**:
там нода из этой ветки и модель движения из ветки
[`algorithm`](https://github.com/KroLor/tram-backup-odometry/tree/algorithm), инструкция по сборке
и запуску, описание модели и отчёт о точности.

## Что в ветке

```
src/tram_odometry/     нода: приём и проверка входов (метки времени, значения, пропуски),
                       надзор за оценщиками — основной backup_model (модель пакета
                       tram_backup_odometry) и запасной wheel_baseline, коррекция положения
                       по редким точкам GNSS в пути, публикация /result/velocity, /result/position,
                       ускорения, проскальзывания и /diagnostics; launch, параметры, 56 тестов
tools/check_stand.sh   стенд организаторов (check-code) с решением: сборка, нода, судья,
                       воспроизведение bag, ресурсы и задержка
tools/eval/            оценка без ROS: копия стенда (check_offline.py), все bag датасета
                       (offline_eval.py), окна сбоев (fault_eval.py), внесение сбоев
                       (make_faulty_bag.py), осмотр bag (peek_bag.py)
tools/dev/             глобальный тест перед каждым коммитом (Windows + WSL2 с ROS 2 Humble):
                       сборка без сети, тесты, ROS-прогоны, стенд, датасет, сбои — 62 проверки
tools/build_route_map.py  карта маршрута запасного оценщика по GNSS обучающих прогонов
docs/REPORT.md         отчёт: точность, быстродействие, устойчивость, ограничения
```

## Как ветка связана с остальными

- Пакета модели `tram_backup_odometry` здесь нет — он в ветке `algorithm`. Без него нода запускается
  на запасном оценщике `wheel_baseline` (с сообщением в логе). Глобальный тест сам берёт пакет модели
  из ветки `algorithm`, то есть проверяет ровно то, что получится в `main`.
- Изменения ноды делаются здесь и после зелёного глобального теста сливаются в `main`.

## Проверка в этой ветке

```bash
python tools/dev/global_test.py        # из корня репозитория, Windows + WSL2 Ubuntu-22.04 с ROS 2 Humble
python tools/eval/check_offline.py     # копия стенда без ROS (нужен пакет модели на PYTHONPATH)
```

Коммит — только если глобальный тест закончился строкой «ВСЁ PASS».

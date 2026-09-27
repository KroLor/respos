# Резервная одометрия трамвая (ROS 2 Humble)

Решение кейса «Резервная одометрия по модели» Хакатона Московского транспорта.
Нода `tram_odometry` в реальном времени оценивает продольную скорость и положение трамвая
по датчикам скорости тележек и положению ручки контроллера водителя и публикует
`/result/velocity` и `/result/position`. Расчёт — модель движения трамвая (пакет
`tram_backup_odometry`); GNSS нужен только для начальной выставки в первые секунды и, если
в пути изредка приходят точки, для коррекции (так разрешили организаторы).

**Результат на стенде организаторов** (`check-code`, bag `30618_88aea4d9`, 1309 с, 5,5 км,
`hackathon_solution_checker`, воспроизведение ×1):

| Метрика судьи | RMSE | max |
|---|---|---|
| Скорость, м/с | 0,051 | 0,98 |
| Положение 3D (distance), м | 1,209 | 11,50 |
| x / y / z, м | 0,944 / 0,742 / 0,137 | 8,41 / 7,84 / 2,21 |

Задержка «вход → результат» p50 0,9 мс, p99 до 2,3 мс; частота 20 Гц; CPU 3,9 % одного ядра;
память до 70 МБ. Модель `tram_backup_odometry` отдельно (её собственная нода) на том же стенде:
1,772 / 14,19 м. Подробности и сравнения — [docs/REPORT.md](docs/REPORT.md).

## Состав репозитория

```
src/
├── tram_odometry/          нода для жюри: входы → предобработка → оценщики под надзором →
│                           публикация результата, диагностика; launch, config/params.yaml, тесты
│   └── tram_odometry/estimators/backup_model.py — модель tram_backup_odometry внутри ноды
│                           + коррекция по редким точкам GNSS в пути
└── tram_backup_odometry/   модель движения: фильтр Калмана по скоростям тележек
                            с моделью привода, эхо-сеть, отбраковка проскальзываний, выставка по GNSS,
                            карта пути со стрелками и остановками; своя нода, тесты, golden-тест
tools/
├── check_stand.sh          стенд организаторов с нашим решением «как у жюри» (ROS 2)
├── eval/                   инструменты оценки без ROS: копия стенда (check_offline.py), оценка по
│                           всем bag датасета (offline_eval.py), внесение сбоев (make_faulty_bag.py),
│                           осмотр bag (peek_bag.py)
├── build_route_map.py      карта маршрута для запасного оценщика
└── *.py                    построение карты пути модели, калибровка, извлечение данных (модель)
bench/                      офлайн-стенд модели: 13 сценариев сбоев, отчёты (модель)
docs/                       REPORT.md — отчёт; MODEL.md — модель; INTEGRATION.md — встраивание ядра
reports/                    калибровка и проверка точности модели
```

История разработки сохранена: нода и инструменты — ветка `interface`, модель — ветка `algorithm`;
обе слиты в `main`, это итоговая версия.

## Как устроено

```
/vehicle/front|rear_bogie_velocity ─┐
/vehicle/driver_position_cmd ───────┼─► предобработка ─► supervisor ─┬─ backup_model (основной)
/sensing/gnss/{master,rover}/fix ───┘   (проверка,        (сбой →     │  модель tram_backup_odometry:
                                         диагностика)      запасной)  │  скорость, положение base_link
                                                                      │  в MGRS, проскальзывание
                                                                      └─ wheel_baseline (запасной)
                            ─► /result/velocity, /result/position, /result/acceleration,
                               /result/slip_detected, /result/slip_ratio, /diagnostics
```

- **Модель** (`backup_model`, подробно — [docs/MODEL.md](docs/MODEL.md)): скорость — фильтр
  Калмана, прогноз по откалиброванной тяговой/тормозной характеристике a(позиция контроллера,
  скорость) с задержкой, инерционным звеном, уклоном и эхо-сетью; коррекция — показания тележек
  (км/ч → м/с) с отбраковкой проскальзывания, юза, залипания, выбросов и расхождения тележек.
  Положение — пройденный путь по карте пути (`data/pathgraph.json`: линия, кольца, съезды
  в депо) с выбором ветки на стрелках по профилю скорости и коррекцией на местах остановок.
  Сырые сообщения передаются модели ровно по её контракту
  ([docs/INTEGRATION.md](docs/INTEGRATION.md)): тесты подтверждают, что внутри ноды она даёт те же
  выходы, что сама по себе.
- **GNSS**: выставка по двум антеннам в первые секунды (точка base_link и курс). Если в пути
  приходят редкие точки (в bag стенда — всплески по 1–3 с раз в 2–3 мин), каждая точка
  со status ≥ 2 уточняет место на карте вдоль пути фильтром Калмана; если несколько точек подряд
  лежат на другой ветке карты, трамвай переставляется на неё (ошибка выбора ветки на стрелке).
  На стенде это снижает 3D RMSE положения с 1,77 до 1,21 м.
- **Надзор**: `wheel_baseline` (скорость по тележкам) работает параллельно; при исключении,
  нечисловом или нефизичном результате модели выход берётся у него, модель перезапускается,
  положение продолжается от последней точки модели. До выставки положение не публикуется.
- **Публикация**: на каждое сообщение контроллера (~20 Гц), при его молчании — на сообщения
  колёс; `header.stamp` результата = метка входного сообщения; метки строго растут.

## Требования

- Ubuntu 22.04, ROS 2 Humble (`ros-humble-ros-base`), `python3-colcon-common-extensions`,
  `python3-numpy` (есть в ROS 2 Humble). Интернет для сборки не нужен (проверено в изолированной сети).
- Пакет сообщений организаторов `tram_vehicle_msgs` — в том же workspace (своей копии в
  репозитории нет, чтобы не было дубликата пакета рядом с пакетом стенда).

## Сборка

В стенде организаторов (`check-code`, там уже есть `src/tram_vehicle_msgs`):

```bash
cd check-code
git clone https://github.com/KroLor/respos.git /tmp/respos
cp -r /tmp/respos/src/* src/
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

В отдельном workspace — рядом с `tram_vehicle_msgs` из датасета:

```bash
mkdir -p ~/ws/src && cd ~/ws/src
git clone https://github.com/KroLor/respos.git
cp -r "<датасет>/tram_vehicle_msgs" .
cd ~/ws && source /opt/ros/humble/setup.bash && colcon build && source install/setup.bash
```

`tram_vehicle_msgs` должен содержать оба сообщения (`VelocitySensor`, `DriverControllerCommand`)
и тег `<maintainer>` — см. «Изменения файлов организаторов».

## Запуск

```bash
# терминал 1 — нода (monitor:=true — ещё замер задержки, частоты, CPU и памяти)
ros2 launch tram_odometry tram_odometry.launch.py
# терминал 2 — воспроизведение прогона
ros2 bag play <bag> --delay 3
```

`--delay 3` — пауза, чтобы подписки ноды успели обнаружить плеер (DDS): иначе первые
сообщения, в том числе GNSS первых секунд, могут потеряться. Нода от `/clock` не зависит — время
берётся из `header.stamp`, поэтому работает любая скорость воспроизведения.

Аргументы launch: `params_file:=<файл>`; `estimator:=wheel_baseline` — без модели;
`monitor:=true`; `use_sim_time:=true`.

Весь стенд организаторов одной командой (сборка, нода, судья, запись, воспроизведение, ресурсы):

```bash
bash tools/check_stand.sh <каталог check-code> 1.0 ~/stand_out monitor:=true
```

## Выходные топики

| Топик | Тип | Содержимое |
|---|---|---|
| `/result/velocity` | `tram_vehicle_msgs/msg/VelocitySensor` | продольная скорость, м/с; `header.stamp` — метка входного сообщения; `frame_id` = `base_link` |
| `/result/position` | `nav_msgs/msg/Odometry` | `pose.pose.position` — точка **base_link** (ось первой тележки на уровне рельса) в **плоских MGRS**, как у эталона судьи: x = UTM 37N восток − 300 000, y = север − 6 100 000, z — эллипсоидальная высота рельса, м; `frame_id` = `map`, `child_frame_id` = `base_link`; ориентация — курс пути; `twist.twist.linear.x` — скорость; ковариация положения — σ вдоль и поперёк пути, повёрнутые в оси x/y. До выставки по GNSS не публикуется |
| `/result/acceleration` | `geometry_msgs/msg/AccelStamped` | продольное ускорение, м/с² |
| `/result/slip_detected` | `std_msgs/msg/Bool` | проскальзывание или юз колёс (модель отбраковывает такие показания) |
| `/result/slip_ratio` | `std_msgs/msg/Float64` | относительное расхождение скорости колеса и оценки |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 1 Гц: по каждому входу — принято, отброшено с причинами, пропуски; оценщик — активный, сбои, режим модели (выставка, карта), коррекции по остановкам и GNSS, переустановки ветки, σ пути, сбой датчика; выход — частота и время обработки |

## Проверка

| Что | Как | Ожидаемо |
|---|---|---|
| Стенд организаторов | `bash tools/check_stand.sh <check-code> 1.0` | итог судьи (см. таблицу вверху), ресурсы ноды |
| То же без ROS, за секунды | `python3 tools/eval/check_offline.py [--minutes]` | те же метрики судьи по bag стенда |
| Все bag датасета | `python3 tools/eval/offline_eval.py` | скорость и положение (эталон — пара антенн GNSS) |
| Лог ноды | вывод `ros2 launch tram_odometry tram_odometry.launch.py` | при старте — основной и запасной оценщики, масштаб колёс, карта и остановки запасного оценщика; раз в 5 с — скорость, путь, активный оценщик, принято (и отброшено) по входам, опубликовано; предупреждения о сбоях данных |
| Частота | `ros2 topic hz /result/velocity` | ≈ 20 Гц |
| Задержка, CPU, память | `ros2 launch tram_odometry tram_odometry.launch.py monitor:=true` | в логе раз в 5 с строка `latency_monitor`: частота `/result/velocity` и `/result/position`, задержка «вход → результат» p50/p95/p99/max, CPU, RSS |
| Метрики судьи онлайн | `ros2 run hackathon_solution_checker metrics` (стенд `check-code`) | RMSE и max скорости и положения раз в 5 с |
| Диагностика | `ros2 topic echo /diagnostics` | см. выше |
| Устойчивость | `python3 tools/eval/make_faulty_bag.py <bag> <новый bag>` | 10 видов сбоев входных данных |
| Тесты | `colcon test && colcon test-result --verbose` | 67 тестов (56 ноды + 11 модели, включая golden) |

Нода не падает на некорректных данных: NaN, значения вне диапазона, нулевые, повторные и сбитые
метки отбрасываются с учётом в `/diagnostics`; при сбое модели выход берётся у запасного оценщика.

## Параметры

Все параметры ноды с пояснениями — [`src/tram_odometry/config/params.yaml`](src/tram_odometry/config/params.yaml),
параметры модели — [`src/tram_backup_odometry/README.md`](src/tram_backup_odometry/README.md) §5. Основные:

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `estimator` | `backup_model` | оценщик: `backup_model` — модель; `wheel_baseline` — простой по тележкам (положение в локальной ENU, для судьи не подходит); `gnss_passthrough` — только отладка |
| `gnss_correction` | `true` | коррекция по редким точкам GNSS в пути |
| `vehicle_id`, `wheel_speed_scale` | `30618`, 0,0 | масштаб колёс запасного оценщика |
| `driver_cmd_timeout` | 0,15 | молчание контроллера (с), после которого триггер публикации — колёса |
| `frame_id`, `child_frame_id` | `map`, `base_link` | системы координат результата |

## Изменения файлов организаторов

Изменены сами оригиналы (копий в репозитории нет); после правок пакет `tram_vehicle_msgs`
датасета и стенда совпадают побайтно:

1. `Резервное позиционирование/tram_vehicle_msgs/package.xml` (датасет) — добавлен тег
   `<maintainer email="maintainer@example.com">Hackathon maintainers</maintainer>`, тот же, что
   у организаторов в `check-code`. Без него `colcon build` в Humble завершается ошибкой
   «Package 'tram_vehicle_msgs' must declare at least one maintainer».
2. `check-code/src/tram_vehicle_msgs` (стенд) — добавлено сообщение
   `msg/DriverControllerCommand.msg` (из датасета) и его строка в `CMakeLists.txt`. Без него нет
   типа топика `/vehicle/driver_position_cmd`; организаторы в чате 27.09: «Можете использовать
   tram_vehicle_msgs из основного датасета. Топик driver_position_cmd будет присутствовать».

Расхождение README датасета с данными (не правка файлов): скорость колёс — в км/ч, а не в м/с.

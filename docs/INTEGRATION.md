# Встраивание расчётного ядра в другое ROS-окружение

Как перенести резервную одометрию в чужой ROS 2 рантайм так, чтобы все наши тесты проходили без изменений.

## 1. Что из чего состоит

| Слой | Файлы (`src/tram_backup_odometry/`) | Зависимости |
|---|---|---|
| **Ядро** — весь расчёт | `tram_backup_odometry/{core, estimator, model, esn, imm, pathmap, geo}.py` | Python 3.10+ (проверено: 3.10 в Humble, 3.12), `numpy`. **ROS не нужен** |
| Данные ядра | `config/model.json`, `config/esn.json`, `data/pathgraph.json`, `data/approach_tracks.json`, `data/branch_profiles.json` | — |
| Обёртка ROS 2 | `tram_backup_odometry/node.py`, `launch/backup_odometry.launch.py`, `config/params.yaml` | `rclpy`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `diagnostic_msgs`, `tram_vehicle_msgs` |
| Вспомогательные ноды для жюри | `latency_monitor.py`, `online_eval.py` | как у обёртки, для работы не нужны |
| Тесты | `test/test_estimator.py`, `test/test_golden.py`, `test/data/golden_88aea4d9.npz` | `pytest`, `numpy` |
| Сообщения | `tram_vehicle_msgs` — пакет организаторов (в репозиторий не входит, берётся из их стенда или датасета) | — |

Ядро ничего не знает о ROS. Оно получает вызовы «пришло сообщение» и возвращает оценку. Обёртка только
переводит сообщения ROS в эти вызовы и оценку обратно в сообщения. Поэтому при совмещении с другим рантаймом
ядро переносится как есть, а меняется (если нужно) только обёртка.

## 2. Контракт ядра

### Сборка

```python
from tram_backup_odometry.core import build_estimator, default_files

est = build_estimator(**default_files())                    # данные из share пакета / TRAM_ODOM_SHARE / исходников
est = build_estimator(**default_files(), overrides={'use_esn': False})     # явные параметры Params
```

`build_estimator(model_file, map_file, esn_file, overrides)`:

- берёт масштаб колёс и пути из `model.json` (калибровка);
- применяет `overrides` поверх умолчаний и калибровки;
- подключает карту и ESN, если они включены в `Params`.

Нода, тесты и офлайн-стенд собирают ядро так же. **Не собирайте `Estimator(...)` вручную:** легко потерять
калибровку `wheel_scale`, и скорость уйдёт на 0.03 %, а golden-тест упадёт.

`default_files(share=None)` ищет данные по порядку:

1. явный `share`;
2. переменная окружения `TRAM_ODOM_SHARE`;
3. каталог исходников пакета.

В установленном пакете это `get_package_share_directory('tram_backup_odometry')`.

> **Важно:** `approach_tracks.json` и `branch_profiles.json` должны лежать **в одном каталоге с `pathgraph.json`**.
> `PathMap` ищет их рядом с картой. Без них ядро молча работает хуже: нет путей выезда и выбора ветки на стрелках
> у депо. На bag стенда это +46 м максимальной ошибки. Тест `test_golden_data_loaded` это ловит.

### Входы — вызывать на каждое входное сообщение

| Вызов | Когда | Аргументы |
|---|---|---|
| `est.on_wheel(side, stamp, value)` | `/vehicle/front_bogie_velocity`, `/vehicle/rear_bogie_velocity` | `side` — `'front'` / `'rear'`; `stamp` — `header.stamp` в секундах (float); `value` — **сырое** `VelocitySensor.velocity`, **км/ч**. Не пересчитывать в м/с: масштаб 1/3.5988 внутри ядра |
| `est.on_cmd(stamp, position)` | `/vehicle/driver_position_cmd` | `position` — `int`, −15…15 |
| `est.on_fix(antenna, stamp, lat, lon, alt, status)` | `/sensing/gnss/{master,rover}/fix` | `antenna` — `'master'` / `'rover'`; `status` — `NavSatFix.status.status`. Только до выставки: после `est.gnss_done == True` подписки нужно закрыть (регламент: GNSS — только начальная выставка) |

`on_wheel` и `on_cmd` возвращают `Output` или `None` (устаревшее сообщение из очереди рекордера в начале bag).
`on_fix` ничего не возвращает.

### Выход — `Output` (`estimator.py`)

| Поле | Смысл | Куда в ROS |
|---|---|---|
| `stamp` | = stamp входного сообщения | `header.stamp` всех выходов — **ровно stamp входа**, судья сопоставляет с допуском 0.05 с |
| `v` | скорость, м/с | `/result/velocity` (`VelocitySensor.velocity`), `Odometry.twist.twist.linear.x` |
| `e`, `n`, `u` | плоские MGRS 37U CB (x = UTM E − 300 000, y = N − 6 100 000), эллипс. высота рельса; точка base_link | `Odometry.pose.pose.position` |
| `yaw` | курс, рад от оси x | `Odometry.pose.pose.orientation` |
| `pos_valid` | положение известно (была выставка) | **до `True` положение не публиковать** |
| `sigma_along`, `sigma_cross`, `var_v` | неопределённости | ковариации `Odometry` |
| `a`, `slip`, `slip_kind`, `mode`, `lat`/`lon`/`h`, `map_edge`, `map_s`, … | ускорение, проскальзывание, режим, геокоординаты, диагностика | `/result/acceleration`, `/result/slip`, `/result/geo`, `/result/diagnostics` |

Как собирать сообщения ROS (ковариации, frame_id `map` / `base_link`, один выход на stamp) — см. `node.py::_publish`.

### Правила исполнения — нарушение ломает результат без ошибок

1. **Один поток.** Ядро не потокобезопасно. Все вызовы `on_*` — из одного потока, в порядке прихода сообщений.
   В rclpy — `SingleThreadedExecutor` (по умолчанию `rclpy.spin`). С `MultiThreadedExecutor` все подписки ядра
   кладутся в одну `MutuallyExclusiveCallbackGroup`.
2. **Время — только из `header.stamp`**, не стенные часы и не время прихода. `use_sim_time` на ядро не влияет.
   Если рантайм перештамповывает сообщения (stamp = now), выходы разойдутся с эталоном судьи.
3. **Единицы входов как в топиках**: скорость колёс в км/ч, позиция контроллера — целое.
4. **Один выход на stamp.** Тележки часто приходят с одинаковым stamp. Повторный выход с тем же stamp
   не публиковать (`node.py`: `recent_stamps`).
5. **QoS входов — best effort** (глубина 100). Подписчик best effort принимает и reliable-издателей.
6. **Исключения.** Нода оборачивает вызовы в `try/except` и при ошибке делает `est.reset()` (новая выставка).
   В своей обёртке сделайте так же: падение ядра не должно ронять процесс.
7. **Новый прогон / скачок времени назад > 10 с** ядро обнаруживает само и сбрасывается. Обёртка должна снова
   открыть GNSS-подписки для выставки (`node.py::_check_gnss_release`).

## 3. Варианты совмещения

### A. Пакет целиком в их colcon workspace (рекомендуется)

Минимум изменений: наша нода — отдельный процесс в их системе.

1. Скопировать `src/tram_backup_odometry/` в `src/` их workspace. `tram_vehicle_msgs` копировать, только если
   его там нет. Если есть — оставить их версию: наш код использует только `VelocitySensor.velocity`,
   `DriverControllerCommand.position` и `header`.
2. Зависимости: `rosdep install --from-paths src --ignore-src -y` (numpy, pytest — стандартные).
3. Сборка и тесты:
   ```bash
   colcon build --packages-select tram_backup_odometry
   colcon test  --packages-select tram_backup_odometry && colcon test-result --verbose
   ```
4. Запуск — включить наш launch в их launch:
   ```python
   IncludeLaunchDescription(PythonLaunchDescriptionSource(
       os.path.join(get_package_share_directory('tram_backup_odometry'), 'launch', 'backup_odometry.launch.py')))
   ```
   Если у них другие имена топиков или namespace — `remappings` в `Node(...)` нашего launch
   (`('/vehicle/front_bogie_velocity', '/их/топик')`, …). Параметры — свой YAML поверх `config/params.yaml`
   (секция `tram_backup_odometry.ros__parameters.estimator.*`, список — `src/tram_backup_odometry/README.md` §5).
5. Если данные ядра лежат не в share пакета — параметры ноды `model_file`, `map_file`, `esn_file`.
   `approach_tracks.json` и `branch_profiles.json` должны лежать рядом с `map_file`.

### B. Ядро внутри их ноды на Python

Скопировать пакет Python `tram_backup_odometry/` (или зависеть от нашего пакета) и вызывать ядро из своей ноды:

```python
from tram_backup_odometry.core import build_estimator, default_files
from ament_index_python.packages import get_package_share_directory

class TheirNode(Node):
    def __init__(self):
        super().__init__('their_node')
        self.est = build_estimator(**default_files(get_package_share_directory('tram_backup_odometry')))
        # подписки — в одной MutuallyExclusiveCallbackGroup (или SingleThreadedExecutor)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self.on_wheel('front', m), qos_best_effort)
        ...

    def on_wheel(self, side, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        try:
            o = self.est.on_wheel(side, t, float(m.velocity))        # км/ч, без пересчёта
        except Exception:
            self.est.reset(); return
        if o is not None:
            self.publish(o, m.header.stamp)                            # stamp входа; положение — только при o.pos_valid
```

Публикацию проще всего взять из `node.py::_publish` целиком.

### C. Не-Python рантайм (C++, rclcpp, свой middleware)

- **Отдельный процесс** — наша нода как есть (вариант A), обмен через топики. Проще всего и не влияет на их
  процесс. Задержка межпроцессного DDS — доли мс, у нас бюджет 100 мс.
- **Встроенный Python** (pybind11 / `Python.h`). Держать один экземпляр `Estimator` и звать `on_*` из одного потока
  (GIL тоже гарантирует последовательность, но порядок сообщений должен сохраниться). Контракт — §2.
- Перенос ядра на C++ — отдельная работа (план развития, `docs/MODEL.md` §8). Приёмка — тот же golden-тест:
  его входы и выходы в `test/data/golden_88aea4d9.npz` (numpy, читается из C++ через cnpy).

## 4. Тесты после совмещения

Порядок: сначала ядро без ROS, потом ROS-обёртка, потом точность на стенде.

| № | Что проверяет | Команда | Ожидаемо |
|---|---|---|---|
| 1 | Ядро: физика, отбраковка, выставка, MGRS, робастность выбора ветки | `cd src/tram_backup_odometry && python -m pytest -q test` | `11 passed` (~4 с) |
| 1a | **Golden**: 50 848 входов bag стенда → те же выходы (скорость ±1e-6 м/с, положение ±1e-4 м, те же stamp и режимы), подхвачены все данные, 2 переустановки ветки, 15 коррекций по остановкам | входит в п. 1 (`test_golden.py`) | 3 теста зелёные |
| 2 | Сборка и тесты в их workspace | `colcon build --packages-select tram_backup_odometry && colcon test --packages-select tram_backup_odometry && colcon test-result --verbose` | `0 errors, 0 failures` |
| 3 | Обёртка ROS = ядро: официальный стенд организаторов в их окружении | `check-code`: наша нода + `ros2 run hackathon_solution_checker metrics` + `ros2 bag play bags/30618_88aea4d9` (в этом репозитории — `bash tools/check_stand.sh <check-code> 1.0`, нода `tram_odometry` с этим ядром) | таблица ниже |
| 4 | То же офлайн (без ROS, ~20 с) | `cd bench && python check_offline.py` (нужны `extracted_check/` и эталон `check-code/_ref.parquet`) или без подготовки данных — `python3 tools/eval/check_offline.py --no-gnss-correction` | 3D RMSE 1.77 м, max 14.24 м (второй вариант: 1.80 / 14.18 м) |
| 5 | Регрессия по всем прогонам и сценариям сбоев (нужен датасет `extracted/`) | `python bench/run_bench.py --tag <имя> --split all -j 14` на прежней и новой версии, сравнить `reports/bench/<имя>/per_run.csv` двух запусков | совпадение по всем строкам (ядро то же) |

Эталон официального стенда (`hackathon_solution_checker`, bag 30618_88aea4d9, 1309 с, 5.5 км; ROS 2 Humble,
воспроизведение ×1, 27.09.2026):

| Метрика | RMSE | max |
|---|---|---|
| Скорость, м/с | 0.056 | 0.99 |
| x, м | 1.29 | 7.01 |
| y, м | 1.19 | 12.42 |
| z, м | 0.14 | 2.21 |
| 3D (distance), м | **1.76** | **14.18** |

Офлайн-копия (п. 4) даёт то же: скорость 0.053, 3D 1.77 / 14.24. Чекер сопоставляет по-своему (n = 38 438),
отсюда разница в третьем знаке. Заметно худший RMSE в п. 3 означает, что нарушен контракт §2.

### Если тест не проходит

| Симптом | Причина |
|---|---|
| `test_golden_data_loaded`: нет `branch_profiles.json` / `approach_tracks.json` | данные не рядом с `pathgraph.json`, либо `setup.py` не установил `data/*` |
| `test_golden_data_loaded`: масштаб колёс | ядро собрано не через `build_estimator`, или подложен другой `model.json` |
| `test_golden_outputs_same`: скорость расходится на ~1e-3 и больше | изменён код ядра или данные (`model.json`, `esn.json`), либо сильно другая версия numpy |
| `test_golden_outputs_same`: положение расходится, скорость — нет | другая карта или профили стрелок |
| п. 3: RMSE скорости ≫ 0.05 | скорость колёс пересчитана в м/с до ядра, либо вызовы из нескольких потоков |
| п. 3: положение не публикуется | нет GNSS в первые секунды (по ТЗ) или закрыты GNSS-подписки до выставки |
| п. 3: судья не сопоставляет выходы | `header.stamp` выхода не равен stamp входа (перештамповка в рантайме) |
| п. 3: ошибка растёт к концу, скорость в норме | потеряны `branch_profiles.json` / карта (уход на стрелке у депо) |

## 5. Если меняется само ядро

Golden фиксирует поведение текущей версии. При **осознанном** изменении алгоритма или данных:

1. Прогнать п. 4 и п. 5 и убедиться, что стало не хуже.
2. Пересоздать golden: `python tools/make_golden.py`. Нужен выгруженный bag стенда в `extracted_check/`:
   `tools/extract_bags.py --data check-code/bags --out extracted_check`.
3. Обновить ожидаемые числа в этом документе.

Если меняется только обёртка или окружение, golden пересоздавать нельзя: он и есть проверка, что ядро то же.

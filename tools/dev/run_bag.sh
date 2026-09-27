#!/bin/bash
# Прогон ноды на одном bag с записью всех выходов (выполняется в WSL).
# Использование: run_bag.sh <оценщик> <bag> <каталог_вывода> [скорость_воспроизведения]
# Результат: <каталог>/result (bag с выходами), node.log, record.log, play.log.
# Параллельные прогоны — только с разными ROS_DOMAIN_ID.
# WS — workspace с собранной нодой (по умолчанию ~/respos_ws; глобальный тест — сборка «как у жюри»).
#
# set -m (управление заданиями): фоновые процессы получают свою группу и не игнорируют
# SIGINT, поэтому их можно остановить как по Ctrl+C — сигналом всей группе. Без этого
# launch не останавливает ноду, и она «оживает» в следующем прогоне того же домена.
set -m
source /opt/ros/humble/setup.bash
source "${WS:-$HOME/respos_ws}/install/setup.bash"
EST=${1:?оценщик}; BAG=${2:?bag}; OUT=${3:?каталог вывода}; RATE=${4:-1.0}
rm -rf "$OUT"; mkdir -p "$OUT"

ros2 launch tram_odometry tram_odometry.launch.py estimator:="$EST" > "$OUT/node.log" 2>&1 &
NODE_PID=$!
ros2 bag record -o "$OUT/result" /result/velocity /result/position /result/acceleration \
    /result/slip_detected /result/slip_ratio /diagnostics > "$OUT/record.log" 2>&1 &
REC_PID=$!
sleep 4   # нода и рекордер успевают подняться
# --delay: плеер ждёт перед публикацией, чтобы подписки успели его обнаружить (DDS),
# иначе первые секунды данных теряются
ros2 bag play "$BAG" --rate "$RATE" --delay 3 > "$OUT/play.log" 2>&1
sleep 3   # дать /diagnostics выйти после конца данных

kill -INT -- "-$REC_PID"; wait "$REC_PID"
kill -INT -- "-$NODE_PID"; wait "$NODE_PID"
sleep 1
if pgrep -f "[t]ram_odometry_node" > /dev/null; then
    echo "ВНИМАНИЕ: процесс ноды не завершился (или идёт параллельный прогон)" | tee -a "$OUT/node.log"
fi
echo "Готово: $OUT (ошибок в логе ноды: $(grep -c -E 'ERROR|Traceback' "$OUT/node.log"))"

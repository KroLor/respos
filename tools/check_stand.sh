#!/usr/bin/env bash
# Официальный стенд организаторов (check-code) с нашим решением — как у жюри, в ROS 2 Humble.
#
# Собирает чистый workspace: пакеты стенда (tram_vehicle_msgs, hackathon_solution_checker) +
# пакеты решения из src/ репозитория; запускает ноду (tram_odometry.launch.py), судью
# `ros2 run hackathon_solution_checker metrics`, запись результатов и `ros2 bag play`.
# В конце печатает итог судьи и ресурсы ноды (CPU, RSS по /proc).
#
# Использование (в WSL/Linux, из любого каталога):
#   tools/check_stand.sh <каталог check-code> [скорость=1.0] [каталог вывода] [аргументы launch...]
#   tools/check_stand.sh ~/check-code 1.0 ~/stand_out estimator:=wheel_baseline
# Переменные: REPO — репозиторий решения (по умолчанию — тот, где лежит скрипт);
#             STAND_WS — каталог workspace (по умолчанию ~/stand_ws),
#             EXTRA_SRC — дополнительные каталоги пакетов через пробел.
# ×1 bag стенда идёт ~22 мин: запускайте через nohup/setsid.
set -m
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECK="$(cd -- "${1:?укажите каталог check-code}" && pwd)"
RATE="${2:-1.0}"
OUT="${3:-$HOME/stand_out}"
shift $(( $# < 3 ? $# : 3 ))
WS="${STAND_WS:-$HOME/stand_ws}"
BAG="$(ls -d "$CHECK"/bags/*/ | head -1)"

source /opt/ros/humble/setup.bash
rm -rf "$WS" "$OUT"; mkdir -p "$WS/src" "$OUT"
cp -r "$CHECK"/src/* "$WS/src/"
for pkg in "$REPO"/src/* $EXTRA_SRC; do cp -r "$pkg" "$WS/src/"; done
find "$WS/src" -name __pycache__ -prune -exec rm -rf {} +
( cd "$WS" && colcon build > "$OUT/build.log" 2>&1 ) || { tail -20 "$OUT/build.log"; exit 1; }
source "$WS/install/setup.bash"

ros2 launch tram_odometry tram_odometry.launch.py "$@" > "$OUT/node.log" 2>&1 &
NODE=$!
ros2 run hackathon_solution_checker metrics > "$OUT/metrics.log" 2>&1 &
METRICS=$!
ros2 bag record -o "$OUT/rec" /result/velocity /result/position /diagnostics \
    /localization/kinematic_state > "$OUT/record.log" 2>&1 &
RECORD=$!
sleep 5
PID=$(pgrep -f "[t]ram_odometry_node" | head -1)
# Ресурсы ноды раз в секунду: процессорное время (utime+stime, тики) и RSS, кБ
( while kill -0 "$PID" 2>/dev/null; do
      read -r -a st < "/proc/$PID/stat"
      echo "$(date +%s.%N) $(( st[13] + st[14] )) $(awk '/VmRSS/ {print $2}' "/proc/$PID/status")"
      sleep 1
  done ) > "$OUT/proc.txt" &
MONITOR=$!
ros2 bag play "$BAG" --rate "$RATE" > "$OUT/play.log" 2>&1
sleep 3
kill -INT -- "-$METRICS"; wait "$METRICS"
kill -INT -- "-$RECORD"; wait "$RECORD"
kill -INT -- "-$NODE"; wait "$NODE"
kill "$MONITOR" 2>/dev/null; wait "$MONITOR" 2>/dev/null

echo "=== Судья (hackathon_solution_checker), bag $(basename "$BAG"), ×$RATE ==="
grep -E "metrics \[" "$OUT/metrics.log" | tail -2 | sed 's/.*\] //'
awk -v tck="$(getconf CLK_TCK)" -v cores="$(nproc)" '
    NR == 1 { t0 = $1; c0 = $2 } { t = $1; c = $2; if ($3 > rss) rss = $3 }
    END { if (t > t0) printf "Нода: CPU %.1f %% одного ядра (среднее), RSS max %.0f МБ\n",
                             100 * (c - c0) / tck / (t - t0), rss / 1024 }' "$OUT/proc.txt"
# Задержка «вход → результат» — отчёты latency_monitor раз в 5 с (при monitor:=true), без первого
grep -a 'задержка, мс' "$OUT/node.log" | tail -n +2 |
    sed 's/.*velocity *\([0-9.]*\) Гц.*p50 \([0-9.]*\) p95 \([0-9.]*\) p99 \([0-9.]*\) max \([0-9.]*\).*/\1 \2 \3 \4 \5/' |
    sort -k2 -g |
    awk '{ n++; hz += $1; p50[n] = $2; if ($4 > p99) p99 = $4; if ($5 > mx) mx = $5 }
         END { if (n) printf "Задержка: p50 %.2f мс (медиана отчётов), p99 до %.2f мс, max %.2f мс; частота %.1f Гц\n",
                             p50[int((n + 1) / 2)], p99, mx, hz / n }'
echo "Ошибок в логе ноды: $(grep -v latency_monitor "$OUT/node.log" | grep -c -E 'ERROR|Traceback')"

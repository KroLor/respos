#!/bin/bash
# Сборка и тесты «как у жюри»: чистый workspace ~/jury_check/ws (в src — ровно те файлы,
# что попадут в коммит), подключён только ROS 2 Humble, без сети.
# Запускает global_test.py так:
#   wsl -u root --exec unshare -n su - <пользователь> -c 'bash <репозиторий>/tools/dev/jury_build.sh'
# (unshare -n — отдельное сетевое пространство без интерфейсов: интернета нет).
set -o pipefail
WS=$HOME/jury_check/ws
source /opt/ros/humble/setup.bash
cd "$WS" || { echo "JURY_STATUS=1 (нет $WS)"; exit 1; }
status=0

if curl -s -m 3 -o /dev/null https://github.com; then
    echo "СЕТЬ: доступна — проверка «без интернета» не выполнена"; status=1
else
    echo "СЕТЬ: недоступна — сборка без интернета"
fi
echo "=== rosdep check"
rosdep check --from-paths src --ignore-src -r 2>&1 | tail -n 1 || status=1
echo "=== colcon build"
colcon build 2>&1 | tail -n 3 || status=1
echo "=== colcon test"
colcon test 2>&1 | tail -n 1 || status=1
colcon test-result --all 2>&1 | tail -n 1 || status=1
echo "JURY_STATUS=$status"
exit $status

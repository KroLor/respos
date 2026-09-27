"""Запуск ноды резервной одометрии трамвая.

Примеры:
  ros2 launch tram_odometry tram_odometry.launch.py
  ros2 launch tram_odometry tram_odometry.launch.py estimator:=wheel_baseline     # без модели
  ros2 launch tram_odometry tram_odometry.launch.py estimator:=gnss_passthrough   # тест конвейера
  ros2 launch tram_odometry tram_odometry.launch.py params_file:=/путь/к/params.yaml
  ros2 launch tram_odometry tram_odometry.launch.py monitor:=true   # + замер задержки, CPU, RSS
  ros2 launch tram_odometry tram_odometry.launch.py gnss_correction:=false   # GNSS только для выставки
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _create_node(context):
    overrides = {
        'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower()
        in ('true', '1'),
    }
    # Пустой аргумент estimator — оценщик берётся из файла параметров
    estimator = LaunchConfiguration('estimator').perform(context)
    if estimator:
        overrides['estimator'] = estimator
    # Пустой аргумент gnss_correction — значение из файла параметров
    gnss_correction = LaunchConfiguration('gnss_correction').perform(context)
    if gnss_correction:
        overrides['gnss_correction'] = gnss_correction.lower() in ('true', '1')
    return [Node(
        package='tram_odometry',
        executable='tram_odometry_node',
        name='tram_odometry',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file').perform(context), overrides],
    )]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('tram_odometry'), 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Файл параметров ноды'),
        DeclareLaunchArgument(
            'estimator', default_value='',
            description='Переопределить оценщик: backup_model (основной), wheel_baseline, '
                        'gnss_passthrough (тест)'),
        DeclareLaunchArgument(
            'gnss_correction', default_value='',
            description='Коррекция положения по редким точкам GNSS в пути: true / false '
                        '(false — GNSS только для начальной выставки); пусто — из файла параметров'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Брать время из /clock (ros2 bag play --clock)'),
        DeclareLaunchArgument(
            'monitor', default_value='false',
            description='Запустить замер задержки «вход → результат», частоты, CPU и RSS ноды '
                        '(latency_monitor пакета tram_backup_odometry)'),
        OpaqueFunction(function=_create_node),
        Node(package='tram_backup_odometry', executable='latency_monitor', name='latency_monitor',
             output='screen', condition=IfCondition(LaunchConfiguration('monitor'))),
    ])

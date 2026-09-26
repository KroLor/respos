"""Запуск ноды резервной одометрии трамвая.

Примеры:
  ros2 launch tram_odometry tram_odometry.launch.py
  ros2 launch tram_odometry tram_odometry.launch.py estimator:=gnss_passthrough   # тест конвейера
  ros2 launch tram_odometry tram_odometry.launch.py params_file:=/путь/к/params.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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
            description='Переопределить оценщик: wheel_baseline, gnss_passthrough (тест)'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Брать время из /clock (ros2 bag play --clock)'),
        OpaqueFunction(function=_create_node),
    ])

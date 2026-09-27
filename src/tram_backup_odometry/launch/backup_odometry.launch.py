"""Запуск ноды резервной одометрии.

ros2 launch tram_backup_odometry backup_odometry.launch.py [params_file:=...] [monitor:=true] [eval:=true]
  monitor:=true — дополнительно нода замера задержки/частоты/ресурсов;
  eval:=true    — онлайн-оценка точности по GNSS-эталону (только проверка).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tram_backup_odometry')
    params = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'params.yaml')),
        DeclareLaunchArgument('monitor', default_value='false'),
        DeclareLaunchArgument('eval', default_value='false'),
        Node(package='tram_backup_odometry', executable='odometry_node', name='tram_backup_odometry',
             output='screen', parameters=[params]),
        Node(package='tram_backup_odometry', executable='latency_monitor', name='latency_monitor',
             output='screen', condition=IfCondition(LaunchConfiguration('monitor'))),
        Node(package='tram_backup_odometry', executable='online_eval', name='online_eval',
             output='screen', condition=IfCondition(LaunchConfiguration('eval'))),
    ])

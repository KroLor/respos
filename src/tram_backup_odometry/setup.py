from glob import glob

from setuptools import setup

package_name = 'tram_backup_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/data', glob('data/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='vnek.tokrev@gmail.com',
    description='Резервная одометрия трамвая по модели',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odometry_node = tram_backup_odometry.node:main',
            'latency_monitor = tram_backup_odometry.latency_monitor:main',
            'online_eval = tram_backup_odometry.online_eval:main',
        ],
    },
)

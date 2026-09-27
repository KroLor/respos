from glob import glob

from setuptools import find_packages, setup

package_name = 'tram_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='respos team',
    maintainer_email='noreply@example.com',
    description='Резервная одометрия трамвая: оценка скорости и положения без GNSS',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tram_odometry_node = tram_odometry.node:main',
        ],
    },
)

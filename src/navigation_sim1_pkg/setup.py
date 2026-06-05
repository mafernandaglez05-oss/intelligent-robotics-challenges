from setuptools import find_packages, setup
import os
from glob import glob 


package_name = 'retomike'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob('launch/*.[pxy][yma]*')), #,py o #xml, *yaml
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mafer',
    maintainer_email='mafer@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'astar_planner= retomike.astar_planner:main',
        'rrt_planner= retomike.rrt_planner:main',
        
        'controller= retomike.controller:main',
        'odometry= retomike.odometry:main',
        
        'grid_visualizer= retomike.grid_visualizer:main',
        'rrt_visualizer= retomike.rrt_visualizer:main',
        ],
    },
)

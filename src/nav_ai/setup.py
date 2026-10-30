from setuptools import find_packages, setup

package_name = 'nav_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='63285553+switchonthenight@users.noreply.github.com',
    description='Imitation learning (CNN / simplified ACT) for camera-only slalom driving in Gazebo',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collector = nav_ai.data_collector:main',
            'drive_node = nav_ai.drive_node:main',
            'episode_collector = nav_ai.episode_collector:main',
            'act_drive_node = nav_ai.act_drive_node:main'
        ],
    },
)

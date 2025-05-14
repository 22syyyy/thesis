from setuptools import find_packages, setup

package_name = 'px4_offboard_py'

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
    maintainer='syy',
    maintainer_email='syy@todo.todo',
    description='Python-based offboard control node for PX4 and ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_control_py = px4_offboard_py.offboard_control:main',
        ],
    },
)


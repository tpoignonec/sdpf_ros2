# Copyright 2024 ICube Laboratory, University of Strasbourg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Author: Thibault Poignonec

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import LogInfo, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition  # noqa: F401
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution
)


def generate_launch_description():
    this_package_name = 'sdpf_bringup'

    # Initialize Arguments
    use_fake_hardware = LaunchConfiguration('use_fake_hardware', default='false')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')

    config_dir = PathJoinSubstitution(
        [FindPackageShare(this_package_name), 'config', 'fd']
    )

    controllers_file = PathJoinSubstitution([
        config_dir,
        'controllers.yaml'
    ])

    # Generate URDF
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [
                    config_dir,
                    'fd.config.xacro',
                ]
            ),
            ' use_fake_hardware:=', use_fake_hardware,
        ]
    )

    robot_description = {'robot_description': robot_description_content}

    # rviz
    rviz_config_file = PathJoinSubstitution(
        [config_dir, 'display_robot.rviz']
    )

    rviz_node = Node(
        package='rviz2',
        condition=IfCondition(launch_rviz),
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],  # + ['--ros-args', '--log-level', 'DEBUG'],
        parameters=[
            robot_description,
        ],
    )

    # Start robot state publisher
    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace='/',
        output='screen',
        parameters=[robot_description]
    )

    # Launch controllers
    debug_args = []  # ['--ros-args', '--log-level', 'DEBUG']

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, controllers_file],
        output='both',
        arguments=[] + debug_args,
    )

    nodes = [
        rviz_node,
        robot_state_pub_node,
        control_node
    ]

    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_force_torque_sensor_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'force_torque_sensor_broadcaster'],
        output='screen'
    )

    load_impedance_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'cartesian_vic_controller'],
        output='screen'
    )

    # Load inertia broadcaster

    load_inertia_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'fd_inertia_broadcaster'],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_constant_inertia'))
    )

    robot_inertia_diag = [0.2, 0.15, 0.15]
    end_effector_weight = 0.2

    inertia_data_str = (
        '{'
        + '"layout": {'
        + '"dim": ['
        + '{"size": 6, "stride": 36, "label": "rows"},'
        + '{"size": 6, "stride": 6, "label": "cols"}'
        + '],'
        + '"data_offset": 0'
        + '},'
        + '"data": ['
        + 'INERTIA_DIAG_X, 0.0, 0.0, 0.0, 0.0, 0.0,'
        + '0.0, INERTIA_DIAG_Y, 0.0, 0.0, 0.0, 0.0,'
        + '0.0, 0.0, INERTIA_DIAG_Z, 0.0, 0.0, 0.0,'
        + '0.0, 0.0, 0.0, 0.1, 0.0, 0.0,'
        + '0.0, 0.0, 0.0, 0.0, 0.1, 0.0,'
        + '0.0, 0.0, 0.0, 0.0, 0.0, 0.1'
        + ']'
        + '}'
        ).replace(
            'INERTIA_DIAG_X', str(
                robot_inertia_diag[0] + end_effector_weight
            )
        ).replace(
            'INERTIA_DIAG_Y', str(
                robot_inertia_diag[1] + end_effector_weight
            )
        ).replace(
            'INERTIA_DIAG_Z', str(
                robot_inertia_diag[2] + end_effector_weight
            )
        )

    constant_inertia_broadcaster = ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '--rate', '500',
            '/fd_inertia',
            'std_msgs/msg/Float64MultiArray',
            inertia_data_str
        ],
        output='log',
        condition=IfCondition(LaunchConfiguration('use_constant_inertia'))
    )

    controllers_loaders = [
        load_inertia_broadcaster,
        constant_inertia_broadcaster,
        RegisterEventHandler(
            OnProcessStart(
                target_action=constant_inertia_broadcaster,
                on_start=[
                    TimerAction(
                        period=1.0,
                        actions=[
                            LogInfo(
                                msg='constant_inertia_broadcaster ready'
                                + 'Loading impedance ctrl...'
                            ),
                            load_impedance_controller
                        ],
                        cancel_on_shutdown=True
                    ),
                ],
            ),
            condition=IfCondition(LaunchConfiguration('use_constant_inertia'))
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_inertia_broadcaster,
                on_exit=[
                    LogInfo(
                        msg='fd_inertia_broadcaster ready'
                        + 'Loading impedance ctrl...'
                    ),
                    load_impedance_controller
                ],
            ),
            condition=UnlessCondition(LaunchConfiguration('use_constant_inertia'))
        ),
        load_joint_state_broadcaster,
        load_force_torque_sensor_broadcaster,
    ]

    # Create launch description and populate
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Indicate whether robot is running with '
            'mock hardware mirroring command to its states.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_constant_inertia',
            default_value='true',
            description='Indicate whether to use a constant inertia matrix.'
            + ' By default, the inertia matrix is retrieved from the fd SDK.'
        )
    )

    return LaunchDescription(
        declared_arguments + nodes + controllers_loaders
    )

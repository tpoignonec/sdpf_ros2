# import numpy as np
# import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ========================================
    # Declare launch parameters
    # ========================================
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'pf_method',
            description='Used passivation method among "SIPF", "SIPF+", "SDPF-QP", "SDPF-PPF", "SDPF-PPF-adaptive".',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'record_bags',
            default_value='false',
            description='Run the ros2bag record process.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'bag_path',
            default_value=['rosbags/new_recordings/', LaunchConfiguration('pf_method')],
            description='Output path for the ros2bag record process.',
        )
    )

    record_bags = LaunchConfiguration('record_bags')
    record_bag_path = LaunchConfiguration('bag_path')

    # ========================================
    # Simulation global parameters
    # ========================================
    global_setting = {
        'beta_max': 100.0,
    }

    # ========================================
    # Launch rosbag
    # ========================================

    # Docs:
    #  - personal notes and comments: https://gist.github.com/tpoignonec/db9cdd14c3840fcb347e00f392688173
    #  - MCAP as default ROS bag format: https://foxglove.dev/blog/mcap-as-the-ros2-default-bag-format
    #  - storage-preset-profile: https://github.com/ros2/rosbag2/tree/rolling/rosbag2_storage_mcap/#writer-configuration

    rosbag_process = ExecuteProcess(
        cmd=[
            'ros2',
            'bag',
            'record',
            '--storage', 'mcap',
            '--all',
            '--output', record_bag_path,
            # '--storage-preset-profile', 'fastwrite',  # faster, but not recommended for storage of data...
        ],
        output="screen",
        condition=IfCondition(record_bags)
    )

    # ========================================
    # Launch nodes
    # ========================================
    nodes = []

    # Launch Maciej SIDP (using V2 storage)
    # ---------------------------------------
    nodes += Node(
        package='vic_ppf_python',
        executable='bednarczyk_node',
        name='SIPF_W2_node',
        remappings=[],
        parameters=[
            global_setting,
            {
                'passivation_function': 'bednarczyk_W2'
            },
        ],
        condition=LaunchConfigurationEquals('pf_method', 'SIPF')
    ),

    # Launch Maciej SIDP+ (using V4 storage)
    # ---------------------------------------
    nodes += Node(
        package='vic_ppf_python',
        executable='bednarczyk_node',
        name='SIPF_W2_node',
        remappings=[],
        parameters=[
            global_setting,
            {
                'passivation_function': 'bednarczyk_W4'
            },
        ],
        condition=LaunchConfigurationEquals('pf_method', 'SIPF+')
    ),

    # Launch QP SDDF
    # ---------------------------------------
    nodes += Node(
        package='vic_ppf_python',
        executable='qp_pf_node',
        name='SDPF_QP_node',
        remappings=[],
        parameters=[
            global_setting,
        ],
        condition=LaunchConfigurationEquals('pf_method', 'SDPF-QP')
    ),

    # Launch PPF SDDF, z_min = 0
    # ---------------------------------------
    nodes += Node(
        package='vic_ppf_python',
        executable='ppf_node',
        name='SDPF_PPF_node',
        remappings=[],
        parameters=[
            global_setting,
            {
                'passivation_method': 'z_lower_bound'
            },
        ],
        condition=LaunchConfigurationEquals('pf_method', 'SDPF-PPF')
    ),

    # Launch PPF SDDF, z_min adaptive
    # ---------------------------------------
    nodes += Node(
        package='vic_ppf_python',
        executable='ppf_node',
        name='SDPF_PPF_adaptive_node',
        remappings=[],
        parameters=[
            global_setting,
            {
                'passivation_method': 'z_adaptative_lower_bound',
                'tau_delay_adaptive_z_min': 3.0,
            },
        ],
        condition=LaunchConfigurationEquals('pf_method', 'SDPF-PPF-adaptive')
    ),

    # ========================================
    # Return :)
    # ========================================
    return LaunchDescription(
        declared_arguments
        + nodes
        + [
            rosbag_process,
        ]
    )

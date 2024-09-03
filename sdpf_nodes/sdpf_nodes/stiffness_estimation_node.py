# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy
from rclpy.node import Node

import numpy as np
# from scipy.spatial.transform import Rotation
# import copy

from vic_controllers.commons import CompliantFrameTrajectory
from vic_controllers.commons import MeasurementData
# from vic_controllers.math import SpsdToolbox

# from vic_msgs.msg import CompliantFrameTrajectory \
#    as CompliantFrameTrajectoryMsg
from vic_msgs.msg import CompliantFrame as CompliantFrameMsg
from vic_msgs.msg import CartesianState as CartesianStateMsg


class StiffnessEstimationNode(Node):
    def __init__(self):
        super().__init__('StiffnessEstimationNode')

        cartesian_state_topic_name = 'fd_cartesian_state'
        # 'filtered_compliant_frame'
        compliant_frame_topic_name = 'fd_compliance_frame_reference'

        # Inertia setting
        self._max_inertia_lambda = 1.0

        # Impedance traj. setting
        self._damping_ratios = np.array([0.9, 0.9, 0.9])
        self._desired_inertia_matrix = np.eye(3)*0.8

        # Setup compliant frame publisher
        self.get_logger().info('Setting up comms...')
        self._publisher_compliant_frame = self.create_publisher(
            CompliantFrameMsg,
            compliant_frame_topic_name,
            1
        )

        # Init data
        self.measurement_data = MeasurementData(dimension=3)
        self.filtered_compliance_traj = CompliantFrameTrajectory(
            dimension=3,
            trajectory_lenght=1
        )
        self.inertia_robot = None
        self._t0 = None

        # Setup measurements subscriber
        self._is_ready = False
        self._latest_robot_state_msg = None
        self._subscriber_cartesian_pose = self.create_subscription(
            CartesianStateMsg,
            cartesian_state_topic_name,
            self.callback_robot_measurements,
            1
        )

    def callback_robot_measurements(self, state_msg):
        self._latest_robot_state_msg = state_msg
        self.measurement_data.p = np.array([
            state_msg.pose.position.x,
            state_msg.pose.position.y,
            state_msg.pose.position.z
        ])
        self.measurement_data.p_dot = np.array([
            state_msg.velocity.linear.x,
            state_msg.velocity.linear.y,
            state_msg.velocity.linear.z
        ])
        self.measurement_data.f_ext = np.array([
            state_msg.wrench.force.x,
            state_msg.wrench.force.y,
            state_msg.wrench.force.z
        ])
        self.inertia_robot = np.array(
            state_msg.natural_inertia.data).reshape((6, 6)).astype(float)[:3, :3]
        return self.measurement_data

    def send_compliant_frame(self, p, K):
        # Define utils
        def package_array(nd_array):
            return nd_array.reshape([1, -1])[0].tolist()

        # Fill frame
        filtered_compliant_frame_msg = CompliantFrameMsg()
        filtered_compliant_frame_msg.header.stamp = self.get_clock().now().to_msg()
        # - Fill ref. position
        filtered_compliant_frame_msg.pose.pose.position.x = p[0]
        filtered_compliant_frame_msg.pose.pose.position.y = p[1]
        filtered_compliant_frame_msg.pose.pose.position.z = p[2]
        # - Fill inertia
        filtered_compliant_frame_msg.inertia.data = package_array(
            self._desired_inertia_matrix
        )
        # - Fill stiffness
        filtered_compliant_frame_msg.stiffness.data = package_array(
            K
        )
        # - Fill damping
        filtered_compliant_frame_msg.damping.data = package_array(
            2 * self._damping_ratios*np.sqrt(
                K * self._desired_inertia_matrix
            )
        )
        # Send frame as msg
        self._publisher_compliant_frame.publish(filtered_compliant_frame_msg)


def main(args=None):
    rclpy.init()
    import time
    node = StiffnessEstimationNode()

    def send_frame_and_wait(p_d, K, delay=2.0):
        rclpy.spin_once(node)
        node.send_compliant_frame(p_d, K)
        time.sleep(delay)
        rclpy.spin_once(node)

    p_d = np.array([0.0, 0.03, 0.03])
    K_min = np.diag([500.0, 10.0, 500.0])
    K_max = np.diag([500.0, 1000.0, 500.0])

    send_frame_and_wait(p_d, K_min, delay=3.0)

    data_f = []
    data_p = []
    for interp in np.linspace(0, 1, 100).tolist():
        K = K_min + interp * (K_max - K_min)
        send_frame_and_wait(p_d, K, delay=1.0)
        node.get_logger().info(f'K = {np.diag(K).tolist()}')

        p = node.measurement_data.p
        node.get_logger().info(f'p[1] = {p[1]}')
        f_ext = -node.measurement_data.f_ext
        node.get_logger().info(f'f_ext[1] = {f_ext[1]}')

        data_p += [p[1]]
        data_f += [f_ext[1]]

    node.get_logger().info(f'\ndata_f = {data_f}')
    node.get_logger().info(f'data_p = {data_p}')

    reg_A = np.ones((len(data_p), 2))
    reg_A[:, 1] = -np.array(data_p)

    reg_B = np.ones((len(data_p), 1))
    reg_B[:, 0] = np.array(data_f)

    print(reg_A)
    print(reg_B)

    sol = np.linalg.pinv(reg_A) @ reg_B
    print(sol)

    K_env = sol[1]
    p_0_env = sol[0]/K_env
    node.get_logger().info(f'Solution: K_env = {K_env}')
    node.get_logger().info(f'Solution: p_0_env = {p_0_env}')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(np.array(data_p), np.array(data_f))
    plt.scatter([p_0_env], [0.0], marker='o', color='red')
    linspace_p = np.linspace(-0.03, 0.03, 300)
    plt.plot(linspace_p, K_env*(p_0_env - linspace_p))
    plt.grid()
    plt.xlabel(r'$p$ (m)')
    plt.ylabel(r'$f_{ext}$ (N)')
    plt.show()

    while (rclpy.ok()):
        time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

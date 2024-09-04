# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy
from rclpy.node import Node

# from geometry_msgs.msg import Transform

import numpy as np
# from scipy.spatial.transform import Rotation
# import copy

from vic_controllers.commons import CompliantFrameTrajectory
from vic_controllers.commons import MeasurementData
# from vic_controllers.math import SpsdToolbox

from cartesian_control_msgs.msg import (
    CompliantFrameTrajectory as CompliantFrameTrajectoryMsg,
    CartesianTrajectoryPoint as CartesianTrajectoryPointMsg,
    CartesianCompliance as CartesianComplianceMsg,
    VicControllerState as VicControllerStateMsg,
    KeyValues as KeyValuesMsg
)
from std_msgs.msg import Float64 as FloatMsg


def spawn_pf_node(node):
    import time
    # from rclpy.executors import MultiThreadedExecutor

    initialized = False
    while ((not initialized) and rclpy.ok()):
        rclpy.spin_once(node)
        initialized = node.initialize()
    node.get_logger().info("Waiting 3s for the robot to get to its initial position...")
    time.sleep(3.0)
    node.start()
    try:
        rclpy.spin(node)
    except SystemExit:  # <--- process the exception
        rclpy.logging.get_logger('Quitting').info('Done')
    node.destroy_node()
    rclpy.shutdown()


class PassivityFilterNodeBase(Node):

    def __init__(
        self,
        name='PassivityFilterNode',
        dim=3,
        control_rate=500,
        fixed_M=np.eye(3)*0.2
    ):
        super().__init__(name)
        self._dim = dim
        self._control_rate = control_rate
        self._Ts = 1/control_rate
        self._t_max = 9.0

        self.declare_parameter('base_frame', 'fd_base')
        self.declare_parameter('ee_frame', 'fd_ee')
        self.declare_parameter(
            'vic_controller_name',
            'cartesian_vic_controller'
        )

        vic_controller_name = self.get_parameter('vic_controller_name').value

        vic_controller_state_topic_name = '/' + vic_controller_name + '/status'
        compliant_trajectory_topic_name = \
            '/' + vic_controller_name + '/reference_compliant_frame_trajectory'

        desired_compliance_topic_name = 'desired_compliance'
        diagnostic_topic_name = 'passivity_filter_diagnostic_data'
        simulation_time_topic_name = 'simulation_time'

        # Inertia setting
        self._max_inertia_lambda = 1.0
        if fixed_M is None:
            self._match_natural_inertia = True
        else:
            self._match_natural_inertia = False
            self._desired_inertia = fixed_M
            self._max_inertia_lambda = np.max(fixed_M)

        # Impedance traj. setting
        self._K_min_diag = np.array([10.0, 200.0, 200.0])
        self._K_max_diag = np.array([200.0, 200.0, 200.0])
        self._damping_ratios = np.array([0.1, 0.1, 0.1])
        self._D_min_diag = 2 * self._damping_ratios*np.sqrt(
            self._K_min_diag * self._max_inertia_lambda
        )
        fixed_D = True
        if fixed_D:
            self._D_max_diag = self._D_min_diag
        else:
            self._D_max_diag = 2 * self._damping_ratios*np.sqrt(
                self._max_inertia_lambda * self._K_max_diag
            )
        # Attention !!!
        # alpha = min(eig(D))/max(eig(M)) --> see "get_dummy_reference()"
        self._max_M = self._max_inertia_lambda
        self._min_d = np.min(self._D_min_diag)

        self.get_logger().info('Setting up comms...')
        # Setup compliant frame publisher
        self._publisher_vic_ref = self.create_publisher(
            CompliantFrameTrajectoryMsg,
            compliant_trajectory_topic_name,
            1
        )
        self._publisher_desired_compliant_frame = self.create_publisher(
            CartesianComplianceMsg,
            desired_compliance_topic_name,
            1
        )
        self._publisher_diagnostic = self.create_publisher(
            KeyValuesMsg,
            diagnostic_topic_name,
            5
        )
        self._publisher_simulation_time = self.create_publisher(
            FloatMsg,
            simulation_time_topic_name,
            5
        )
        # Init data
        self.measurement_data = MeasurementData(dimension=3)
        self.filtered_compliance_traj = CompliantFrameTrajectory(
            dimension=self._dim,
            trajectory_lenght=1
        )
        self.inertia_robot = None
        self._t0 = None

        # Setup measurements subscriber
        self._is_ready = False
        self._latest_vic_state_msg = None
        self._subscriber_cartesian_pose = self.create_subscription(
            VicControllerStateMsg,
            vic_controller_state_topic_name,
            self.callback_robot_measurements,
            1
        )

    def init_controller(self):
        raise NotImplementedError("Abstract class!")

    def compute_control(self):
        raise NotImplementedError("Abstract class!")

    @property
    def current_time(self):
        if (self._t0 is None):
            return 0.0
        else:
            current_t_ns = self.get_clock().now().nanoseconds
            return (current_t_ns - self._t0.nanoseconds)*(1e-9)

    def callback_robot_measurements(self, state_msg):
        self._latest_vic_state_msg = state_msg
        if (not self._is_ready):
            # RQ: once the control loop is initialized, the data processing is handled there
            self.process_measurements(self._latest_vic_state_msg)

    def process_measurements(self, state_msg):
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

    def initialize(self):
        # Init controller
        self.get_logger().info('controllersInitializing the control loop...')
        self.init_controller()
        self.get_logger().info('Controller initialized!')

        # Init logic
        self.get_logger().info('Initializing the control loop...')
        if (self._latest_vic_state_msg is None):
            return False
        if (not self._is_ready):
            self.ref_compliant_frame_traj = self.get_dummy_reference(0.0)
            self.compute_control()
            # Static target for initialization
            self.ref_compliant_frame_traj.p_dot_desired.fill(0)
            self.ref_compliant_frame_traj.p_ddot_desired.fill(0)
            self.send_filtered_compliant_frame(
                self.get_clock().now().to_msg(),
                also_publish_ref=False
            )
            self.get_logger().info('Sending initial state ref...')
            self._is_ready = True
            self.get_logger().info('initialize() -> OK')
            return True
        else:
            self.get_logger().error('initialize() -> NOK, already initialized!')
            return False

    def start(self):
        if (not self._is_ready):
            if (not self.initialize()):
                return False
        # Reset and start timer
        self.get_logger().info('Starting the control loop...')
        self._t0 = self.get_clock().now()
        self._timer = self.create_timer(1.0/float(self._control_rate), self.control_logic)
        self.get_logger().info('start() -> OK')
        return True

    def stop_control(self):
        self.get_logger().info('Stoping the control loop!')
        self.ref_compliant_frame_traj.p_dot_desired.fill(0)
        self.ref_compliant_frame_traj.p_ddot_desired.fill(0)
        self.send_filtered_compliant_frame(
            self.get_clock().now().to_msg(),
            also_publish_ref=False
        )
        raise SystemExit           # <--- here is we exit the node

    def get_dummy_reference(self, current_t):
        ref_compliant_frame_traj = CompliantFrameTrajectory(
            dimension=3,
            trajectory_lenght=self._N
        )

        def duplicate_vector(matrix): return np.repeat(
            matrix[np.newaxis, :], self._N, axis=0)

        def duplicate_matrix(matrix): return np.repeat(
            matrix[np.newaxis, :, :], self._N, axis=0)

        # Dummy desired trajectory
        '''
        penetration_dist = 0.01
        angle_support = 20 * np.pi/180.0
        p_A = np.array([
            -0.032223711649693246,
            -0.017756821923915533 - penetration_dist * np.cos(angle_support),
            0.015449580905827431 - penetration_dist * np.sin(angle_support)
        ])
        p_B = p_A + np.array([
            0.5,
            0.0,
            0.0
        ])
        '''
        p_A = np.array([
            -0.02, 0.0, 0.0
        ])

        def get_cartesian_data_point(time):
            p = p_A
            dp = np.zeros((3,))
            ddp = np.zeros((3,))
            return p, dp, ddp

        # Dummy compliance
        K_min = np.diag(self._K_min_diag)
        K_max = np.diag(self._K_max_diag)
        D_min = np.diag(self._D_min_diag)
        D_max = np.diag(self._D_max_diag)

        w = 2*np.pi/2

        def get_K_and_D(time):
            gamma = 0.5 * (1 - np.cos(w*time))
            K_d = K_min + (K_max - K_min)*gamma
            D_d = D_min + (D_max - D_min)*gamma
            return K_d, D_d

        def get_K_dot_and_D_dot(time):
            gamma_dot = w * 0.5 * np.sin(w*time)
            K_d_dot = (K_max - K_min)*gamma_dot
            D_d_dot = (D_max - D_min)*gamma_dot
            return K_d_dot, D_d_dot

        ref_compliant_frame_traj.p_desired = np.zeros((self._N, 3))
        ref_compliant_frame_traj.p_dot_desired = np.zeros((self._N, 3))
        ref_compliant_frame_traj.p_ddot_desired = np.zeros((self._N, 3))
        ref_compliant_frame_traj.K_desired = np.zeros((self._N, 3, 3))
        ref_compliant_frame_traj.D_desired = np.zeros((self._N, 3, 3))

        for stage in range(self._N):
            future_t = current_t + float(stage)*self._Ts
            # Set cartesian traj.
            p, dp, ddp = get_cartesian_data_point(future_t)
            ref_compliant_frame_traj.p_desired[stage, :] = p
            ref_compliant_frame_traj.p_dot_desired[stage, :] = dp
            ref_compliant_frame_traj.p_ddot_desired[stage, :] = ddp
            # Set K and D
            K_d, D_d = get_K_and_D(future_t)
            ref_compliant_frame_traj.K_desired[stage, :, :] = K_d
            ref_compliant_frame_traj.D_desired[stage, :, :] = D_d

        if (self._match_natural_inertia):
            ref_compliant_frame_traj.M_desired = duplicate_matrix(
                self.inertia_robot
            )
        else:
            ref_compliant_frame_traj.M_desired = duplicate_matrix(
                self._max_inertia_lambda * np.eye(3)
            )

        # ref_compliant_frame_traj.M_dot_desired = \
        #   duplicate_matrix(np.zeros((3, 3)))
        # ref_compliant_frame_traj.K_dot_desired = \
        #   duplicate_matrix(np.zeros((3, 3)))
        # ref_compliant_frame_traj.D_dot_desired = \
        #   duplicate_matrix(np.zeros((3, 3)))

        return ref_compliant_frame_traj

    def control_logic(self):
        if (not self._is_ready):
            # self.get_logger().info('Waiting for input data...')
            return

        self.process_measurements(self._latest_vic_state_msg)

        # Check for t_max
        if self.current_time > self._t_max:
            self.stop_control()

        # Dummy variable stiffness profile
        self.ref_compliant_frame_traj = self.get_dummy_reference(self.current_time)

        # compute control and fill _filtered_M_d, _filtered_D_d, _filtered_K_d, and _diagnostic_data
        if not (self.compute_control()):
            self.get_logger().error('Failled to compute controls!')

        # Send filtered reference + unfiltered ref (for logging purposes)
        timestamp = self.get_clock().now().to_msg()
        self.send_filtered_compliant_frame(timestamp, also_publish_ref=True)

        # Publish data for logging and plotting
        self.send_diagnostic_data(timestamp, self._diagnostic_data)
        simulation_time_msg = FloatMsg()
        simulation_time_msg.data = self.current_time
        self._publisher_simulation_time.publish(simulation_time_msg)

    def send_filtered_compliant_frame(self, timestamp, also_publish_ref=True):
        filtered_compliant_frame_msg = CompliantFrameTrajectoryMsg()
        filtered_compliant_frame_msg.header.stamp = timestamp

        # Get cartesian traj. point
        cartesian_trajectory_point = CartesianTrajectoryPointMsg()
        cartesian_trajectory_point.time_from_start = rclpy.duration.Duration(seconds=float(0.0)).to_msg()

        # Fill ref. position
        cartesian_trajectory_point.pose.position.x = \
            self.ref_compliant_frame_traj.get_p_desired(0)[0]
        cartesian_trajectory_point.pose.position.y = \
            self.ref_compliant_frame_traj.get_p_desired(0)[1]
        cartesian_trajectory_point.pose.position.z = \
            self.ref_compliant_frame_traj.get_p_desired(0)[2]

        # Fill ref. velocity
        cartesian_trajectory_point.velocity.linear.x = \
            self.ref_compliant_frame_traj.get_p_dot_desired(0)[0]
        cartesian_trajectory_point.velocity.linear.y = \
            self.ref_compliant_frame_traj.get_p_dot_desired(0)[1]
        cartesian_trajectory_point.velocity.linear.z = \
            self.ref_compliant_frame_traj.get_p_dot_desired(0)[2]

        # Fill ref. acc
        cartesian_trajectory_point.acceleration.linear.x = \
            self.ref_compliant_frame_traj.get_p_ddot_desired(0)[0]
        cartesian_trajectory_point.acceleration.linear.y = \
            self.ref_compliant_frame_traj.get_p_ddot_desired(0)[1]
        cartesian_trajectory_point.acceleration.linear.z = \
            self.ref_compliant_frame_traj.get_p_ddot_desired(0)[2]

        # append trajectory point
        filtered_compliant_frame_msg.cartesian_trajectory_points.append(
                cartesian_trajectory_point)

        # Get compliance point
        compliance_point = CartesianComplianceMsg()

        # Fill inertia
        def package_array(nd_array):
            array_6D = np.zeros((6, 1))
            return nd_array.reshape([1, -1])[0].tolist()

        M = np.eye(6)
        M[0:3, 0:3] = self._filtered_M_d
        compliance_point.inertia.data = package_array(M)

        # Fill stiffness
        K = np.zeros((6, 6))
        K[0:3, 0:3] = self._filtered_K_d
        compliance_point.stiffness.data = package_array(K)

        # Fill damping
        D = np.zeros((6, 6))
        D[0:3, 0:3] = self._filtered_D_d
        compliance_point.damping.data = package_array(D)


        filtered_compliant_frame_msg.compliance_at_points.append(compliance_point)
        self._publisher_vic_ref.publish(filtered_compliant_frame_msg)

        # Publish ground truth
        if (also_publish_ref):
            desired_compliant_frame_msg = CartesianComplianceMsg()
            desired_compliant_frame_msg = filtered_compliant_frame_msg.compliance_at_points[0]
            desired_compliant_frame_msg.inertia.data = package_array(
                self.ref_compliant_frame_traj.get_M_desired(0)
            )
            desired_compliant_frame_msg.stiffness.data = package_array(
                self.ref_compliant_frame_traj.get_K_desired(0)
            )
            desired_compliant_frame_msg.damping.data = package_array(
                self.ref_compliant_frame_traj.get_D_desired(0)
            )
            self._publisher_desired_compliant_frame.publish(
                desired_compliant_frame_msg
            )

    def send_diagnostic_data(self, timestamp, diagnostic_data: dict):
        msg = KeyValuesMsg()
        msg.header.stamp = timestamp
        msg.keys = []
        msg.values = []

        def append_key_value(key, value):
            msg.keys.append(key)
            msg.values.append(value)

        for key, value in diagnostic_data.items():
            append_key_value(key, value)

        self._publisher_diagnostic.publish(msg)

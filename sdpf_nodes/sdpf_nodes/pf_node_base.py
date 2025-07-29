# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy
from rclpy.node import Node

# from geometry_msgs.msg import Transform

import numpy as np
from scipy.spatial.transform import Rotation
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

from .interpolation_functions import (
    fct_sinus,
    fct_cosinus,
    fct_step,
    fct_tanh_alternating
)

def T_mat_func(euler_xyz):
    """Maps the angular velocity to the euler angle rates."""
    return np.array([
        [1,
         np.sin(euler_xyz[0]) * np.tan(euler_xyz[1]),
         np.cos(euler_xyz[0]) * np.tan(euler_xyz[1])],
        [0,
         np.cos(euler_xyz[0]),
         -np.sin(euler_xyz[0])],
        [0,
         np.sin(euler_xyz[0]) / np.cos(euler_xyz[1]),
         np.cos(euler_xyz[0]) / np.cos(euler_xyz[1])]
    ])

def T_inv_mat_func(euler_xyz):
    """Maps the euler angle rates to the angular velocity."""
    return np.array([
        [1, 0, -np.sin(euler_xyz[1])],
        [0, np.cos(euler_xyz[0]), np.sin(euler_xyz[0]) * np.cos(euler_xyz[1])],
        [0, -np.sin(euler_xyz[0]), np.cos(euler_xyz[0]) * np.cos(euler_xyz[1])]
    ])


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
        name='PassivityFilterNode'
    ):
        super().__init__(name)
        self._dim = None

        self.declare_parameter('control_rate', 200.0)
        assert self.get_parameter('control_rate').value > 0, \
            'Invalid control rate!'
        self._control_rate = self.get_parameter('control_rate').value
        self._Ts = 1/self._control_rate
        self._t_max = 0.0

        self.declare_parameter('base_frame', 'fd_base')
        self.declare_parameter('ee_frame', 'fd_ee')
        self.declare_parameter(
            'vic_controller_name',
            'cartesian_vic_controller'
        )

        # Scenario setting
        self.declare_parameter('scenario', 'impedance_ft_elastic')
        assert self.get_parameter('scenario').value in [
            'impedance_ft_elastic',
            'admittance_ur5_phri',
        ], 'Invalid scenario name!'

        self.declare_parameter(
            'trajectory_type',
            'static'
        )
        assert self.get_parameter('trajectory_type').value in [
            'static',
            'circular'
        ], 'Invalid trajectory type!'

        self.declare_parameter('interpolation_function', 'cosinus')
        assert self.get_parameter('interpolation_function').value in [
            'sinus',
            'cosinus',
            'step',
            'tanh_alternating'
        ], 'Invalid interpolation function!'

        self._trajectory_type = self.get_parameter('trajectory_type').value

        interpolation_function_name = \
            self.get_parameter('interpolation_function').value
        if interpolation_function_name == 'sinus':
            self._interpolation_function = fct_sinus
        elif interpolation_function_name == 'cosinus':
            self._interpolation_function = fct_cosinus
        elif interpolation_function_name == 'step':
            self._interpolation_function = fct_step
        elif interpolation_function_name == 'tanh_alternating':
            self._interpolation_function = fct_tanh_alternating
        else:
            raise ValueError(
                'Invalid interpolation function name: {}'.format(
                    interpolation_function_name
                )
            )

        self.get_logger().info(
            f'Scenario: {self.get_parameter("scenario").value}, '
            f'Trajectory type: {self._trajectory_type}, '
            f'Interpolation function: {
                self.get_parameter("interpolation_function").value}'
        )

        vic_controller_name = self.get_parameter('vic_controller_name').value

        vic_controller_state_topic_name = '/' + vic_controller_name + '/status'
        compliant_trajectory_topic_name = \
            '/' + vic_controller_name + '/reference_compliant_frame_trajectory'

        desired_compliance_topic_name = 'desired_compliance'
        diagnostic_topic_name = 'passivity_filter_diagnostic_data'
        simulation_time_topic_name = 'simulation_time'

        # Inertia setting
        self._match_natural_inertia = False

        # For UR5 PHRI scenario
        if (self.get_parameter('scenario').value == 'admittance_ur5_phri'):
            self._dim = 6
            self._t_max = 15
            self._period_var_impedance = self._t_max / 3  # seconds
            self._desired_inertia = np.diag(np.array([
                5.0, 5.0, 5.0,
                0.5, 0.5, 0.5
            ]))
            self._K_min_diag = np.array(
                [50.0, 50.0, 200.0, 20.0, 20.0, 20.0])
            self._K_max_diag = np.array(
                [200.0, 200.0, 200.0, 20.0, 20.0, 20.0])

            self._damping_ratios = np.array([0.3] * 6)
            self._max_inertia_lambda = np.max(self._desired_inertia)

            self._D_min_diag = 2 * self._damping_ratios * np.sqrt(
                self._K_min_diag * self._max_inertia_lambda
            )
            self._D_max_diag = self._D_min_diag.copy()
            # self._D_max_diag = 2 * self._damping_ratios * np.sqrt(
            #     self._max_inertia_lambda * self._K_max_diag
            # )
        elif (self.get_parameter('scenario').value == 'impedance_ft_elastic'):
            self._dim = 3
            self._t_max = 10
            self._period_var_impedance = self._t_max / 2  # seconds
            # Impedance traj. setting
            self._desired_inertia = np.diag(np.array(
                [0.7] * 3
            ))
            self._K_min_diag = np.array([50.0, 500.0, 500.0])
            self._K_max_diag = np.array([500.0, 500.0, 500.0])
            self._max_inertia_lambda = np.max(self._desired_inertia)
            self._damping_ratios = np.array([0.2] * 3)
            self._D_min_diag = 2 * self._damping_ratios * np.sqrt(
                self._K_min_diag * self._max_inertia_lambda
            )
            # self._D_max_diag = self._D_min_diag.copy()
            self._D_max_diag = 2 * self._damping_ratios * np.sqrt(
                self._K_max_diag * self._max_inertia_lambda
            )
        else:
            raise ValueError('Invalid scenario name! (got {})'.format(
                self.get_parameter('scenario').value
            ))

        # Attention !!!
        # alpha = min(eig(D))/max(eig(M)) --> see "get_dummy_reference()"
        self._max_M = np.max(np.diag(self._desired_inertia))
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
        assert (self._dim is not None) and (self._dim > 0), \
            'Invalid dimension!'
        self.measurement_data = MeasurementData(dimension=self._dim)
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
        raise NotImplementedError('Abstract class!')

    def compute_control(self):
        raise NotImplementedError('Abstract class!')

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
        if self._dim == 6:
            euler_xyz = Rotation.from_quat([
                state_msg.pose.orientation.x,
                state_msg.pose.orientation.y,
                state_msg.pose.orientation.z,
                state_msg.pose.orientation.w
            ]).as_euler('xyz', degrees=True)

            T_mat = T_mat_func(euler_xyz)
            T_inv_mat = T_inv_mat_func(euler_xyz)
            euler_rates = np.dot(
                T_mat,
                np.array([
                    state_msg.velocity.angular.x,
                    state_msg.velocity.angular.y,
                    state_msg.velocity.angular.z
                ])
            )
            euler_repr_torques = np.dot(
                T_inv_mat.T,
                np.array([
                    state_msg.wrench.torque.x,
                    state_msg.wrench.torque.y,
                    state_msg.wrench.torque.z
                ])
            )
            self.measurement_data.p = np.array([
                state_msg.pose.position.x,
                state_msg.pose.position.y,
                state_msg.pose.position.z,
                euler_xyz[0],
                euler_xyz[1],
                euler_xyz[2]
            ])
            self.measurement_data.p_dot = np.array([
                state_msg.velocity.linear.x,
                state_msg.velocity.linear.y,
                state_msg.velocity.linear.z,
                euler_rates[0],
                euler_rates[1],
                euler_rates[2]
            ])
            self.measurement_data.f_ext = np.array([
                state_msg.wrench.force.x,
                state_msg.wrench.force.y,
                state_msg.wrench.force.z,
                euler_repr_torques[0],
                euler_repr_torques[1],
                euler_repr_torques[2]
            ])
            self.inertia_robot = np.array(
                state_msg.natural_inertia.data).reshape((6, 6)).astype(float)
        elif self._dim == 3:
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
        else:
            raise ValueError(
                f'Invalid dimension {self._dim} for the passivity filter node!'
            )

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

    def get_cartesian_data_point(self, current_t, type='static'):
        """
        Returns a dummy cartesian data point.
        This is used to generate the reference trajectory.
        """
        if (self.get_parameter('scenario').value == 'admittance_ur5_phri'):
            center = np.array([
                0.133,
                0.52,
                0.52,
                0.0, 0.0, 0.0
            ])
            radius = 0.15  # meters
            period = self._t_max  # seconds
        elif (self.get_parameter('scenario').value == 'impedance_ft_elastic'):
            if self._trajectory_type == 'static':
                center = np.array([
                    - 0.04, 0.0, 0.0
                ])
            else:
                center = np.array([
                    0.015, 0.0, 0.0
                ])
            radius = 0.015
            period = self._t_max  # seconds
        else:
            raise ValueError('Invalid scenario name!')

        if (type == 'static'):
            p = center
            dp = np.array(center.shape[0] * [0.0])
            ddp = np.array(center.shape[0] * [0.0])
            return p, dp, ddp
        elif (type == 'circular'):
            # Circular trajectory
            angle = np.pi + 2 * np.pi * current_t / period  # 2 seconds period
            p = center + radius * np.array([
                np.cos(angle),
                np.sin(angle)
            ] + (center.shape[0] - 2) * [0.0])
            dp = radius * np.array([
                -np.sin(angle) * (2 * np.pi / period),
                np.cos(angle) * (2 * np.pi / period)
            ] + (center.shape[0] - 2) * [0.0])
            ddp = radius * np.array([
                -np.cos(angle) * (2 * np.pi / period)**2,
                -np.sin(angle) * (2 * np.pi / period)**2
            ] + (center.shape[0] - 2) * [0.0])
            return p, dp, ddp
        else:
            raise ValueError('Invalid trajectory type!')

    def get_dummy_reference(self, current_t):
        ref_compliant_frame_traj = CompliantFrameTrajectory(
            dimension=self._dim,
            trajectory_lenght=self._N
        )

        def duplicate_vector(matrix): return np.repeat(
            matrix[np.newaxis, :], self._N, axis=0)

        def duplicate_matrix(matrix): return np.repeat(
            matrix[np.newaxis, :, :], self._N, axis=0)

        # Dummy compliance
        K_min = np.diag(self._K_min_diag)
        K_max = np.diag(self._K_max_diag)
        D_min = np.diag(self._D_min_diag)
        D_max = np.diag(self._D_max_diag)

        def get_K_and_D(time):
            gamma = self._interpolation_function(
                time,
                period=self._period_var_impedance,
                delay=0,
                derivative=0
            )
            K_d = K_min + (K_max - K_min) * gamma
            D_d = D_min + (D_max - D_min) * gamma
            return K_d, D_d

        def get_K_dot_and_D_dot(time):
            gamma_dot = self._interpolation_function(
                time,
                period=self._period_var_impedance,
                delay=0,
                derivative=1
            )
            K_d_dot = (K_max - K_min) * gamma_dot
            D_d_dot = (D_max - D_min) * gamma_dot
            return K_d_dot, D_d_dot

        ref_compliant_frame_traj.p_desired = np.zeros((self._N, self._dim))
        ref_compliant_frame_traj.p_dot_desired = np.zeros((self._N, self._dim))
        ref_compliant_frame_traj.p_ddot_desired = np.zeros((self._N, self._dim))
        ref_compliant_frame_traj.K_desired = np.zeros(
            (self._N, self._dim, self._dim))
        ref_compliant_frame_traj.D_desired = np.zeros(
            (self._N, self._dim, self._dim))

        for stage in range(self._N):
            future_t = current_t + float(stage) * self._Ts
            # Set cartesian traj.
            p, dp, ddp = self.get_cartesian_data_point(future_t, type=self._trajectory_type)
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
                self._desired_inertia
            )

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
        M[0:self._dim, 0:self._dim] = self._filtered_M_d
        compliance_point.inertia.data = package_array(M)

        # Fill stiffness
        K = np.zeros((6, 6))
        K[0:self._dim, 0:self._dim] = self._filtered_K_d
        compliance_point.stiffness.data = package_array(K)

        # Fill damping
        D = np.zeros((6, 6))
        D[0:self._dim, 0:self._dim] = self._filtered_D_d
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

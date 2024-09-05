# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy
import numpy as np
import copy

from .pf_node_base import PassivityFilterNodeBase, spawn_pf_node
from vic_controllers.controllers import PpfController
from vic_controllers.math import SpsdToolbox


class PpfControlNode(PassivityFilterNodeBase):
    """
    Control node for Predictive Passivity Filter (PPF).
    """

    def __init__(self):
        super().__init__(name='PredictivePassivityFilter')
        super().declare_parameter('beta_max', 100.0)
        super().declare_parameter('N', 1)  # 10
        super().declare_parameter('passivation_method', 'z_adaptative_lower_bound')
        super().declare_parameter('tau_delay_adaptive_z_min', 1.0)

    def init_controller(self):
        # Retrieve and check node parameters
        beta_max = self.get_parameter('beta_max').value
        assert (beta_max > 0.0)
        self.get_logger().info(f'beta_max : {beta_max}')
        self.undeclare_parameter('beta_max')

        N = self.get_parameter('N').value
        assert (N > 0)
        self.get_logger().info(f'N : {N}')
        self.undeclare_parameter('N')

        tau_delay_adaptive_z_min = self.get_parameter('tau_delay_adaptive_z_min').value
        assert (tau_delay_adaptive_z_min > 0)
        self.get_logger().info(f'tau_delay_adaptive_z_min : {tau_delay_adaptive_z_min}')
        self.undeclare_parameter('tau_delay_adaptive_z_min')

        passivation_method = self.get_parameter('passivation_method').get_parameter_value().string_value
        self.get_logger().info(f'passivation_method : {passivation_method}')
        passivation_method_is_valid = (passivation_method in [
                'z_adaptative_lower_bound',
                'z_dot_lower_bound',
                'z_lower_bound',
                'none',
        ])
        assert (passivation_method_is_valid)
        self.undeclare_parameter('passivation_method')

        # Instantiate controller
        self.ppf_controller = PpfController({
            'dim': self._dim,
            'Ts': self._Ts,
            'N_horizon': N,
            'matrix_parameterization': 'diagonal',
            'beta_max': beta_max,
            'passivation_method': passivation_method,
            'tau_delay_adaptive_z_min': tau_delay_adaptive_z_min,
            'passivation_function': 'bednarczyk_W2',
            'cost_function': 'maximize_beta',
            'controller_name': 'beta_nd_ppf',
            'verbose': False,
            'N_logging': None,
            'qp_solver_iter_max': 50,
            "log_prediction_horizon": False
        })
        # * passivation methods:
        #   'z_adaptative_lower_bound'
        #   'z_dot_lower_bound'
        #   'z_lower_bound'
        #   'none'
        # * cost functions:
        #   'maximize_beta'
        #   'minimize_compliance_tracking_error'

        X_mpc_initial = None
        self.ppf_controller.set_alpha(self._min_d/self._max_M)
        print('Alpha = ', self.ppf_controller.settings['alpha'])
        self.ppf_controller.reset(X_mpc_initial)
        self._N = self.ppf_controller.settings['N_horizon']+1

        # Init data
        self.nb_iter_nmpc = 0

    def compute_control(self):
        # Profile passivation
        try:
            self.nb_iter_nmpc += 1
            self.ppf_controller.compute_control(
                self.ppf_controller.settings['Ts'],
                self.measurement_data,
                self.ref_compliant_frame_traj,
                self.inertia_robot
            )
        except Exception as e:
            self.get_logger().error(
                f'NMPC "solve()" failed at after {self.current_time} seconds! '
                + 'Error: ' + str(e)
            )
            self.get_logger().warning(
                'Reseting the NMPC solver internal state...!'
            )
            z_min = copy.deepcopy(self.ppf_controller.runtime_z_min)
            self.ppf_controller.reset(
                self.ppf_controller.nmpc_wrapper.X_current
            )
            self.ppf_controller.runtime_set_z_min(z_min)
        # Retrieve filtered impedance profile
        X_current_dict = \
            self.ppf_controller.nmpc_wrapper.x_index_map.split_data(
                self.ppf_controller.nmpc_wrapper.X_current,
                is_time_series=False
            )
        self._filtered_M_d = self.ref_compliant_frame_traj.get_M_desired(0)
        self._filtered_D_d = SpsdToolbox.create_SPSD_from_flatten(
            X_current_dict['D_flat'].reshape((-1,)),
            self.ppf_controller.model.dim,
            self.ppf_controller.model.matrix_parameterization
        )
        self._filtered_K_d = SpsdToolbox.create_SPSD_from_flatten(
            X_current_dict['K_flat'].reshape((-1,)),
            self.ppf_controller.model.dim,
            self.ppf_controller.model.matrix_parameterization
        )

        # Diagnostic data
        self._diagnostic_data = {
            "z_dot": self.ppf_controller.nmpc_wrapper.Z_pred_horizon[
                0,
                self.ppf_controller.nmpc_wrapper.z_index_map['z_dot']
            ],
            "z": self.ppf_controller.nmpc_wrapper.X_current[
                    self.ppf_controller.nmpc_wrapper.x_index_map['z']
            ],
            "z_min": self.ppf_controller.runtime_z_min,
            "beta": self.ppf_controller.nmpc_wrapper.U_pred_horizon[
                0,
                self.ppf_controller.nmpc_wrapper.u_index_map['beta']
            ],
            "CPU_time_solver":
                self.ppf_controller.nmpc_wrapper.ocp_solver.get_stats('time_tot'),
            "SQP_iter":
                self.ppf_controller.nmpc_wrapper.ocp_solver.get_stats('sqp_iter'),
            "max_QP_iter":
                np.max(
                    self.ppf_controller.nmpc_wrapper.ocp_solver.get_stats(
                        'qp_iter')),
        }
        '''
        eigenval_M, _ = np.linalg.eig(self.inertia_robot)
        append_key_value(
            "max_eig(M)",
            np.max(eigenval_M)
        )
        '''

        return True


def main(args=None):
    rclpy.init()
    node = PpfControlNode()
    spawn_pf_node(node)


if __name__ == '__main__':
    main()

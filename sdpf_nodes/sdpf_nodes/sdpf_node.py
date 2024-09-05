# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy

from .pf_node_base import PassivityFilterNodeBase, spawn_pf_node
from vic_controllers.controllers import SdpfController

'''
passivation_method: string -> Passivation method among
 - `none` (default)
 - `z_lower_bound` -> :math:`\\int_0^t w(t) \\geq z_{min}`
 - `z_adaptative_lower_bound`
        -> :math:`\\int_0^t w(t) \\geq z_{min}(t), \\,\\, z_{min}(t) \\geq 0 \\forall t`
 - `w_lower_bound` -> :math:`w(t) \\geq 0`
'''
valid_passivation_methods = [
    'none',
    'z_lower_bound',
    'z_adaptative_lower_bound',
    'w_lower_bound'
]
class SdpfNode(PassivityFilterNodeBase):

    def __init__(self):
        super().__init__(name='QpPassivityFilter')
        super().declare_parameter('verbose', False)
        super().declare_parameter('beta_max', 100.0)
        super().declare_parameter('solver', 'LP')
        super().declare_parameter('passivation_method', 'w_lower_bound')

    def init_controller(self):
        # Retrieve and check node parameters
        self._verbose = self.get_parameter('verbose').value
        self.get_logger().info(f'verbose : {self._verbose}')
        self.undeclare_parameter('verbose')

        beta_max = self.get_parameter('beta_max').value
        assert (beta_max > 0.0)
        self.get_logger().info(f'beta_max : {beta_max}')
        self.undeclare_parameter('beta_max')

        solver_type = self.get_parameter('solver').get_parameter_value().string_value
        assert (solver_type in ['QP', 'LP'])
        self.get_logger().info(f'solver_type : {solver_type}')
        self.undeclare_parameter('solver')


        passivation_method = self.get_parameter('passivation_method').get_parameter_value().string_value
        assert (passivation_method in valid_passivation_methods)
        self.get_logger().info(f'passivation_method : {passivation_method}')
        self.undeclare_parameter('passivation_method')

        tau_delay_adaptive_z_min = None
        z_max = None


        # Instantiate controller
        sdpf_args = {
            'dim': self._dim,
            'alpha': self._min_d/self._max_M,
            'beta_max': beta_max,
            'independent_beta_values' : False,
            'passivation_method' : passivation_method,
            'tau_delay_adaptive_z_min': tau_delay_adaptive_z_min,
            'filter_implementation' : solver_type,
            'verbose' : self._verbose,
            'N_logging' : None,
        }

        if (passivation_method == 'z_adaptative_lower_bound'):
            # declare extra node arg
            super().declare_parameter('tau_delay_adaptive_z_min', rclpy.Parameter.Type.DOUBLE)
            tau_delay_adaptive_z_min = \
                self.get_parameter('tau_delay_adaptive_z_min').value
            assert (tau_delay_adaptive_z_min > 0.0)
            self.get_logger().info(f'tau_delay_adaptive_z_min : {tau_delay_adaptive_z_min}')
            self.undeclare_parameter('tau_delay_adaptive_z_min')
            # append to args
            sdpf_args['tau_delay_adaptive_z_min'] = tau_delay_adaptive_z_min
        if (passivation_method == 'z_lower_bound'):
            # append to args
            super().declare_parameter('z_max', rclpy.Parameter.Type.DOUBLE)
            z_max = self.get_parameter('z_max').value
            assert (z_max > 0.0)
            self.get_logger().info(f'z_max : {z_max}')
            self.undeclare_parameter('z_max')
            sdpf_args['z_max'] = z_max


        self._controller_SDPF = SdpfController(sdpf_args)

        print('Alpha = ', self._controller_SDPF.settings['alpha'])
        self._controller_SDPF.reset()
        self._N = 1

        # Init data
        self.nb_iter_nmpc = 0

    def compute_control(self):
        # Profile passivation
        try:
            self.nb_iter_nmpc += 1
            self._controller_SDPF.compute_control(
                self._Ts,
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
            # z_min = 0.0
            # TODO(tpoignonec): variable z ?

        # Retrieve filtered impedance profile
        self._filtered_M_d = self._controller_SDPF.controller_log['M'][0, :, :]
        self._filtered_D_d = self._controller_SDPF.controller_log['D'][0, :, :]
        self._filtered_K_d = self._controller_SDPF.controller_log['K'][0, :, :]

        # Additional diagnostic data
        self._diagnostic_data = {
            "z_dot": self._controller_SDPF.controller_log['z_dot'][0],
            "z": self._controller_SDPF.controller_log['z'][0],
            "z_min": 0.0,
            "beta": self._controller_SDPF.controller_log['beta'][0],
            "storage_V": self._controller_SDPF.controller_log['V'][0],
        }
        '''
        append_key_value(
            "CPU_time_solver",
            self._controller_SDPF.nmpc_wrapper.ocp_solver.get_stats(
                'time_tot')[0]
        )
        '''
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
    node = SdpfNode()
    spawn_pf_node(node)


if __name__ == '__main__':
    main()

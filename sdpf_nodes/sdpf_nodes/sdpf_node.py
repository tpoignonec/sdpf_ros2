# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy

from .pf_node_base import PassivityFilterNodeBase, spawn_pf_node
from vic_controllers.controllers import QpPfController


class SdpfNode(PassivityFilterNodeBase):

    def __init__(self):
        super().__init__(name='QpPassivityFilter')
        super().declare_parameter('beta_max', 100.0)
        super().declare_parameter('solver', 'QP')

    def init_controller(self):
        # Retrieve and check node parameters
        beta_max = self.get_parameter('beta_max').value
        assert (beta_max > 0.0)
        self.get_logger().info(f'beta_max : {beta_max}')
        self.undeclare_parameter('beta_max')

        solver_type = self.get_parameter('solver').get_parameter_value().string_value
        assert (solver_type in ['QP', 'NLP'])
        self.get_logger().info(f'solver_type : {solver_type}')
        self.undeclare_parameter('solver')

        # Instantiate controller
        self.qp_pf_controller = QpPfController({
            'dim': self._dim,
            'alpha': self._min_d/self._max_M,
            'beta_max': beta_max,
            'passivation_function': 'bednarczyk_W2',
            'filter_implementation': solver_type,
            'verbose': False,
            'N_logging': None,
        })

        print('Alpha = ', self.qp_pf_controller.settings['alpha'])
        self.qp_pf_controller.reset()
        self._N = 1

        # Init data
        self.nb_iter_nmpc = 0

    def compute_control(self):
        # Profile passivation
        try:
            self.nb_iter_nmpc += 1
            self.qp_pf_controller.compute_control(
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
        self._filtered_M_d = self.qp_pf_controller.controller_log['M'][0, :, :]
        self._filtered_D_d = self.qp_pf_controller.controller_log['D'][0, :, :]
        self._filtered_K_d = self.qp_pf_controller.controller_log['K'][0, :, :]

        # Additional diagnostic data
        self._diagnostic_data = {
            "z_dot": self.qp_pf_controller.controller_log['z_dot'][0],
            "z": self.qp_pf_controller.controller_log['z'][0],
            "z_min": 0.0,
            "beta": self.qp_pf_controller.controller_log['beta'][0],
            "storage_V": self.qp_pf_controller.controller_log['V'][0],
        }
        '''
        append_key_value(
            "CPU_time_solver",
            self.qp_pf_controller.nmpc_wrapper.ocp_solver.get_stats(
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

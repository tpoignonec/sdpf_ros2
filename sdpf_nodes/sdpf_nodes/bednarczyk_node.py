# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

import rclpy
import numpy as np

from .pf_node_base import PassivityFilterNodeBase, spawn_pf_node
from vic_controllers.controllers import Bednarczyk2020


class BednarczykControlNode(PassivityFilterNodeBase):

    def __init__(self):
        super().__init__(name='BednarczykControlNode')
        super().declare_parameter('passivation_function', 'bednarczyk_W2')
        super().declare_parameter('beta_max', 100.0)

    def init_controller(self):
        # Check the scenario is valid
        assert (not self._match_natural_inertia)
        assert (np.all(self._desired_inertia == np.diag(self._desired_inertia.diagonal())))

        # Check node parameters
        passivation_function = self.get_parameter('passivation_function').get_parameter_value().string_value
        assert (passivation_function in ['bednarczyk_W2', 'bednarczyk_W4'])
        self.get_logger().info(f'passivation_function : {passivation_function}')
        self.undeclare_parameter('passivation_function')

        beta_max = self.get_parameter('beta_max').value
        assert (beta_max > 0.0)
        self.get_logger().info(f'beta_max : {beta_max}')
        self.undeclare_parameter('beta_max')

        # Instantiate controller
        self.pf_controller = Bednarczyk2020({
            'dim': self._dim,
            'beta': beta_max,
            'passivation_function': passivation_function,
            'verbose': False,
            'N_logging': None,
            'M': self._desired_inertia,
            'K_max': np.diag(self._K_max_diag),
            'K_min': np.diag(self._K_min_diag),
            'D_max': np.diag(self._D_max_diag),
            'D_min': np.diag(self._D_min_diag)
        })
        self.pf_controller.reset()
        self._N = 1

        # Extra verbose
        if passivation_function == 'bednarczyk_W2':
            self.get_logger().info(
                f'Using "bednarczyk_W2" with alpha values: {self.pf_controller.get_alpha_values()}')

    def compute_control(self):
        # Profile passivation
        try:
            self.pf_controller.compute_control(
                self._Ts,
                self.measurement_data,
                self.ref_compliant_frame_traj,
                self.inertia_robot
            )
        except Exception as e:
            self.get_logger().error(
                f'compute_control() failed at after {self.current_time} seconds! '
                + 'Error: ' + str(e)
            )
            raise

        # Retrieve filtered impedance profile
        self._filtered_M_d = np.diag(
            self.pf_controller.controller_log['M_diag'][0, :])
        self._filtered_D_d = np.diag(
            self.pf_controller.controller_log['D_diag'][0, :])
        self._filtered_K_d = np.diag(
            self.pf_controller.controller_log['K_diag'][0, :])

        # Additional diagnostic data
        self._diagnostic_data = {
            "gamma_X": self.pf_controller.controller_log['gamma'][0, 0],
            "gamma_target_X": self.pf_controller.controller_log['gamma_target'][0, 0],
            "gamma_dot_X": self.pf_controller.controller_log['gamma_dot'][0, 0],
            "gamma_Y": self.pf_controller.controller_log['gamma'][0, 1],
            "gamma_target_Y": self.pf_controller.controller_log['gamma_target'][0, 1],
            "gamma_dot_Y": self.pf_controller.controller_log['gamma_dot'][0, 1],
            "gamma_Z": self.pf_controller.controller_log['gamma'][0, 2],
            "gamma_target_Z": self.pf_controller.controller_log['gamma_target'][0, 2],
            "gamma_dot_Z": self.pf_controller.controller_log['gamma_dot'][0, 2],
            "storage_V": self.pf_controller.controller_log['V2'][0, 0],
        }

        return True


def main(args=None):
    rclpy.init()
    node = BednarczykControlNode()
    spawn_pf_node(node)


if __name__ == '__main__':
    main()

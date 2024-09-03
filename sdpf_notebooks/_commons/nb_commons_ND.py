from tqdm import tqdm
import numpy as np
# import casadi as ca
# import scipy.linalg
# from copy import deepcopy
from collections.abc import Callable

from vic_controllers.commons import ControllerBase
from vic_controllers.simulation import build_simulator, export_linear_mass_model
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory


def simulate_controller(
    controller: ControllerBase,
    simulation_data: dict,
    N_horizon_controller: int = 0,
    callback: Callable[[], any] = None
) -> (np.ndarray, np.ndarray):
    # Build simulator
    linear_mass_model = export_linear_mass_model(
        inertia=simulation_data['robot_inertia_real'],
        f_ext_as_param=False,  # TODO: add force
        dim=simulation_data['dim']
    )
    linear_mass_model_integrator = build_simulator(
        linear_mass_model,
        time_step=simulation_data['Ts']
    )

    # Prepare output data vector
    nx = linear_mass_model.x.size()[0]
    nu = linear_mass_model.u.size()[0]

    simX = np.full((simulation_data['N'], nx), np.nan)
    simU = np.full((simulation_data['N'], nu), np.nan)

    # Call controller in loop:
    simX[0, :] = np.concatenate([
        simulation_data['p_at_t0'],
        simulation_data['p_dot_at_t0']
    ])
    linear_mass_model_integrator.set("seed_adj", np.ones((nx, 1)))
    for i in tqdm(range(simulation_data['N'])):
        # Receding horizon at the end...
        if (i <= simulation_data['N'] - N_horizon_controller - 1):
            actual_idx_end_horizon = i + N_horizon_controller + 1
        else:
            actual_idx_end_horizon = simulation_data['N']
        range_idx_horizon = slice(i, actual_idx_end_horizon)

        # Package measurements
        measurements = MeasurementData(dimension=simulation_data['dim'])
        measurements.p = simX[i, :simulation_data['dim']].reshape(-1,)
        measurements.p_dot = simX[i, simulation_data['dim']:].reshape(-1,)
        measurements.f_ext = simulation_data['f_ext'][i].reshape(-1,)

        # Package compliant frame reference trajectory for current horizon
        ref_compliant_frame_traj = CompliantFrameTrajectory(
            dimension=simulation_data['dim'],
            trajectory_lenght=actual_idx_end_horizon - i
        )
        ref_compliant_frame_traj.p_desired = np.asarray(simulation_data['p_desired'][range_idx_horizon])
        ref_compliant_frame_traj.p_dot_desired = np.asarray(simulation_data['p_dot_desired'][range_idx_horizon])
        ref_compliant_frame_traj.p_ddot_desired = np.asarray(simulation_data['p_dot2_desired'][range_idx_horizon])
        ref_compliant_frame_traj.M_desired = np.asarray(simulation_data['M_d'][range_idx_horizon])
        ref_compliant_frame_traj.K_desired = np.asarray(simulation_data['K_d'][range_idx_horizon])
        ref_compliant_frame_traj.D_desired = np.asarray(simulation_data['D_d'][range_idx_horizon])
        ref_compliant_frame_traj.M_dot_desired = np.asarray(simulation_data['M_dot_d'][range_idx_horizon])
        ref_compliant_frame_traj.K_dot_desired = np.asarray(simulation_data['K_dot_d'][range_idx_horizon])
        ref_compliant_frame_traj.D_dot_desired = np.asarray(simulation_data['D_dot_d'][range_idx_horizon])

        # Compute control input
        simU[i, :] = controller.compute_control(
            simulation_data['Ts'],
            measurements,
            ref_compliant_frame_traj,
            simulation_data['robot_inertia_model']
        ).reshape((-1,))

        # Simulate time step
        linear_mass_model_integrator.set("x", simX[i, :])
        linear_mass_model_integrator.set("u", simU[i, :])
        status = linear_mass_model_integrator.solve()
        if status != 0:
            raise Exception(f'acados returned status {status}.')
        # Save state (at k+1)
        if (i < simulation_data['N'] - 1):
            simX[i+1, :] = linear_mass_model_integrator.get("x")
        # Evaluate user code (e.g., to log current prediction horizon of NMPC controllers)
        if callback is not None:
            callback()
    return simX, simU

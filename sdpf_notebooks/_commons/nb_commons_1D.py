import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
# import casadi as ca
# import scipy.linalg
# from copy import deepcopy

# Build integrator system
from vic_controllers.simulation import build_simulator, export_linear_mass_model
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory


# see acados/examples/acados_python/pendulum_on_cart/sim/extensive_example_sim.py
def simulate_controller(controller, simulation_data, N_horizon_controller=0, callback=None):
    # Extract simulation data
    N = simulation_data['N']
    X0 = simulation_data['x0_mass_spring_damper']

    # Build system symulator
    linear_mass_model = export_linear_mass_model(inertia=np.array([[simulation_data['real_mass']]]), f_ext_as_param=True, dim=1)
    acados_sim = build_simulator(linear_mass_model, time_step=simulation_data['Ts'])

    # Prepare output data vector
    nx = linear_mass_model.x.size()[0]
    nu = linear_mass_model.u.size()[0]

    simX = np.full((N, nx), np.nan)
    simU = np.full((N, nu), np.nan)
    sim_f_ext = np.full((N, nx), np.nan)

    # Call controller in loop:
    simX[0, :] = X0
    acados_sim.set("seed_adj", np.ones((nx, 1)))
    for i in tqdm(range(N)):
        # Receding horizon at the end...
        if (i <= N-N_horizon_controller-1):
            actual_idx_end_horizon = i + N_horizon_controller + 1
        else:
            actual_idx_end_horizon = N
        # Compute control input
        range_idx_horizon = range(i, actual_idx_end_horizon)

        # Measurement data
        measurements = MeasurementData(dimension=1)
        measurements.p = simX[i, 0].reshape(-1,)
        measurements.p_dot = simX[i, 1].reshape(-1,)

        # External force
        K_env = simulation_data['stiffness_env'][i]
        measurements.f_ext = K_env * (0.0 - simX[i, 0]).reshape(-1,)
        measurements.f_ext += simulation_data['f_ext_extra'][i].reshape(-1,)

        # Compliant frame reference
        ref_compliant_frame_traj = CompliantFrameTrajectory(
            dimension=1,
            trajectory_lenght=len(range_idx_horizon)
        )
        # Fill desired robot trajectory
        ref_compliant_frame_traj.p_desired = simulation_data['X_d'][range_idx_horizon, 0].reshape(-1, 1)
        ref_compliant_frame_traj.p_dot_desired = simulation_data['X_d'][range_idx_horizon, 1].reshape(-1, 1)
        ref_compliant_frame_traj.p_ddot_desired = simulation_data['X_d'][range_idx_horizon, 2].reshape(-1, 1)
        # Fill desired robot compliance
        ref_compliant_frame_traj.K_desired = simulation_data['K_d'][range_idx_horizon].reshape(-1, 1, 1)
        ref_compliant_frame_traj.K_dot_desired = simulation_data['K_d_dot'][range_idx_horizon].reshape(-1, 1, 1)
        ref_compliant_frame_traj.D_desired = simulation_data['D_d'][range_idx_horizon].reshape(-1, 1, 1)
        ref_compliant_frame_traj.D_dot_desired = simulation_data['D_d_dot'][range_idx_horizon].reshape(-1, 1, 1)
        ref_compliant_frame_traj.M_desired = simulation_data['M_d'][range_idx_horizon].reshape(-1, 1, 1)
        ref_compliant_frame_traj.M_dot_desired = simulation_data['M_d_dot'][range_idx_horizon].reshape(-1, 1, 1)

        # Simulate time step
        simU[i, :] = controller.compute_control(
            dt=simulation_data['Ts'],
            measurements=measurements,
            ref_compliance_frame=ref_compliant_frame_traj,
            model_inertia=np.array([[simulation_data['model_mass']]])
        )
        sim_f_ext[i, :] = measurements.f_ext

        acados_sim.set("x", simX[i, :])
        acados_sim.set("u", simU[i, :])
        acados_sim.set("p", measurements.f_ext)
        status = acados_sim.solve()
        if status != 0:
            raise Exception(f'acados returned status {status}.')
        # Save state (at k+1)
        if (i < N-1):
            simX[i+1, :] = acados_sim.get("x")
        # Evaluate user code (e.g., to log current prediction horizon of NMPC controllers)
        if callback is not None:
            callback()
    return simU, simX, sim_f_ext


def plot_results_bednarczyk(simulation_data, controller_handle):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(
        simulation_data['time'],
        controller_handle.controller_log['K_diag'],
        label="K"
    )
    plt.plot(
        simulation_data['time'],
        simulation_data['K_d'],
        "--",
        label="K_d"
    )
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(
        simulation_data['time'],
        controller_handle.controller_log['D_diag'],
        label="D"
    )
    plt.plot(
        simulation_data['time'],
        simulation_data['D_d'],
        "--",
        label="D_d"
    )
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    if controller_handle.settings['passivation_function'] == 'bednarczyk_W2':
        plt.plot(
            simulation_data['time'],
            controller_handle.controller_log['V2'],
            label="Storage function V2"
        )
    if controller_handle.settings['passivation_function'] == 'bednarczyk_W4':
        plt.plot(
            simulation_data['time'],
            controller_handle.controller_log['V4'],
            label="Storage function V4"
        )
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        simulation_data['time'],
        controller_handle.controller_log['gamma'],
        label="gamma"
    )
    plt.plot(
        simulation_data['time'],
        controller_handle.controller_log['gamma_target'],
        label="gamma_target"
    )
    plt.legend()

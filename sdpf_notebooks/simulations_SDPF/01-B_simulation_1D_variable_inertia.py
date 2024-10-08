# %% [markdown]
# # Generate simulation data

# %%

# ##################################
# Simulation settings
# ##################################

epsilon_stability = 1e-3

SAVE_FIGS = True
export_figs_dir = "export_figures/simulations_variable_inertia"

# ##################################

import matplotlib
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 1, use_latex=True)
# plt.rcParams['text.usetex'] = True

from tqdm import tqdm
import numpy as np
import scipy.linalg
from copy import deepcopy

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

import os
import sys

parent_folder = os.path.abspath(os.path.join(__file__, os.pardir))
commons_module_path = os.path.abspath(os.path.join(parent_folder, os.pardir, '_commons/'))
if commons_module_path not in sys.path:
    print (f'adding {commons_module_path} to PYTHON_PATH...')
    sys.path.append(commons_module_path)

import plot_utils_1D
plot_utils_1D.ensure_dir_exists(export_figs_dir)

# Base simulation scenario
import simulation_scenarios
import nb_commons_1D

simulation_data = simulation_scenarios.make_simulation_data('scenario_variable_inertia')

simulate_controller_and_package_data = nb_commons_1D.simulate_controller_and_package_data

alpha_value = (np.min(simulation_data['D_d']) - epsilon_stability) / np.min(simulation_data['M_d'])
print(f"alpha = {alpha_value}")
#%%
# ---------------------
# Vanilla controller
# ---------------------
vanilla_VIC_controller_sim_data = \
    nb_commons_1D.get_vanilla_VIC_controller_sim_data(simulation_data, alpha_value)

# ----------------------
# Our controller SDPF
#  -> beta_M = beta_D = beta_K
# ----------------------
from vic_controllers.controllers import SdpfController

# Setup and build PPF controller
controller_SDPF = SdpfController({
    'dim' : 1,
    'alpha' : alpha_value,
    'epsilon_stability' : epsilon_stability,
    'independent_beta_values' : False,
    'beta_max' : 100.0,
    'filter_implementation' : 'LP',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
})

controller_SDPF_sim_data = simulate_controller_and_package_data(controller_SDPF, simulation_data, 'SDPF')


# %% [markdown]
# # Plot benchmark results

# %% Define utils

SDPF_controllers_sim_datasets = [
    controller_SDPF_sim_data
]

placeholder_dataset = {
    'is_vanilla' : False,
    'is_placeholder' : True,
}

controller_sim_datasets = [
    vanilla_VIC_controller_sim_data,
    placeholder_dataset
] + SDPF_controllers_sim_datasets

# precompute the integrals
import scipy
for controller_sim_data in SDPF_controllers_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        continue
    print('Computing z for controller "' + controller_sim_data['label'])
    # controller_sim_data['z_dot_integral'] = np.empty_like(simulation_data['time'])
    controller_sim_data['controller'].controller_log['z_dot_integral'] = np.cumsum(
        controller_sim_data['controller'].controller_log['z_dot'].reshape((-1,))
    ) * simulation_data['Ts']

# %% [markdown]
# ##  Main results: SIPF vs. SDPF

# %% Plot all results
fig_profile, axs_profile = plot_utils_1D.plot_M_and_D(
    simulation_data, controller_sim_datasets, num_columns=2)

fig_state_meas, axs_state_meas = plot_utils_1D.plot_cartesian_state(
    simulation_data, controller_sim_datasets, num_columns=2)

fig_z_z_dot_beta, axs_z_z_dot_beta = plot_utils_1D.plot_z_dot_z_and_beta(
    simulation_data, controller_sim_datasets, num_columns=2)

fig_vic_errors, axs_vic_errors = plot_utils_1D.plot_vic_tracking_errors(
    simulation_data, controller_sim_datasets, num_columns=2)

# %% Export figures

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    prepend_to_figname = ""
    multi_format_savefig(
        figure = fig_profile,
        dir_name = export_figs_dir,
        fig_name = "impedance_profiles" + prepend_to_figname
    )
    multi_format_savefig(
        figure = fig_state_meas,
        dir_name = export_figs_dir,
        fig_name = "pos_vel_and_force" + prepend_to_figname
    )
    # No need for prepend, SIPF_W4 is already ignored by default
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "z_dot_z_and_beta"
    )
    multi_format_savefig(
        figure = fig_vic_errors,
        dir_name = export_figs_dir,
        fig_name = "vic_errors" + prepend_to_figname
    )

# Show figure in GUI if is main() script
if __name__ == '__main__':
    import sys
    try:
        # Put matplotlib.pyplot in interactive mode so that the plots are shown in a background thread.
        plt.ion()
        while(True):
            plt.show(block=True)

    except KeyboardInterrupt:
        print ("Caught KeyboardInterrupt, terminating workers")
        sys.exit(0)

# %% [markdown]
# # Generate simulation data

# %%

# ##################################
# Simulation settings
# ##################################

epsilon_stability = 0.0
plot_SIPF_W4 = False  # Set to False to not plot SIPF W4 results

SAVE_FIGS = True
export_figs_dir = "export_figures/simulations_SIPF_vs_SDPF"

if plot_SIPF_W4:
    print("Plotting SIPF W4 results")
else:
    print("Not plotting SIPF W4 results, only SIPF W2 and SDPF")

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

import plot_utils
import plot_utils_1D
plot_utils.ensure_dir_exists(export_figs_dir)
plot_utils.set_linestyle_list()  # Set default linestyle cycle

# Base simulation scenario
import simulation_scenarios
import nb_commons_1D

simulation_data = simulation_scenarios.make_simulation_data('scenario_1')  # 'scenario_1_K_only')

simulate_controller_and_package_data = nb_commons_1D.simulate_controller_and_package_data

alpha_value = (np.min(simulation_data['D_d']) - epsilon_stability) / np.max(simulation_data['M_d'])
print(f"alpha = {alpha_value}")
#%%
# ---------------------
# Vanilla controller
# ---------------------
vanilla_VIC_controller_sim_data = \
    nb_commons_1D.get_vanilla_VIC_controller_sim_data(simulation_data, alpha_value)

# ---------------------
# Maciej's controllers
# ---------------------
from vic_controllers.controllers import Bednarczyk2020
plot_results_bednarczyk = nb_commons_1D.plot_results_bednarczyk

# Setup and build PPF controller
controller_SIPF_W2 = Bednarczyk2020({
    'dim' : 1,
    'beta' : 100.0,
    'passivation_function' : 'bednarczyk_W2',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
    'M' : np.array([[np.min(simulation_data['M_d'])]]),
    'K_min' : np.array([[np.min(simulation_data['K_d'])]]),
    'K_max' : np.array([[np.max(simulation_data['K_d'])]]),
    'D_min' : np.array([[np.min(simulation_data['D_d'])]]),
    'D_max' : np.array([[np.max(simulation_data['D_d'])]])
})

controller_SIPF_W2_sim_data = simulate_controller_and_package_data(controller_SIPF_W2, simulation_data, 'SIPF')


controller_SIPF_W4_settings = deepcopy(controller_SIPF_W2.settings.data)
controller_SIPF_W4_settings['passivation_function'] = 'bednarczyk_W4'
controller_SIPF_W4 = Bednarczyk2020(controller_SIPF_W4_settings)
controller_SIPF_W4_sim_data = simulate_controller_and_package_data(controller_SIPF_W4, simulation_data, 'SIPF+')

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

if plot_SIPF_W4:
    controller_sim_datasets = [
        vanilla_VIC_controller_sim_data,
        controller_SIPF_W2_sim_data,
        controller_SIPF_W4_sim_data
    ] + SDPF_controllers_sim_datasets
else:
    controller_sim_datasets = [
        vanilla_VIC_controller_sim_data,
        controller_SIPF_W2_sim_data,
        placeholder_dataset
    ] + SDPF_controllers_sim_datasets

# precompute the integrals
import scipy
for controller_sim_data in [controller_SIPF_W2_sim_data] + SDPF_controllers_sim_datasets:
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
ncols = 4 if plot_SIPF_W4 else 3
fig_profile, axs_profile = plot_utils_1D.plot_K_and_D(
    simulation_data, controller_sim_datasets, num_columns=ncols)

fig_K, axs_K = plot_utils_1D.plot_K(
    simulation_data, controller_sim_datasets, num_columns=ncols)

fig_state_meas, axs_state_meas = plot_utils_1D.plot_cartesian_state(
    simulation_data, controller_sim_datasets, num_columns=ncols)

fig_z_z_dot_beta, axs_z_z_dot_beta = plot_utils_1D.plot_z_dot_z_and_beta(
    simulation_data, controller_sim_datasets, num_columns=ncols)

fig_z_z_dot_beta_annotated, axs_z_z_dot_beta_annotated = \
    plot_utils_1D.plot_z_dot_z_and_beta(
        simulation_data,
        controller_sim_datasets,
        num_columns=ncols,
        restrict_z_dot_y_range=True,
        annotate_nominal_peaks=True
    )

fig_vic_errors, axs_vic_errors = plot_utils_1D.plot_vic_tracking_errors(
    simulation_data, controller_sim_datasets, num_columns=ncols)

# %% Export figures

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    prepend_to_figname = "_with_SIPF_W4" if plot_SIPF_W4 else ""
    multi_format_savefig(
        figure = fig_profile,
        dir_name = export_figs_dir,
        fig_name = "impedance_profiles" + prepend_to_figname
    )
    multi_format_savefig(
        figure = fig_K,
        dir_name = export_figs_dir,
        fig_name = "stiffness_only_profiles" + prepend_to_figname
    )
    multi_format_savefig(
        figure = fig_state_meas,
        dir_name = export_figs_dir,
        fig_name = "pos_vel_and_force" + prepend_to_figname
    )
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "z_dot_z_and_beta" + prepend_to_figname
    )
    multi_format_savefig(
        figure = fig_z_z_dot_beta_annotated,
        dir_name = export_figs_dir,
        fig_name = "z_dot_z_and_beta(annotated)" + prepend_to_figname
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
        # Put matplotlib.pyplot in interactive mode so that the plots
        # are shown in a background thread.
        plt.ion()
        while (True):
            plt.show(block=True)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        sys.exit(0)

# %% [markdown]
# # Generate simulation data

# %%

import matplotlib
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 2, use_latex=True)
# plt.rcParams['text.usetex'] = True

from tqdm import tqdm
import numpy as np
import scipy.linalg
from copy import deepcopy

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

import os
import sys
commons_module_path = os.path.abspath(os.path.join('../_commons/'))
if commons_module_path not in sys.path:
    sys.path.append(commons_module_path)

import plot_utils_1D

export_figs_dir = "export_figures/simulations_SIPF_vs_SDPF"
plot_utils_1D.ensure_dir_exists(export_figs_dir)

# Base simulation scenario
import simulation_scenarios
import nb_commons_1D
simulate_controller_and_package_data = nb_commons_1D.simulate_controller_and_package_data


simulation_data = \
    simulation_scenarios.make_simulation_data('scenario_1')
    # simulation_scenarios.make_simulation_data('scenario_1_K_only')
tau_delay_adaptive_z_min = 3.0
z_max = 1.0


alpha_value = np.min(simulation_data['D_d'])/np.max(simulation_data['M_d'])

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
    'independent_beta_values' : False,
    'beta_max' : 100.0,
    'filter_implementation' : 'LP',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
})

controller_SDPF_sim_data = simulate_controller_and_package_data(controller_SDPF, simulation_data, 'SDPF')

# ---------------------------------------------
# Our controller SDPF, with integral condition
# ---------------------------------------------

controller_SDPF_integral = SdpfController({
    'dim' : 1,
    'alpha' : alpha_value,
    'independent_beta_values' : False,
    'passivation_method' : 'z_lower_bound',
    'z_max': z_max,
    'beta_max' : 100.0,
    'filter_implementation' : 'LP',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
})

controller_SDPF_integral_sim_data = simulate_controller_and_package_data(controller_SDPF_integral, simulation_data, 'SDPF integral')

# ------------------------------------------------------
# Our controller SDPF, with ADAPTIVE integral condition
# ------------------------------------------------------

controller_SDPF_adaptive = SdpfController({
    'dim' : 1,
    'alpha' : alpha_value,
    'independent_beta_values' : False,
    'passivation_method' : 'z_adaptative_lower_bound',
    'tau_delay_adaptive_z_min': tau_delay_adaptive_z_min,
    'beta_max' : 100.0,
    'filter_implementation' : 'LP',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
})

controller_SDPF_adaptive_sim_data = simulate_controller_and_package_data(controller_SDPF_adaptive, simulation_data, 'SDPF adaptive')


# %% [markdown]
# # Plot benchmark results
#
# ## Define utils

# %%
# Prepare plots
SAVE_FIGS = True
plot_SIPF_W4 = True

SDPF_controllers_sim_datasets = [
    controller_SDPF_sim_data,
    controller_SDPF_integral_sim_data,
    controller_SDPF_adaptive_sim_data
]

placeholder_dataset = {
    'is_vanilla' : False,
    'is_placeholder' : True,
}

if plot_SIPF_W4:
    controller_sim_datasets = [
        vanilla_VIC_controller_sim_data,
        placeholder_dataset,
        controller_SIPF_W4_sim_data
    ] + SDPF_controllers_sim_datasets
else:
    controller_sim_datasets = [
        vanilla_VIC_controller_sim_data,
        placeholder_dataset,
        placeholder_dataset
    ] + SDPF_controllers_sim_datasets

# precompute the integrals
import scipy
for controller_sim_data in [controller_SIPF_W2_sim_data] + SDPF_controllers_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        continue
    print('Computing z for controller "' + controller_sim_data['label'])
    # controller_sim_data['z_dot_integral'] = np.empty_like(simulation_data['time'])
    controller_sim_data['z_dot_integral'] = np.cumsum(
        controller_sim_data['controller'].controller_log['z_dot'].reshape((-1,))
    ) * simulation_data['Ts']

vanilla_VIC_controller_sim_data['z_dot_integral'] = np.cumsum(
        vanilla_VIC_controller_sim_data['z_dot'].reshape((-1,))
    ) * simulation_data['Ts']

color_list = plot_utils_1D.get_color_list()

flip = plot_utils_1D.flip
highlight_regions = plot_utils_1D.highlight_regions
annotate_regions = plot_utils_1D.annotate_regions

# %% [markdown]
# ##  Main results


# %% Plot all results
fig_profile, axs_profile = plot_utils_1D.plot_K_and_D(
    simulation_data, controller_sim_datasets, num_columns=3)

fig_state_meas, axs_state_meas = plot_utils_1D.plot_cartesian_state(
    simulation_data, controller_sim_datasets, num_columns=3)

fig_z_z_dot_beta, axs_z_z_dot_beta = plot_utils_1D.plot_z_dot_z_and_beta(
    simulation_data, controller_sim_datasets, num_columns=4)

fig_vic_errors, axs_vic_errors = plot_utils_1D.plot_vic_tracking_errors(
    simulation_data, controller_sim_datasets, num_columns=4)

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

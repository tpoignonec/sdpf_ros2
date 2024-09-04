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

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

import os
import sys
commons_module_path = os.path.abspath(os.path.join('../_commons/'))
if commons_module_path not in sys.path:
    sys.path.append(commons_module_path)

import simulation_scenarios
import nb_commons_1D
plot_results_bednarczyk = nb_commons_1D.plot_results_bednarczyk
import plot_utils_1D
color_list = plot_utils_1D.get_color_list()
flip = plot_utils_1D.flip
highlight_regions = plot_utils_1D.highlight_regions
annotate_regions = plot_utils_1D.annotate_regions
simulate_controller_and_package_data = nb_commons_1D.simulate_controller_and_package_data



SAVE_FIGS = True
export_figs_dir = "export_figures/effect_of_z_max_and_tau_z"
simulation_data = simulation_scenarios.make_simulation_data('scenario_1')  # 'scenario_1_K_only')

tau_delay_adaptive_z_min_list = [1., 5., 10.]
z_max_list = [0.05, 0.1, 0.2, 0.3]

plot_utils_1D.ensure_dir_exists(export_figs_dir)

# %% [markdown]
# # Effect of tau filter z(t)

# %%
controller_sim_datasets = []
i = 0
for tau_delay_adaptive_z_min in tau_delay_adaptive_z_min_list:
    i += 1
    # --------------------------------
    # SDPF with ADAPTIVE z constraint
    # --------------------------------
    from vic_controllers.controllers import SdpfController

    _controller_SDPF = SdpfController({
        'dim' : 1,
        'alpha' : np.min(simulation_data['D_d'])/np.max(simulation_data['M_d']),
        'independent_beta_values' : False,
        'passivation_method' : 'z_adaptative_lower_bound',
        'tau_delay_adaptive_z_min': tau_delay_adaptive_z_min,
        'beta_max' : 100.0,
        'filter_implementation' : 'LP',
        'verbose' : False,
        'N_logging' : simulation_data['N'],
    })
    _controller_SDPF_sim_data = simulate_controller_and_package_data(
        _controller_SDPF, simulation_data, str(tau_delay_adaptive_z_min))

    _controller_SDPF_sim_data['tau_delay_value'] = tau_delay_adaptive_z_min

    controller_sim_datasets += [_controller_SDPF_sim_data]

fig_z_z_dot_beta, axs = plot_utils_1D.plot_K_z_dot_z_and_beta(
    simulation_data,
    controller_sim_datasets,
    num_columns=len(tau_delay_adaptive_z_min_list),
    plot_z_min=True,
    plot_z_max=False
)

axs[0].legend(
    title=r'Time constant for the filtered variable $\bar{z}(t)$',
    ncol=len(tau_delay_adaptive_z_min_list),
    bbox_to_anchor=(0.5, 2.0),
    loc='upper center',
)  # , framealpha=0.5)

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "effect_of_tau_filtered_z"
    )

# %% [markdown]
# # Effect of z_max

# %%
controller_sim_datasets = []
i = 0
for z_max in z_max_list:
    i += 1
    # --------------------------------
    # SDPF with z constraint
    # --------------------------------
    from vic_controllers.controllers import SdpfController

    _controller_SDPF = SdpfController({
        'dim' : 1,
        'alpha' : np.min(simulation_data['D_d'])/np.max(simulation_data['M_d']),
        'independent_beta_values' : False,
        'passivation_method' : 'z_lower_bound',
        'z_max': z_max,
        'beta_max' : 100.0,
        'filter_implementation' : 'LP',
        'verbose' : False,
        'N_logging' : simulation_data['N'],
    })
    _controller_SDPF_sim_data = simulate_controller_and_package_data(
        _controller_SDPF, simulation_data, str(z_max))

    _controller_SDPF_sim_data['z_max'] = z_max

    controller_sim_datasets += [_controller_SDPF_sim_data]


fig_z_z_dot_beta, axs = plot_utils_1D.plot_K_z_dot_z_and_beta(
    simulation_data,
    controller_sim_datasets,
    num_columns=4,
    plot_z_min=False,
    plot_z_max=True
)


axs[0].legend(
    title=r'Upper bound $z_{max}$ on $z(t)$',
    ncol=4,
    bbox_to_anchor=(0.5, 2.0),
    loc='upper center',
)  # , framealpha=0.5)


# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "effect_of_z_max"
    )

# %% [markdown]
# # Generate simulation data

# %%
# ##################################
# Simulation settings
# ##################################

epsilon_stability = 1e-3

# list of parameter values to test
epsilon_value_list = [0.0, 0.2, 0.5, 1]

SAVE_FIGS = True
export_figs_dir = "export_figures/effect_of_alpha_and_epsilon"
# ##################################

import matplotlib
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 1, use_latex=True)
# plt.rcParams['text.usetex'] = True

from tqdm import tqdm
import numpy as np
import scipy.linalg

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

import os
import sys
parent_folder = os.path.abspath(os.path.join(__file__, os.pardir))
commons_module_path = os.path.abspath(os.path.join(parent_folder, os.pardir, '_commons/'))
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

plot_utils_1D.ensure_dir_exists(export_figs_dir)
simulation_data = simulation_scenarios.make_simulation_data('scenario_1')  # 'scenario_1_K_only')

ideal_alpha_value = (np.min(simulation_data['D_d']) - epsilon_stability) / np.min(simulation_data['M_d'])
print(f"ideal_alpha_value = {ideal_alpha_value}")

alpha_value_list = [0.0, ideal_alpha_value/2.0, ideal_alpha_value]

# %% [markdown]
# # Effect of alpha on SDPF "classic"

# %%
# Effect of alpha on SDPF "classic"
controller_sim_datasets = []
i = 0
for alpha_value in alpha_value_list:
    i += 1
    # --------------------------------
    # SDPF with ADAPTIVE z constraint
    # --------------------------------
    from vic_controllers.controllers import SdpfController

    _controller_SDPF = SdpfController({
        'dim' : 1,
        'alpha' : alpha_value,
        'epsilon_stability' : 0.0,
        'independent_beta_values' : False,
        'passivation_method' : 'w_lower_bound',
        'beta_max' : 100.0,
        'filter_implementation' : 'LP',
        'verbose' : False,
        'N_logging' : simulation_data['N'],
    })
    _controller_SDPF_sim_data = simulate_controller_and_package_data(
        _controller_SDPF, simulation_data, str(np.round(alpha_value, 2)))

    _controller_SDPF_sim_data['alpha'] = alpha_value

    controller_sim_datasets += [_controller_SDPF_sim_data]

fig_z_z_dot_beta, axs = plot_utils_1D.plot_K_z_dot_z_and_beta(
    simulation_data,
    controller_sim_datasets,
    num_columns=len(alpha_value_list),
    plot_z_min=False,
    plot_z_max=False
)

axs[0].legend(
    title=r'Value chosen for $\alpha$ (n.b., "ideal" $\approx$ ' + str(np.round(ideal_alpha_value, 2)) + r')',
    ncol=len(alpha_value_list),
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
        fig_name = "effect_of_alpha_on_SDPF"
    )


# %% [markdown]
# # Effect of epsilon on SDPF "classic"

# %%
# Effect of epsilon on SDPF "classic"
controller_sim_datasets = []
i = 0
for epsilon_value in epsilon_value_list:
    i += 1
    # --------------------------------
    # SDPF with ADAPTIVE z constraint
    # --------------------------------
    from vic_controllers.controllers import SdpfController

    alpha_value = \
        np.min(simulation_data['D_d'] - epsilon_value)/np.max(simulation_data['M_d'])

    _controller_SDPF = SdpfController({
        'dim' : 1,
        'alpha' : alpha_value,
        'epsilon_stability' : epsilon_value,
        'independent_beta_values' : False,
        'passivation_method' : 'w_lower_bound',
        'beta_max' : 100.0,
        'filter_implementation' : 'LP',
        'verbose' : False,
        'N_logging' : simulation_data['N'],
    })
    _controller_SDPF_sim_data = simulate_controller_and_package_data(
        _controller_SDPF, simulation_data, str(epsilon_value))

    _controller_SDPF_sim_data['epsilon_stability'] = epsilon_value

    controller_sim_datasets += [_controller_SDPF_sim_data]

fig_EPSILON_z_z_dot_beta, axs = plot_utils_1D.plot_K_z_dot_z_and_beta(
    simulation_data,
    controller_sim_datasets,
    num_columns=len(epsilon_value_list),
    plot_z_min=False,
    plot_z_max=False
)

axs[0].legend(
    title=r'Value chosen for $\epsilon_0$',
    ncol=len(epsilon_value_list),
    bbox_to_anchor=(0.5, 2.0),
    loc='upper center',
)  # , framealpha=0.5)

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_EPSILON_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "effect_of_epsilon_on_SDPF"
    )


# %%
# Test existence of solutions...

from LPP_QPP_utils import *

# null error and error rate
beta_max = 100.
LPP_in = {
    'beta_max': beta_max,
    'epsilon_stability': 0.0,
    'err': 0.1,
    'err_dot': 0.0,
    'f_ext': 0.0,
    'M': 5.,
    'D': 5.0,
    'K': 10.0,
    'M_d': np.array([[2.]]),
    'D_d': np.array([[5.0]]),
    'K_d': np.array([[50.0]]),
    'c': - np.array([1., 1., 1.]),
}

def get_solution(A_ub, b_ub):
    # beta_M = beta_D = beta_K
    a1 = LPP_out['b_ub'][0]
    a2 = -1 * (LPP_out['A_ub'][0, 0] + LPP_out['A_ub'][0, 1] + LPP_out['A_ub'][0, 2])
    # Analytical solution
    sol_base = 0.0
    if a2 >= 0 or np.abs(a2) < 1e-9:
        sol_base = np.array([beta_max])
    else:
        sol_base = np.array([- a1 / a2])
    w_est = sol_base * a2 + a1
    solution = np.clip(sol_base, 0.0, beta_max)
    return np.array([float(solution)])[0], w_est

beta_solutions = []
epsilon_values = np.linspace(0.0, 2.0, 10)

for epsilon_stability in epsilon_values:
    print('epsilon_stability = ', epsilon_stability)
    LPP_in['epsilon_stability'] = epsilon_stability
    LPP_in['alpha'] = np.min(LPP_in['D_d'] - epsilon_stability)/np.max(LPP_in['M_d'])
    LPP_out = build_LPP(LPP_in)
    beta, w_estimated = get_solution(LPP_out['A_ub'], LPP_out['b_ub'])
    print('beta = ', beta, ' and w_est = ', w_estimated)
    beta_solutions += [beta]

    assert(w_estimated > - 1e-9)
    assert(beta > - 1e-9)


plt.figure()
plt.plot(epsilon_values, beta_solutions, '.-')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\beta$')


beta_solutions = []
err_values = np.linspace(0.0, 0.1, 10)
LPP_in['epsilon_stability'] = 0.1
LPP_in['alpha'] = np.min(LPP_in['D_d'] - LPP_in['epsilon_stability'])/np.max(LPP_in['M_d'])

for err_value in err_values:
    print('err_value = ', err_value)
    LPP_in['err'] = err_value
    LPP_out = build_LPP(LPP_in)
    beta, w_estimated = get_solution(LPP_out['A_ub'], LPP_out['b_ub'])
    print('beta = ', beta, ' and w_est = ', w_estimated)
    beta_solutions += [beta]

    assert(w_estimated > - 1e-9)
    assert(beta > - 1e-9)


plt.figure()
plt.plot(err_values, beta_solutions, '.-')
plt.xlabel(r'$e(t)$')
plt.ylabel(r'$\beta$')

# Show figure in GUI if is main() script
if __name__ == '__main__':
    try:
        # Put matplotlib.pyplot in interactive mode so that the plots are shown in a background thread.
        plt.ion()
        while(True):
            plt.show(block=True)

    except KeyboardInterrupt:
        print ("Caught KeyboardInterrupt, terminating workers")
        sys.exit(0)
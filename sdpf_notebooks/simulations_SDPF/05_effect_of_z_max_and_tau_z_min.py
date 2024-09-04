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
import casadi as ca
import scipy.linalg
from copy import deepcopy

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

import os
import sys
commons_module_path = os.path.abspath(os.path.join('../_commons/'))
if commons_module_path not in sys.path:
    sys.path.append(commons_module_path)
import nb_commons_1D
plot_results_bednarczyk = nb_commons_1D.plot_results_bednarczyk

def simulate_controller_and_package_data(controller_handle, sim_data, label_str):
    _simU, _simX, _simFext = nb_commons_1D.simulate_controller(controller_handle, sim_data)
    return {
        'is_placeholder' : False,
        'U' : _simU,
        'X' : _simX,
        'Fext' : _simFext,
        'label' : label_str,
        'controller' : controller_handle
    }

SAVE_FIGS = True
export_figs_dir = "export_figures/effect_of_z_max_and_tau_z"
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Base simulation scenario
import simulation_scenarios
simulation_data = simulation_scenarios.make_simulation_data('scenario_1')  # 'scenario_1_K_only')

# %% [markdown]
# # Effect of tau filter z(t)

# %%
results_list = []
i = 0
for tau_delay_adaptive_z_min in [1., 2., 5., 10.]:
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

    local_result = {
        'controller_handle' : _controller_SDPF,
        'sim_data' : _controller_SDPF_sim_data,
        'tau_delay_value' : tau_delay_adaptive_z_min
    }
    results_list += [local_result]

# %%
import scipy

# PLOTS

gs_kw = dict(width_ratios=[1], height_ratios=[2, 5, 5, 5, 5])
fig_z_z_dot_beta, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['center1'],
    ['center2'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
axd['legend'].axis('off')
ax0 = axd['top']
ax1 = axd['center1']
ax2 = axd['center2']
ax3 = axd['bottom']


def highlight_regions(ax):
    alpha = 0.1
    ax.axvspan(simulation_data['t1'], simulation_data['t2'], color='green', alpha=alpha, lw=0)
highlight_regions(ax0)
highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)

'''
annotation_rel_hight = 0.82
annotate(ax1, r'(a)', (1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(b)', (1/3 + 1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(c)', (2/3 + 1/3/2.25, annotation_rel_hight))
'''

i = 0

for result_data in results_list:
    # -------------------------
    # Stiffness
    # -------------------------
    ax0.plot(
        simulation_data['time'],
        simulation_data['K_d'],
        'k--',
        label = '__NO_LABEL'
    )

    ax0.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['K'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    # -------------------------
    # z_dot
    # -------------------------
    ax1.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['z_dot'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    # -------------------------
    # Integral of z_dot
    # -------------------------
    ax2.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['z'].reshape((-1,)),
        color=color_list[i],
    )
    ax2.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['z_min'].reshape((-1,)),
        ':',
        color=color_list[i]
    )
    # -------------------------
    # Beta
    # -------------------------
    ax3.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['beta'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    i += 1

# Labels & legend
ax0.set_ylabel(r'$K(t)$')  # + ' ' + r'\small{(J.s${}^{-1}$)}')
ax1.set_ylabel(r'$w(\beta, t)$')  # + ' ' + r'\small{(J.s${}^{-1}$)}')
ax2.set_ylabel(
    r'$z(t)$'  # = \int_0^t w(\cdot) d\tau$'
    # r'{\setlength{\fboxrule}{0pt} \fbox{ \phantom{${\displaystyle \int_0^t}$} ${\int_0^t w\left(\beta(\tau), \tau\right) d\tau}$}}'
    # + '\n'
    # + ' '
    # + r'\small{(J)}'
)
ax3.set_ylabel(r'$\beta$')  # + ' ' + r'\small{(unitless)}')
ax3.set_xlabel(r'time (s)')

ax0.legend(
    title=r'Time constant for the filtered variable $\bar{z}(t)$',
    ncol=4,
    bbox_to_anchor=(0.5, 2.0),
    loc='upper center',
)  # , framealpha=0.5)

# extra setup
for ax in [ax0, ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)


fig_z_z_dot_beta.align_ylabels([ax0, ax1, ax2, ax3])
ax1.set_xlim((0., np.max(simulation_data['time'])))

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
results_list = []
i = 0
for z_max in [0.05, 0.1, 0.2, 0.3]:
    i += 1
    # --------------------------------
    # SDPF with ADAPTIVE z constraint
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

    local_result = {
        'controller_handle' : _controller_SDPF,
        'sim_data' : _controller_SDPF_sim_data,
        'tau_delay_value' : tau_delay_adaptive_z_min
    }
    results_list += [local_result]

# %%
import scipy

# PLOTS

gs_kw = dict(width_ratios=[1], height_ratios=[2, 5, 5, 5, 5])
fig_z_z_dot_beta, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['center1'],
    ['center2'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
axd['legend'].axis('off')
ax0 = axd['top']
ax1 = axd['center1']
ax2 = axd['center2']
ax3 = axd['bottom']


def highlight_regions(ax):
    alpha = 0.1
    ax.axvspan(simulation_data['t1'], simulation_data['t2'], color='green', alpha=alpha, lw=0)
highlight_regions(ax0)
highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)

i = 0

for result_data in results_list:
    # -------------------------
    # Stiffness
    # -------------------------
    ax0.plot(
        simulation_data['time'],
        simulation_data['K_d'],
        'k--',
        label = '__NO_LABEL'
    )

    ax0.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['K'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    # -------------------------
    # z_dot
    # -------------------------
    ax1.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['z_dot'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    # -------------------------
    # Integral of z_dot
    # -------------------------
    ax2.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['z'].reshape((-1,)),
        color=color_list[i],
    )

    # TODO: plot z_max

    # -------------------------
    # Beta
    # -------------------------
    ax3.plot(
        simulation_data['time'],
        result_data['sim_data']['controller'].controller_log['beta'].reshape((-1,)),
        color=color_list[i],
        label = result_data['sim_data']['label']
    )
    i += 1

# Labels & legend
ax0.set_ylabel(r'$K(t)$')  # + ' ' + r'\small{(J.s${}^{-1}$)}')
ax1.set_ylabel(r'$w(\beta, t)$')  # + ' ' + r'\small{(J.s${}^{-1}$)}')
ax2.set_ylabel(
    r'$z(t)$'  # = \int_0^t w(\cdot) d\tau$'
    # r'{\setlength{\fboxrule}{0pt} \fbox{ \phantom{${\displaystyle \int_0^t}$} ${\int_0^t w\left(\beta(\tau), \tau\right) d\tau}$}}'
    # + '\n'
    # + ' '
    # + r'\small{(J)}'
)
ax3.set_ylabel(r'$\beta$')  # + ' ' + r'\small{(unitless)}')
ax3.set_xlabel(r'time (s)')

ax0.legend(
    title=r'Upper bound $z_{max}$ on $z(t)$',
    ncol=4,
    bbox_to_anchor=(0.5, 2.0),
    loc='upper center',
)  # , framealpha=0.5)

# extra setup
for ax in [ax0, ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)


fig_z_z_dot_beta.align_ylabels([ax0, ax1, ax2, ax3])
ax1.set_xlim((0., np.max(simulation_data['time'])))

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "effect_of_z_max"
    )

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


export_figs_dir = "export_figures/simulations_SIPF_vs_SDPF"

# Base simulation scenario
import simulation_scenarios
simulation_data = simulation_scenarios.make_simulation_data('scenario_1')  # 'scenario_1_K_only')

# ---------------------
# Vanilla controller
# ---------------------
from vic_controllers.commons import VanillaVicController
vanilla_VIC_controller = VanillaVicController({'dim' : 1})
vanilla_VIC_controller_sim_data = simulate_controller_and_package_data(vanilla_VIC_controller, simulation_data, 'no_passivation')

z_dot_vanilla = np.zeros((simulation_data['N'],))

temp_alpha = np.min(simulation_data['D_d'])/np.max(simulation_data['M_d'])
for idx in range(simulation_data['N']):
    temp_err_pos = simulation_data['X_d'][idx, 0] - vanilla_VIC_controller_sim_data['X'][idx, 0]
    temp_err_vel = simulation_data['X_d'][idx, 1] - vanilla_VIC_controller_sim_data['X'][idx, 1]
    temp_C_dot_value = simulation_data['K_d_dot'][idx] + temp_alpha * simulation_data['D_d_dot'][idx]
    z_dot_vanilla[idx] = \
        temp_err_vel.T * (
            simulation_data['D_d'][idx] - temp_alpha * simulation_data['M_d'][idx]
            ) * temp_err_vel \
        + temp_err_pos.T * (
            temp_alpha * simulation_data['K_d'][idx] - 0.5 * temp_C_dot_value
            ) * temp_err_pos
    del temp_C_dot_value
del temp_alpha

# ---------------------
# Maciej's controllers
# ---------------------
from vic_controllers.controllers import Bednarczyk2020
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
    'alpha' : np.min(simulation_data['D_d'])/np.max(simulation_data['M_d']),
    'independent_beta_values' : False,
    'beta_max' : 100.0,
    'filter_implementation' : 'LP',
    'verbose' : False,
    'N_logging' : simulation_data['N'],
})

controller_SDPF_sim_data = simulate_controller_and_package_data(controller_SDPF, simulation_data, 'SDPF')


# %% [markdown]
# # Plot benchmark results
# 
# ## Define utils

# %%
# Prepare plots
SAVE_FIGS = True
plot_SIPF_W4 = False

label_nominal = 'No passivation'  # label_nominal

SDPF_controllers_sim_datasets = [
    controller_SDPF_sim_data
]

placeholder_dataset = {
    'is_placeholder' : True,
}

if plot_SIPF_W4:
    controller_sim_datasets = [
        controller_SIPF_W2_sim_data,
        controller_SIPF_W4_sim_data
    ] + SDPF_controllers_sim_datasets
else: 
    controller_sim_datasets = [
        controller_SIPF_W2_sim_data,
        placeholder_dataset
    ] + SDPF_controllers_sim_datasets

# %%

import numpy as np
np.cumsum(
        controller_SIPF_W2_sim_data['controller'].controller_log['z_dot'].reshape((-1,))
    ) * simulation_data['Ts']

# %%

import os
export_figs_dir = "export_figures/simulations_SIPF_vs_SDPF"
if not os.path.exists(export_figs_dir):
    # Create a new directory because it does not exist
    os.makedirs(export_figs_dir)
    print(f"The directory {export_figs_dir} was created!")

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']


def highlight_regions(ax):
    alpha = 0.1
    # ax.axvspan(0.0, simulation_data['t2'], color='red', alpha=alpha, lw=0)
    ax.axvspan(simulation_data['t1'], simulation_data['t2'], color='green', alpha=alpha, lw=0)
    # ax.axvspan(simulation_data['t2'], np.max(simulation_data['time']), color='red', alpha=alpha, lw=0)

def annotate(ax, text, coord, bgc='white', extra_text_kwargs={}):
    t = ax.text(
        *coord,
        text,
        transform=ax.transAxes,
        **extra_text_kwargs
    )
    t.set_bbox(dict(facecolor=bgc, alpha=0.5, edgecolor='none'))
    return t


import itertools
def flip(items, ncol):
    # https://stackoverflow.com/questions/10101141/matplotlib-legend-add-items-across-columns-instead-of-down
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def eval_energy_supply(velocity_data, f_ext_data):
    return (simulation_data['X_d'][:,1] - velocity) * f_ext_data

def eval_storage_dot(velocity_data, f_ext_data):
    return (simulation_data['X_d'][:,1] - velocity) * f_ext_data

# precompute the integrals
import scipy
# vanilla_z_dot_integral = np.empty_like(simulation_data['time'])
vanilla_z_dot_integral = np.cumsum(
        z_dot_vanilla.reshape((-1,))
    ) * simulation_data['Ts']
for controller_sim_data in [controller_SIPF_W2_sim_data] + SDPF_controllers_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        continue
    print('Computing z for controller "' + controller_sim_data['label'])
    # controller_sim_data['z_dot_integral'] = np.empty_like(simulation_data['time'])
    controller_sim_data['z_dot_integral'] = np.cumsum(
        controller_sim_data['controller'].controller_log['z_dot'].reshape((-1,))
    ) * simulation_data['Ts']

'''
for idx in range(1, simulation_data['time'].shape[0]):
    integrate_args = {'x': simulation_data['time'][0:idx], 'axis':-1}  # , 'even':'avg'}
    vanilla_z_dot_integral[idx] = integrate.quad(
        z_dot_vanilla[0:idx],
        **integrate_args
    )
    for controller_sim_data in [
        controller_SIPF_W2_sim_data,
        controller_SDPF_basic_sim_data,
        controller_SDPF_independent_LP_sim_data,
        controller_SDPF_independent_QP_sim_data,
        controller_SDPF_advanced_sim_data
    ]:
        controller_sim_data['z_dot_integral'][idx] = scipy.integrate.simpson(
            controller_sim_data['controller'].controller_log['z_dot'].reshape((-1,))[0:idx],
            **integrate_args
        )
'''

# %% [markdown]
# ##  Main results: SIPF vs. SDPF

# %%
# plot state and impedance profiles

# -------------------------
# Impedance profile
# -------------------------
gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 5])
fig_profile, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
axd['legend'].axis('off')

ax1 = axd['top']
ax2 = axd['bottom']

highlight_regions(ax1)
highlight_regions(ax2)

# Stiffness nominal
ax1.plot(
    simulation_data['time'],
    simulation_data['K_d'],
    'k--',
    label = label_nominal
)
# Damping nominal
ax2.plot(
    simulation_data['time'],
    simulation_data['D_d'],
    'k--',
    label = label_nominal
)

# Controllers sim data
for controller_sim_data in controller_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        ax1.plot([], [], label = '_hh')
        ax2.plot([], [], label = '_hh')
        continue
    key_K = 'K'
    key_D = 'D'
    key_M = 'M'
    if controller_sim_data['controller'].controller_log.get('K', None) is None:
        key_K = 'K_diag'
        key_D = 'D_diag'
        key_M = 'M_diag'
    # Stiffness
    ax1.plot(
        simulation_data['time'],
        controller_sim_data['controller'].controller_log[key_K].reshape((-1,)),
        label = controller_sim_data['label']
    )
    ax2.plot(
        simulation_data['time'],
        controller_sim_data['controller'].controller_log[key_D].reshape((-1,)),
        label = controller_sim_data['label']
    )

# extra setup
# ------------
ax1.set_ylabel(r'$K$' + '\n' + r'\small{(N.m$^{-1}$)}')
ax2.set_ylabel(r'$D$' + '\n' + r'\small{(N.m$^{-1}$.s)}')
ax2.set_xlabel(r'time (s)')
ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.45),
    loc='upper center'
)
'''
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(
    flip(handles, 2), flip(labels, 2),
    ncol=3,
    bbox_to_anchor=(0.5, 1.45),
    loc='upper center'
)
'''

for ax in [ax1, ax2]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

fig_profile.align_ylabels([ax1, ax2])
ax1.set_xlim((0., np.max(simulation_data['time'])))

# Annotation phases
annotation_rel_hight = 0.86
annotate(ax1, r'(a)', (1/3/2.25, annotation_rel_hight))
annotate(ax1, r'(b)', (1/3 + 1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(c)', (2/3 + 1/3/2.25, annotation_rel_hight))

# ----------------------------------
# Position/velocity errors + force
# ----------------------------------
gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 5, 5])
fig_state_meas, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['center'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
axd['legend'].axis('off')
ax1 = axd['top']
ax2 = axd['center']
ax3 = axd['bottom']
highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)

# ax2.sharex(ax1)
# ax3.sharex(ax1)

plot_ref = True

# plot reference
# -------------------
if plot_ref:
    # Position
    ax1.plot(
        simulation_data['time'],
        vanilla_VIC_controller_sim_data['X'][:,0],
        'k--',
        label = label_nominal
    )
    # Velocity
    ax2.plot(
        simulation_data['time'],
        vanilla_VIC_controller_sim_data['X'][:,1],
        'k--',
        label = label_nominal
    )
    # Force
    ax3.plot(
        simulation_data['time'],
        vanilla_VIC_controller_sim_data['Fext'][:,0],
        'k--',
        label = label_nominal
    )


# Controllers sim data
for controller_sim_data in controller_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        ax1.plot([], [], label = '_hh')
        ax2.plot([], [], label = '_hh')
        ax3.plot([], [], label = '_hh')
        continue
    # Position
    ax1.plot(
        simulation_data['time'],
        controller_sim_data['X'][:,0],
        label = controller_sim_data['label']
    )
    # Velocity
    ax2.plot(
        simulation_data['time'],
        controller_sim_data['X'][:,1],
        label = controller_sim_data['label']
    )
    # Force
    ax3.plot(
        simulation_data['time'],
        controller_sim_data['Fext'][:,0],
        label = controller_sim_data['label']
    )

# labels
ax1.set_ylabel(r'$p$ (m)')

ax2.set_ylabel(r'$\dot{p}$ (m.s$^{-1}$)')

ax3.set_ylabel(r'$f_{ext}$ (N)')
ax3.set_xlabel(r'time (s)')


# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.6),
    loc='upper center'
)
'''
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(
    flip(handles, 2), flip(labels, 2),
    ncol=3,
    bbox_to_anchor=(0.5, 1.6),
    loc='upper center'
)
'''

fig_state_meas.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(simulation_data['time'])))


# -------------------------------
# EXPORT TO FILES
# -------------------------------
prepend_to_figname = "_with_SIPF_W4" if plot_SIPF_W4 else ""
if SAVE_FIGS :
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

# %% [markdown]
# # Plot z_dot, z, and beta

# %%
# plot passivity related data

gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 5, 5])
fig_z_z_dot_beta, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['center'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
axd['legend'].axis('off')
ax1 = axd['top']
ax2 = axd['center']
ax3 = axd['bottom']
highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)

annotation_rel_hight = 0.82
annotate(ax1, r'(a)', (1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(b)', (1/3 + 1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(c)', (2/3 + 1/3/2.25, annotation_rel_hight))

# Nominal
# -------------------------
# z_dot
ax1.plot(
    simulation_data['time'],
    z_dot_vanilla,
    'k--',
    label=label_nominal
)
# Integral of z_dot
ax2.plot(
    simulation_data['time'],
    vanilla_z_dot_integral,
    'k--',
    label=label_nominal
)

# Skip beta
# ax3.plot([], [], label = '_h')

# Controllers sim data
for controller_sim_data in [placeholder_dataset] * 2 + SDPF_controllers_sim_datasets:
    if (controller_sim_data['is_placeholder']):
        ax1.plot([], [], label = '_hh')
        ax2.plot([], [], label = '_hh')
        ax3.plot([], [], label = '_hh')
        print("skip placeholder!")
        continue
    # z_dot
    ax1.plot(
        simulation_data['time'],
        controller_sim_data['controller'].controller_log['z_dot'],
        label = controller_sim_data['label']
    )
    # Integral of z_dot
    z_key = 'z'
    if (not z_key in controller_sim_data['controller'].controller_log.keys()):
        z_key = 'z_dot_integral'
        print('Controller "' + controller_sim_data['label'] + '" doesnt have the z field! Using computed integral')
    ax2.plot(
        simulation_data['time'],
        controller_sim_data['controller'].controller_log[z_key],
        label = controller_sim_data['label']
    )
    ax3.plot(
        simulation_data['time'],
        controller_sim_data['controller'].controller_log['beta'].reshape((-1,)),
        label = controller_sim_data['label']
    )
ax1.set_ylabel(r'$w(\beta, t)$' + '\n' + r'\small{(J.s${}^{-1}$)}')

ax2.set_ylabel(
    r'{\setlength{\fboxrule}{0pt} \fbox{ \phantom{${\displaystyle \int_0^t}$} ${\int_0^t w\left(\beta(\tau), \tau\right) d\tau}$}}'
    + '\n'
    + r'\small{(J)}'
)

ax3.set_ylabel(r'$\beta$' + '\n' + r'\small{(unitless)}')

# ax2.set_ylabel(r'${\displaystyle \int_0^t w\left(\beta(\tau), \tau\right) d\tau}$')  # r'\small{(J)}')
# ax2.set_ylabel(r'$\int_0^t w$' + '\n' + r'\small{(J)}')
ax3.set_xlabel(r'Time(s)')


# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.legend(
    ncol=3,
    bbox_to_anchor=(0.5, 1.6),
    loc='upper center',
)  # , framealpha=0.5)

fig_z_z_dot_beta.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(simulation_data['time'])))

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "z_dot_z_and_beta"
    )

# %% [markdown]
# ## Plot VIC tracking errors

# %%
# plot vic tracking errors

gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 3, 3, 3])
fig_vic_errors, axd = plt.subplot_mosaic([
    ['legend'],
    ['top'],
    ['M'],
    ['D'],
    ['K']],
    gridspec_kw=gs_kw,
    figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*1.5)
    # layout='constrained'
)
axd['legend'].axis('off')

ax1 = axd['top']
ax2 = axd['M']
ax3 = axd['D']
ax4 = axd['K']

highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)
highlight_regions(ax4)

def compute_vic_errors(controller_sim_data):
    key_K = 'K'
    key_D = 'D'
    key_M = 'M'
    if controller_sim_data['controller'].controller_log.get('K', None) is None:
        key_K = 'K_diag'
        key_D = 'D_diag'
        key_M = 'M_diag'
    M_mat = controller_sim_data['controller'].controller_log[key_M].reshape((-1,))
    D_mat = controller_sim_data['controller'].controller_log[key_D].reshape((-1,))
    K_mat = controller_sim_data['controller'].controller_log[key_K].reshape((-1,))

    M_d_mat = simulation_data['M_d']
    D_d_mat = simulation_data['D_d']
    K_d_mat = simulation_data['K_d']

    M_errors = M_d_mat - M_mat
    D_errors = D_d_mat - D_mat
    K_errors = K_d_mat - K_mat

    err_pos_vect = simulation_data['X_d'][:, 0] - controller_sim_data['X'][:,0]
    err_vel_vect = simulation_data['X_d'][:, 1] - controller_sim_data['X'][:,1]
    err_acc_vect = np.zeros(err_vel_vect.shape)
    f_ext_data = controller_sim_data['Fext'][:,0]
    for i in range(err_acc_vect.shape[0]):
        err_acc_vect[i] = 1. / M_mat[i] * (
            -f_ext_data[i] - D_mat[i] * err_vel_vect[i] - K_mat[i] * err_pos_vect[i])

    err_VIC = M_errors * err_acc_vect + D_errors * err_vel_vect + K_errors * err_pos_vect

    norm_err_VIC = np.zeros(err_VIC.shape)
    norm_err_M_e_dot2 = np.zeros(err_VIC.shape)
    norm_err_D_e_dot = np.zeros(err_VIC.shape)
    norm_err_K_e = np.zeros(err_VIC.shape)
    for i in range(norm_err_VIC.shape[0]):
        norm_err_VIC[i] = err_VIC[i]**2
        norm_err_M_e_dot2[i] = (M_errors[i] * err_acc_vect[i])**2
        norm_err_D_e_dot[i] = (D_errors[i] * err_vel_vect[i])**2
        norm_err_K_e[i] = (K_errors[i] * err_pos_vect[i])**2
    return {
        'norm_err_VIC': norm_err_VIC,
        'norm_err_M_e_dot2': norm_err_M_e_dot2,
        'norm_err_D_e_dot': norm_err_D_e_dot,
        'norm_err_K_e': norm_err_K_e
    }

# Controllers sim data
for controller_sim_data in [
    controller_SIPF_W2_sim_data,
    controller_SIPF_W4_sim_data
] + SDPF_controllers_sim_datasets:
    vic_errors = compute_vic_errors(controller_sim_data)
    ax1.plot(
        simulation_data['time'],
        vic_errors['norm_err_VIC'],
        label = controller_sim_data['label']
    )
    ax2.plot(
        simulation_data['time'],
        vic_errors['norm_err_M_e_dot2'],
        label = controller_sim_data['label']
    )
    ax3.plot(
        simulation_data['time'],
        vic_errors['norm_err_D_e_dot'],
        label = controller_sim_data['label']
    )
    ax4.plot(
        simulation_data['time'],
        vic_errors['norm_err_K_e'],
        label = controller_sim_data['label']
    )

# extra setup
# ------------
ax2.set_xlabel(r'Time (s)')

ax1.set_ylabel(r'\small{$ \| \tilde{M} \ddot{e} + \tilde{D} \dot{e} + \tilde{K} e\|^2$}' + '\n' + r'\small{(N)}')
ax2.set_ylabel(r'$ \| \tilde{M} \ddot{e} \|^2$' + '\n' + r'\small{(N)}')
ax3.set_ylabel(r'$ \| \tilde{D} \dot{e} \|^2$' + '\n' + r'\small{(N)}')
ax4.set_ylabel(r'$ \| \tilde{K} e \|^2$' + '\n' + r'\small{(N)}')
ax1.legend(
    ncol=3,
    bbox_to_anchor=(0.5, 1.45),
    loc='upper center'
)
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)
fig_vic_errors.align_ylabels([ax1, ax2, ax3, ax4])
ax1.set_xlim((0., np.max(simulation_data['time'])))

# Annotation phases
annotation_rel_hight = 0.86
annotate(ax1, r'(a)', (1/3/2.25, annotation_rel_hight))
annotate(ax1, r'(b)', (1/3 + 1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(c)', (2/3 + 1/3/2.25, annotation_rel_hight))



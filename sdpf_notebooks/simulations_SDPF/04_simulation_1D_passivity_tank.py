# %%


import numpy as np
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 2, use_latex=True)

import os
import sys
commons_module_path = os.path.abspath(os.path.join('../_commons/'))
if commons_module_path not in sys.path:
    sys.path.append(commons_module_path)
import nb_commons_1D
simulate_controller = nb_commons_1D.simulate_controller

from vic_controllers.simulation import export_linear_mass_model, plot_linear_mass
from vic_controllers.commons import MeasurementData, CompliantFrameTrajectory

# Base simulation scenario
import simulation_scenarios
simulation_data = simulation_scenarios.make_simulation_data('scenario_1')

export_figs_dir = "export_figures/simulations_passivity_tank"
if not os.path.exists(export_figs_dir):
    # Create a new directory because it does not exist
    os.makedirs(export_figs_dir)
    print(f"The directory {export_figs_dir} was created!")

def highlight_regions(ax):
    alpha = 0.1
    ax.axvspan(simulation_data['t1'], simulation_data['t2'], color='green', alpha=alpha, lw=0)

def annotate(ax, text, coord, bgc='white', extra_text_kwargs={}):
    t = ax.text(
        *coord,
        text,
        transform=ax.transAxes,
        **extra_text_kwargs
    )
    t.set_bbox(dict(facecolor=bgc, alpha=0.5, edgecolor='none'))
    return t

# Simulate passivity tank
from vic_controllers.controllers.bench_1D import Ferraguti2013
settings_tank = {
    "T_init" : 0.2,  # 0.25,
    "T_max" : 0.4,
    "epsilon" : 0.01,
    "K_base" : np.min(simulation_data['K_d']),
    "enable_clipping" : True,
    "N_logging" : simulation_data['N']
}

controller_VIC_tank = Ferraguti2013(settings_tank)
simU_tank, simX_tank, simFext_tank = simulate_controller(controller_VIC_tank, simulation_data)

# plot tank result overview
plt.figure()
plot_linear_mass(simulation_data['time'], None, simU_tank, simX_tank, X_des = simulation_data['X_d'], latexify=False)

controller_VIC_tank.plot_log_tank_level(simulation_data['time'])
controller_VIC_tank.plot_actual_K(simulation_data['time'], K_desired = simulation_data['K_d'])

# %%
SAVE_FIGS = True

# ----------------------------------
# Plot K, tank level, and controls
# ----------------------------------
gs_kw = dict(width_ratios=[6], height_ratios=[5, 5, 5])
fig_passivity_tank, axd = plt.subplot_mosaic([
    ['top'],
    ['center'],
    ['bottom']],
    gridspec_kw=gs_kw,
    sharex=True
    # layout='constrained'
)
ax1 = axd['top']
ax2 = axd['center']
ax3 = axd['bottom']
highlight_regions(ax1)
highlight_regions(ax2)
highlight_regions(ax3)

annotation_rel_hight = 0.83
annotate(ax1, r'(a)', (1/3/2.25, annotation_rel_hight))
annotate(ax1, r'(b)', (1/3 + 1/3/2.25, annotation_rel_hight), bgc='none')
annotate(ax1, r'(c)', (2/3 + 1/3/2.25, annotation_rel_hight))

plot_ref = True

# Stiffness
# -------------------

line_pargs = [
    simulation_data['time'][0],
    simulation_data['time'][-1]
]

line_kwargs = {
    'color': 'black',
    'linestyles': 'dashed',
    'alpha': 0.7
}

ax1.step(simulation_data['time'], controller_VIC_tank.log["equivalent_K"], 'k', where='post', label='K')
# ax1.plot(simulation_data['time'], simulation_data['K_d'], 'k--', alpha=0.7, label='$K^d$')
# ax1.hlines(controller_VIC_tank.settings["K_base"], *line_pargs, **line_kwargs, label='K${}_{base}$')

ax1.set_ylim([0.0, 1.1*np.max(controller_VIC_tank.log["equivalent_K"])])
ax1.set_ylabel(r'Rendered $K$' +'\n' + r'(N.m${}^{-1}$)')

# Tank level
# ----------------

line, = ax2.step(simulation_data['time'], controller_VIC_tank.log["tank_level"], 'k', where='post')
ax2.hlines(controller_VIC_tank.settings["T_max"], *line_pargs, **line_kwargs)
ax2.hlines(controller_VIC_tank.settings["epsilon"], *line_pargs, **line_kwargs)

ax2.set_ylim([min(np.min(controller_VIC_tank.log["tank_level"]), 0.0),
            max(
            np.max(controller_VIC_tank.log["tank_level"]),
            1.1*controller_VIC_tank.settings["T_max"]
        )])

annotate(
    ax2,
    r'{\large $\epsilon$}',
    (1.02, controller_VIC_tank.settings["epsilon"]/ax2.get_ylim()[1]-0.02),
    extra_text_kwargs={
        'fontsize': plt.rcParams["font.size"],
    }
)
annotate(
    ax2,
    r'$T_{max}$',
    (1.02, controller_VIC_tank.settings["T_max"]/ax2.get_ylim()[1]-0.05),
    extra_text_kwargs={
        'fontsize': plt.rcParams["font.size"],
    }
)

ax2.set_ylabel(r'Tank level' + '\n' + r'(J)')

# Force
# -----------------------

ax3.plot(
    simulation_data['time'],
    simU_tank,
    'k'
)
ax3.set_ylabel(r'Control $\tau$' + '\n' + r'(N)')
ax3.set_xlabel(r'time (s)')


# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

'''
ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.4),
    loc='upper center',
)  # , framealpha=0.5)
'''

fig_passivity_tank.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(simulation_data['time'])))

# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_passivity_tank,
        dir_name = export_figs_dir,
        fig_name = "passivity_tank_on_3_phases_simulation"
    )

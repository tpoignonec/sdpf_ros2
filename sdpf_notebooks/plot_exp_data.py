# %% [markdown]
# # Prepare datasets
#
# ## Paths and labels

# %%
import scipy
import numpy as np
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 1, use_latex=True)

import os
import sys

sub_dataset = 'new_recordings'

datasets_path_root = '/home/tpoignonec/dev/ros2_workspaces/ros2-jazzy/ws_sdpf_paper/rosbags/' + sub_dataset + '/'

SAVE_FIGS = True
looking_ax_axis = 0  # X
export_figs_dir = 'export_figures' + '/exp_results-' + sub_dataset

if not os.path.exists(export_figs_dir):
    # Create a new directory because it does not exist
    os.makedirs(export_figs_dir)
    print(f"The directory {export_figs_dir} was created!")


commons_module_path = os.path.abspath(os.path.join('_commons/'))
if commons_module_path not in sys.path:
    sys.path.append(commons_module_path)

# Define topics

vic_state_topic_name = '/cartesian_vic_controller/status'
vic_ref_topic_name = '/cartesian_vic_controller/reference_compliant_frame_trajectory'
pf_diagnostic_data_topic_name = '/passivity_filter_diagnostic_data'
desired_compliant_frame_topic_name = '/desired_compliance'
simulation_time_topic_name = '/simulation_time'

topic_list = [
    vic_state_topic_name,
    vic_ref_topic_name,
    pf_diagnostic_data_topic_name,
    desired_compliant_frame_topic_name,
    simulation_time_topic_name
]

# Define datasets
label_no_passivation = 'No passivation'
label_SIPF_W2 = 'SIPF'
label_SIPF_W4 = 'SIPF+'
label_SDPF = r'SDPF, $w(t) \geq 0$'
label_SDPF_integral = r'SDPF, $z(t) \geq 0$'
label_SDPF_adaptive = r'SDPF, $z(t) \geq z_{min}(t)$'

color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
dataset_info_list = [
    {
        'tag': 'SIPF+',
        'path_to_bag': datasets_path_root + 'SIPF+/SIPF+_0.mcap',
        'label': label_SIPF_W4,
        'color': color_list[0],
        'linestyle': '-'
    },
    {
        'tag': 'SDPF',
        'path_to_bag': datasets_path_root + 'SDPF/SDPF_0.mcap',
        'label': label_SDPF,
        'color': color_list[1],
        'linestyle': '-'
    },
    {
        'tag': 'SDPF_integral',
        'path_to_bag': datasets_path_root + 'SDPF-integral/SDPF-integral_0.mcap',
        'label': label_SDPF_integral,
        'color': color_list[2],
        'linestyle': '--'
    },
    {
        'tag': 'SDPF_adaptive',
        'path_to_bag': datasets_path_root + 'SDPF-adaptive/SDPF-adaptive_0.mcap',
        'label': label_SDPF_adaptive,
        'color': color_list[3],
        'linestyle': ':'
    }
]

'''
# --------------
# Other options:
# --------------
{
    'tag': 'NO_PASSIVATION',
    'path_to_bag': datasets_path_root + 'no_passivation/no_passivation_0.mcap',
    'label': label_no_passivation,
    'color': color_list[0],  # change color if added...
    'linestyle': '-'
},
{
    'tag': 'SIPF',
    'path_to_bag': datasets_path_root + 'SIPF/SIPF_0.mcap',
    'label': label_SIPF_W2,
    'color': color_list[0],  # change color if added...
    'linestyle': '-'
},
'''
print()  # suppress output from comment...

# %% [markdown]
# ## Extract data

# %%
'''
# Select subset of dataset if needed
dataset_info_list = [
    dataset_info_list[0]
]
'''

# %%
import nml_bag
from tqdm import tqdm

from bag_utils import (
    get_simulation_time,
    get_vic_state,
    get_reference_compliant_frame_trajectory,
    get_compliant_frame,
    get_diagnostic_data
)
experimental_data = {}

progressbar = tqdm(dataset_info_list, desc='Loading data, please wait...', leave=True)
for dataset_info in progressbar:
    progressbar.set_description(
        f'Loading data (current EXP is {dataset_info['tag']}), please wait...')
    progressbar.refresh() # to show immediately the update
    reader = nml_bag.Reader(dataset_info['path_to_bag'], topics=topic_list)
    rosbag_data = reader.records

    # Extract actual experimental
    local_data = {}
    local_data['vic_state'] = get_vic_state(rosbag_data, topic_name=vic_state_topic_name)
    local_data['vic_ref'] = get_reference_compliant_frame_trajectory(rosbag_data, topic_name=vic_ref_topic_name)
    local_data['desired_compliant_frame'] = get_compliant_frame(rosbag_data, topic_name=desired_compliant_frame_topic_name)
    local_data['diagnostic_data'] = get_diagnostic_data(rosbag_data, topic_name=pf_diagnostic_data_topic_name)

    # Retrieve simulation time
    local_data['simulation_time'] = get_simulation_time(rosbag_data, topic_name=simulation_time_topic_name)

    # Time mapping definition
    simulation_time_max = np.max(local_data['simulation_time']['time'])
    time_regression_res = scipy.stats.linregress(
        local_data['simulation_time']['ros_time'],
        local_data['simulation_time']['time']
    )
    assert(time_regression_res.rvalue > 0.99)

    def unsafe_map_to_simulation_time(ros_time_in):
        return time_regression_res.intercept + time_regression_res.slope*ros_time_in

    def map_to_simulation_time(ros_time_in):
        naive_value = unsafe_map_to_simulation_time(ros_time_in)
        if ((naive_value < 0.0) or (naive_value > simulation_time_max)):
            return -1  # np.NaN
        else:
            return naive_value

    def append_simulation_time(data_dict):
        data_dict['time'] = np.zeros(data_dict['ros_time'].shape)
        for idx in range(data_dict['time'].shape[0]):
            data_dict['time'][idx] = map_to_simulation_time(data_dict['ros_time'][idx])

    # Cropping function
    def crop_time_serie(data_dict, key, slicing = slice(None)):
        return data_dict[key][np.asarray(data_dict['time'] > -1).nonzero()][slicing]

    # Include simulation time to datasets
    append_simulation_time(local_data['vic_state'])
    append_simulation_time(local_data['vic_ref'])
    append_simulation_time(local_data['desired_compliant_frame'])
    append_simulation_time(local_data['diagnostic_data'])

    # Include label, info tag, color, etc.
    local_data['tag'] = dataset_info['tag']
    local_data['label'] = dataset_info['label']
    local_data['color'] = dataset_info['color']
    local_data['linestyle'] = dataset_info['linestyle']

    # Consolidate dataset
    experimental_data[local_data['tag']] = local_data

print('OK!\n')

print('\nAvailable diagnostic data fields per scenario: \n')
for _, data in experimental_data.items():
    print(data['tag'])
    for key, data_ in data['diagnostic_data'].items():
        print(f'  - {key}')

# %% [markdown]
# # Plot stiffness

# %%
gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5])
fig_stiffness, axd = plt.subplot_mosaic([
    ['legend'],
    ['top']
    ],
    gridspec_kw=gs_kw,
    sharex=True,
    figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*0.6)
)
axd['legend'].axis('off')

ax1 = axd['top']

# Stiffness
ax1.plot(
    crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'),
    crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'stiffness')[:, looking_ax_axis, looking_ax_axis],
    'k--',
    label = '__NO_LABEL'
)
for tag, dataset in experimental_data.items():
    ax1.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        crop_time_serie(dataset['vic_state'], 'stiffness')[:, looking_ax_axis, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )

ax1.set_ylabel(r'$K_x$' + '\n' + r'\small{(N.m$^{-1}$)}')

# extra setup
# ------------
ax1.set_xlabel(r'time (s)')
ax1.legend(
    ncol=4,
    columnspacing=0.8,
    bbox_to_anchor=(0.5, 1.26),
    loc='upper center',
)  # , framealpha=0.5)

for ax in [ax1]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.set_xlim((0., np.max(crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'))))

if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_stiffness,
        dir_name = export_figs_dir,
        fig_name = "stiffness_only"
    )

# %% [markdown]
# # Plot position/velocity, wrench

# %%
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

plot_ref = True

# Position
# -------------------
for tag, dataset in experimental_data.items():
    ax1.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        crop_time_serie(dataset['vic_state'], 'position')[:, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )
ax1.set_ylabel(r'$p_x$')

# Velocity
# ----------------
for tag, dataset in experimental_data.items():
    ax2.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        crop_time_serie(dataset['vic_state'], 'velocity')[:, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )
ax2.set_ylabel(r'$\dot{p}_x$')

# Force
# -----------------------
for tag, dataset in experimental_data.items():
    ax3.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        -crop_time_serie(dataset['vic_state'], 'wrench')[:, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )

ax3.set_ylabel(r'$f_{ext, x}$')
ax3.set_xlabel(r'time (s)')


# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.4),
    loc='upper center',
)  # , framealpha=0.5)

fig_state_meas.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'))))


# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_state_meas,
        dir_name = export_figs_dir,
        fig_name = "pos_vel_and_force"
    )

# %%
# ----------------------------------
# Position/velocity errors + force
# ----------------------------------
gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 5, 5])
fig_stiff_and_state_meas, axd = plt.subplot_mosaic([
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

plot_ref = True

# Stiffness
# -------------------
ax1.plot(
    crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'),
    crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'stiffness')[:, looking_ax_axis, looking_ax_axis],
    'k--',
    label = '__NO_LABEL'
)

for tag, dataset in experimental_data.items():
    ax1.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        crop_time_serie(dataset['vic_state'], 'stiffness')[:, looking_ax_axis, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )

ax1.set_ylabel(r'$K_x$')  # + '\n' + r'\small{(N.m$^{-1}$)}')

# Position
# -------------------
for tag, dataset in experimental_data.items():
    ax2.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        crop_time_serie(dataset['vic_state'], 'position')[:, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )
ax2.set_ylabel(r'$p_x$')

# Force
# -----------------------
for tag, dataset in experimental_data.items():
    ax3.plot(
        crop_time_serie(dataset['vic_state'], 'time'),
        -crop_time_serie(dataset['vic_state'], 'wrench')[:, looking_ax_axis],
        label = dataset['label'],
        color = dataset['color'],
        linestyle = dataset['linestyle']
    )

ax3.set_ylabel(r'$f_{ext, x}$')
ax3.set_xlabel(r'time (s)')


# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.4),
    columnspacing=0.8,
    loc='upper center',
)  # , framealpha=0.5)

fig_stiff_and_state_meas.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'))))


# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_stiff_and_state_meas,
        dir_name = export_figs_dir,
        fig_name = "stiffness_pos_and_force"
    )

# %% [markdown]
# # Plot K, z_dot, z

# %%
import scipy

# PLOTS

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

# -------------------------
# z_dot
# -------------------------
for tag, dataset in experimental_data.items():
    if dataset['diagnostic_data'].get('z_dot') is not None:
        ax1.plot(
            crop_time_serie(dataset['diagnostic_data'], 'time'),
            crop_time_serie(dataset['diagnostic_data'], 'z_dot'),
            label = dataset['label'],
            color = dataset['color'],
            # linestyle = dataset['linestyle']
        )

ax1.set_ylabel(r'$w$' + ' ' + r'\small{(J.s${}^{-1}$)}')

# -------------------------
# Integral of z_dot
# -------------------------
for tag, dataset in experimental_data.items():
    if dataset['diagnostic_data'].get('z') is not None:
        ax2.plot(
            crop_time_serie(dataset['diagnostic_data'], 'time'),
            crop_time_serie(dataset['diagnostic_data'], 'z'),
            label = '__no-label-for_' + dataset['label'],
            color = dataset['color'],
            linestyle = dataset['linestyle']
        )
ax2.plot(
    crop_time_serie(experimental_data['SDPF_PPF_adaptive']['diagnostic_data'], 'time'),
    crop_time_serie(experimental_data['SDPF_PPF_adaptive']['diagnostic_data'], 'z_min'),
    label = r'z_{min}',
    color = 'k',
    linestyle = ':'
)

ax2.legend(loc='lower right')

ax2.set_ylabel(
    r'$z$'  # = \int_0^t w(\cdot) d\tau$'
    # r'{\setlength{\fboxrule}{0pt} \fbox{ \phantom{${\displaystyle \int_0^t}$} ${\int_0^t w\left(\beta(\tau), \tau\right) d\tau}$}}'
    # + '\n'
    + ' '
    + r'\small{(J)}'
)


# ax2.legend(loc='lower right')

# ax2.set_ylabel(r'${\displaystyle \int_0^t w\left(\beta(\tau), \tau\right) d\tau}$')  # r'\small{(J)}')
# ax2.set_ylabel(r'$\int_0^t w$' + '\n' + r'\small{(J)}')
# ax.set_xlabel(r'Time(s)')

# -------------------------
# Beta
# -------------------------
for tag, dataset in experimental_data.items():
    if dataset['diagnostic_data'].get('beta') is not None:
        ax3.plot(
            crop_time_serie(dataset['diagnostic_data'], 'time'),
            crop_time_serie(dataset['diagnostic_data'], 'beta'),
            label = dataset['label'],
            color = dataset['color'],
            # linestyle = dataset['linestyle']
        )

ax3.set_ylabel(r'$\beta$' + ' ' + r'\small{(unitless)}')
ax3.set_xlabel(r'time (s)')

# extra setup
for ax in [ax1, ax2, ax3]:
    ax.grid(which='major')
    ax.grid(which='minor', linewidth=0.1)

ax1.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.4),
    loc='upper center',
)  # , framealpha=0.5)

fig_z_z_dot_beta.align_ylabels([ax1, ax2, ax3])
ax1.set_xlim((0., np.max(crop_time_serie(experimental_data['SDPF_QP']['desired_compliant_frame'], 'time'))))

# -------------------------------
# EXPORT TO FILES
# -------------------------------
if SAVE_FIGS :
    multi_format_savefig(
        figure = fig_z_z_dot_beta,
        dir_name = export_figs_dir,
        fig_name = "z_dot_z_and_beta"
    )


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

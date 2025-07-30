# %% [markdown]
# # Prepare datasets
#
# ## Paths and labels


# %%
def export_exp_figs(sub_dataset = 'new_recordings'):
    #%%
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt
    from vic_controllers.plotting import multi_format_savefig, init_plt
    init_plt(full_screen = False, scale = 1, use_latex=True)

    import os
    import sys

    parent_folder = os.path.abspath(os.path.join(__file__, os.pardir))
    commons_module_path = os.path.abspath(os.path.join(parent_folder, '_commons/'))
    datasets_path_root = os.path.abspath(
        os.path.join(parent_folder, os.pardir, os.pardir, os.pardir, 'rosbags', sub_dataset)) + '/'

    print(f'Datasets path: {datasets_path_root}')

    SAVE_FIGS = True
    export_figs_dir = 'export_figures' + '/exp_results-' + sub_dataset

    # Define axes of interest (i.e., XY plane)
    idx_X = 0  # X
    idx_Y = 1  # Y

    if not os.path.exists(export_figs_dir):
        # Create a new directory because it does not exist
        os.makedirs(export_figs_dir)
        print(f"The directory {export_figs_dir} was created!")


    commons_module_path = os.path.abspath(os.path.join('_commons/'))
    if commons_module_path not in sys.path:
        sys.path.append(commons_module_path)
    import plot_utils

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

    color_list = plot_utils.get_color_list()
    print(f'Color list: {color_list}')
    plot_utils.set_linestyle_list()
    linestyle_list = plot_utils.get_linestyle_list()
    print(f'Linestyle list: {linestyle_list}')
    dataset_info_list = [
        # {
        #     'tag': 'SIPF',
        #     'path_to_bag': datasets_path_root + 'SIPF/SIPF_0.mcap',
        #     'label': label_SIPF_W2,
        #     'color': color_list[0],
        #     'linestyle': linestyle_list[0]
        # },
        {
            'tag': 'SIPF+',
            'path_to_bag': datasets_path_root + 'SIPF+/SIPF+_0.mcap',
            'label': label_SIPF_W4,
            'color': color_list[1],
            'linestyle': linestyle_list[1]
        },
        {
            'tag': 'SDPF',
            'path_to_bag': datasets_path_root + 'SDPF/SDPF_0.mcap',
            'label': label_SDPF,
            'color': color_list[2],
            'linestyle': linestyle_list[2]
        },
        {
            'tag': 'SDPF_adaptive',
            'path_to_bag': datasets_path_root + 'SDPF-adaptive/SDPF-adaptive_0.mcap',
            'label': label_SDPF_adaptive,
            'color': color_list[4],
            'linestyle': linestyle_list[4]
        }
    ]

    print(f'Dataset info used for extraction:\n{dataset_info_list}')

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

    # %% Prepare utils

    def get_max_K_periods_idx(ax, K_data):
        K_mid = 0.5 * (np.max(K_data) + np.min(K_data))

        periods_where_K_is_max = []  # pairs of [t1; t2]

        # Find periods where K is above the midpoint (considered "maximum")
        is_high = K_data > K_mid

        # Find transitions from low to high and high to low
        transitions = np.diff(is_high.astype(int))

        # Get indices where transitions occur
        start_indices = np.where(transitions == 1)[0] + 1  # Start of high periods
        end_indices = np.where(transitions == -1)[0] + 1   # End of high periods

        # Handle edge cases
        if is_high[0]:  # If we start in a high period
            start_indices = np.concatenate([[0], start_indices])

        if is_high[-1]:  # If we end in a high period
            end_indices = np.concatenate([end_indices, [len(is_high) - 1]])

        # Create time pairs for high stiffness periods
        for start_idx, end_idx in zip(start_indices, end_indices):
            periods_where_K_is_max.append((start_idx, end_idx))

        return periods_where_K_is_max

    def annotate_K_min_max(ax, dataset, axis_to_check=0):
        time_data = crop_time_serie(dataset['desired_compliant_frame'], 'time')
        K_data = crop_time_serie(
            dataset['desired_compliant_frame'],
            'stiffness'
        )[:, axis_to_check, axis_to_check]
        periods_idx_where_K_is_max = \
            get_max_K_periods_idx(ax, K_data)

        for (start_idx, end_idx) in periods_idx_where_K_is_max:
            ax.axvspan(time_data[start_idx], time_data[end_idx], color='green', alpha=0.1, lw=0)

    def annotate_K_min_max_on_trajectory(ax, dataset, axis_to_check=0):
        K_data = crop_time_serie(
            dataset['vic_state'],
            'stiffness'
        )[:, axis_to_check, axis_to_check]
        periods_idx_where_K_is_max = \
            get_max_K_periods_idx(ax, K_data)

        for (start_idx, end_idx) in periods_idx_where_K_is_max:
            ax.plot(
                crop_time_serie(
                    experimental_data['SDPF']['vic_state'],
                    'desired_position')[start_idx:end_idx, [idx_X]],
                crop_time_serie(
                    experimental_data['SDPF']['vic_state'],
                    'desired_position')[start_idx:end_idx, [idx_Y]],
                label=r'_K_annotation',
                color='green',
                linestyle='-',
                alpha=0.2,
                linewidth=8
            )

    # %% Plot XY trajectory
    gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5])
    fig_xy_trajectory, axd = plt.subplot_mosaic([
        ['legend'],
        ['main'],
        ],
        gridspec_kw=gs_kw,
        sharex=True,
        figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*1.2)
    )

    axd['legend'].axis('off')
    ax1 = axd['main']

    # Plot XY trajectory
    annotate_K_min_max_on_trajectory(ax1, experimental_data['SDPF'])
    ax1.plot(
        crop_time_serie(
            experimental_data['SDPF']['vic_state'],
            'desired_position')[:, [idx_X]],
        crop_time_serie(
            experimental_data['SDPF']['vic_state'],
            'desired_position')[:, [idx_Y]],
        label=r'_$p^d$',
        color='black',
        linestyle='--'
    )

    for tag, dataset in experimental_data.items():
        ax1.plot(
            crop_time_serie(dataset['vic_state'], 'position')[:, [idx_X]],
            crop_time_serie(dataset['vic_state'], 'position')[:, [idx_Y]],
            label=dataset['label'],
            color=dataset['color'],
            linestyle=dataset['linestyle']
        )

    # Plot start / stop
    start_XY = (
        crop_time_serie(experimental_data['SDPF']['vic_state'], 'position')[0, [idx_X]],
        crop_time_serie(experimental_data['SDPF']['vic_state'], 'position')[0, [idx_Y]]
    )
    ax1.scatter(
        start_XY[0], start_XY[1],
        s=5,
        marker='s',
        color='black',
        label='_Start',
        zorder=100
    )
    from plot_utils import annotate
    annotate(
        ax1, 'Start/Stop', (start_XY[0] + 0.1, start_XY[1]),
        bgc='none',
        extra_text_kwargs={
            'ha': 'left',
            'va': 'top',
            'fontsize': 6,
            'zorder': 100
        }
    )

    ax1.set_xlabel(r'$p_x$' + ' ' + r'\small{(m)}')
    ax1.set_ylabel(r'$p_y$' + ' ' + r'\small{(m)}')

    # extra setup
    ncol_legend_traj_XY = 2
    bbox_to_anchor_y = 1.23
    ax1.legend(
        ncol=ncol_legend_traj_XY,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, bbox_to_anchor_y),
        loc='upper center',
    )

    ax1.grid(which='major')
    ax1.grid(which='minor', linewidth=0.1)
    ax1.set_aspect('equal')

    if SAVE_FIGS :
        multi_format_savefig(
            figure = fig_xy_trajectory,
            dir_name = export_figs_dir,
            fig_name = "xy_trajectory"
        )

    # %% Plot stiffness
    gs_kw = dict(width_ratios=[1], height_ratios=[0.5, 5, 5])
    fig_stiffness, axd = plt.subplot_mosaic([
        ['legend'],
        ['top'],
        ['bottom']
        ],
        gridspec_kw=gs_kw,
        sharex=True,
        figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*0.7)
    )
    axd['legend'].axis('off')

    ax1 = axd['top']
    ax2 = axd['bottom']

    annotate_K_min_max(ax1, experimental_data['SDPF'], axis_to_check=idx_X)
    annotate_K_min_max(ax2, experimental_data['SDPF'], axis_to_check=idx_X)

    # Stiffness along X
    for selected_axis, ax in zip([idx_X, idx_Y], [ax1, ax2]):
        ax.plot(
            crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'),
            crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'stiffness')[:, selected_axis, selected_axis],
            'k--',
            label = '__NO_LABEL'
        )
        for tag, dataset in experimental_data.items():
            ax.plot(
                crop_time_serie(dataset['vic_state'], 'time'),
                crop_time_serie(dataset['vic_state'], 'stiffness')[:, selected_axis, selected_axis],
                label = dataset['label'],
                color = dataset['color'],
                linestyle = dataset['linestyle']
            )

    ax1.set_ylabel(r'$K_x$' + '\n' + r'\small{(N.m$^{-1}$)}')
    ax2.set_ylabel(r'$K_y$' + '\n' + r'\small{(N.m$^{-1}$)}')
    ax2.set_xlabel(r'time (s)')

    # extra setup
    # ------------
    ax1.legend(
        ncol=len(experimental_data),
        columnspacing=0.8,
        bbox_to_anchor=(0.45, 1.4),
        loc='upper center',
    )  # , framealpha=0.5)

    for ax in [ax1, ax2]:
        ax.grid(which='major')
        ax.grid(which='minor', linewidth=0.1)

    fig_stiffness.align_ylabels([ax1, ax2])
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

    annotate_K_min_max(ax1, experimental_data['SDPF'], axis_to_check=idx_X)
    annotate_K_min_max(ax2, experimental_data['SDPF'], axis_to_check=idx_X)
    annotate_K_min_max(ax3, experimental_data['SDPF'], axis_to_check=idx_X)

    plot_ref = True

    # Position error
    # -------------------
    for tag, dataset in experimental_data.items():
        norm_error_XY = np.linalg.norm(
            crop_time_serie(dataset['vic_state'], 'desired_position')[:, [idx_X, idx_Y]] -
            crop_time_serie(dataset['vic_state'], 'position')[:, [idx_X, idx_Y]],
            axis=1
        )
        ax1.plot(
            crop_time_serie(dataset['vic_state'], 'time'), norm_error_XY,
            label = dataset['label'],
            color = dataset['color'],
            linestyle = dataset['linestyle']
        )
    ax1.set_ylabel(r'$||e||$' + '\n' + r'\small{(m)}')

    # Velocity
    # ----------------
    for tag, dataset in experimental_data.items():
        norm_velocity_error_XY = np.linalg.norm(
            crop_time_serie(dataset['vic_state'], 'desired_velocity')[:, [idx_X, idx_Y]] -
            crop_time_serie(dataset['vic_state'], 'velocity')[:, [idx_X, idx_Y]],
            axis=1
        )
        ax2.plot(
            crop_time_serie(dataset['vic_state'], 'time'), norm_velocity_error_XY,
            label = dataset['label'],
            color = dataset['color'],
            linestyle = dataset['linestyle']
        )
    ax2.set_ylabel(r'$||\dot{e}||$' + '\n' + r'\small{(m.s${}^{-1}$)}')

    # Force
    # -----------------------
    for tag, dataset in experimental_data.items():
        f_ext_norm_XY = np.linalg.norm(
            crop_time_serie(dataset['vic_state'], 'wrench')[:, [idx_X, idx_Y]],
            axis=1
        )
        ax3.plot(
            crop_time_serie(dataset['vic_state'], 'time'), f_ext_norm_XY,
            label = dataset['label'],
            color = dataset['color'],
            linestyle = dataset['linestyle']
        )

    ax3.set_ylabel(r'$||f_{ext}||$' + '\n' + r'\small{(N)}')
    ax3.set_xlabel(r'time (s)')


    # extra setup
    for ax in [ax1, ax2, ax3]:
        ax.grid(which='major')
        ax.grid(which='minor', linewidth=0.1)

    ax1.legend(
        ncol=4,
        bbox_to_anchor=(0.45, 1.4),
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
            fig_name = "error_pos_vel_and_force"
        )


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

    annotate_K_min_max(ax1, experimental_data['SDPF'], axis_to_check=idx_X)
    annotate_K_min_max(ax2, experimental_data['SDPF'], axis_to_check=idx_X)
    annotate_K_min_max(ax3, experimental_data['SDPF'], axis_to_check=idx_X)

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

    ax1.set_ylabel(r'$w$' + '\n' + r'\small{(J.s${}^{-1}$)}')

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
    if 'SDPF_adaptive' in experimental_data:
        ax2.plot(
            crop_time_serie(
                experimental_data['SDPF_adaptive']['diagnostic_data'], 'time'),
            crop_time_serie(
                experimental_data['SDPF_adaptive']['diagnostic_data'], 'z_min'),
            label=r'z_{min}',
            color='k',
            linestyle=':'
        )

    ax2.legend(loc='lower right')

    ax2.set_ylabel(
        r'$z$'  # = \int_0^t w(\cdot) d\tau$'
        # r'{\setlength{\fboxrule}{0pt} \fbox{ \phantom{${\displaystyle \int_0^t}$} ${\int_0^t w\left(\beta(\tau), \tau\right) d\tau}$}}'
        # + '\n'
        + '\n'
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

    ax3.set_ylabel(r'$\beta$' + '\n' + r'\small{(unitless)}')
    ax3.set_xlabel(r'time (s)')

    # extra setup
    for ax in [ax1, ax2, ax3]:
        ax.grid(which='major')
        ax.grid(which='minor', linewidth=0.1)


    fig_z_z_dot_beta.align_ylabels([ax1, ax2, ax3])
    ax1.set_xlim((0., np.max(crop_time_serie(experimental_data['SDPF']['desired_compliant_frame'], 'time'))))

    if SAVE_FIGS :
        multi_format_savefig(
            figure = fig_z_z_dot_beta,
            dir_name = export_figs_dir,
            fig_name = "z_dot_z_and_beta_no_legend"
        )

    ax1.legend(
        ncol=4,
        bbox_to_anchor=(0.45, 1.4),
        loc='upper center',
    )  # , framealpha=0.5)

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
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot exp data from ros2 bags and save figures to files.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name such that the ros bags are located in "<ROS2_WORKSPACE>/rosbags/<DATASET>/"')
    parser.add_argument('--display-figs', type=bool, default=False, help='Call plt.shox() to display figures')
    args = vars(parser.parse_args())

    parser.print_help()
    print('\n  -> args: ', args)
    print('\n\n')

    export_exp_figs(sub_dataset = args['dataset'])
    if (args['display_figs']):
        try:
            # Put matplotlib.pyplot in interactive mode so that the plots are shown in a background thread.
            import matplotlib.pyplot as plt
            plt.ion()
            while(True):
                plt.show(block=True)

        except KeyboardInterrupt:
            import sys
            print ("Caught KeyboardInterrupt, terminating workers")
            sys.exit(0)

import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_color_list():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def flip(items, ncol):
    # https://stackoverflow.com/questions/10101141/matplotlib-legend-add-items-across-columns-instead-of-down
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def highlight_regions(simulation_data, ax):
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


# plot state and impedance profiles
def plot_K_and_D(simulation_data, controller_sim_datasets, num_columns=3):

    vanilla_VIC_controller_sim_data = None
    for controller_sim_data in controller_sim_datasets:
        if (controller_sim_data['is_vanilla'] == True):
            vanilla_VIC_controller_sim_data = controller_sim_data
            break
    assert (vanilla_VIC_controller_sim_data is not None)
    label_nominal = vanilla_VIC_controller_sim_data['label']

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

    highlight_regions(simulation_data, ax1)
    highlight_regions(simulation_data, ax2)

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
        if (controller_sim_data['is_placeholder'] or
        controller_sim_data['is_vanilla'] == True):
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
        ncol=num_columns,
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

    return fig_profile

def plot_cartesian_state(simulation_data, controller_sim_datasets, num_columns=3):

    vanilla_VIC_controller_sim_data = None
    for controller_sim_data in controller_sim_datasets:
        if (controller_sim_data['is_vanilla'] == True):
            vanilla_VIC_controller_sim_data = controller_sim_data
            break
    assert (vanilla_VIC_controller_sim_data is not None)
    label_nominal = vanilla_VIC_controller_sim_data['label']

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
    highlight_regions(simulation_data, ax1)
    highlight_regions(simulation_data, ax2)
    highlight_regions(simulation_data, ax3)

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
        if (controller_sim_data['is_placeholder'] or
        controller_sim_data['is_vanilla'] == True):
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
        ncol=num_columns,
        bbox_to_anchor=(0.5, 1.6),
        loc='upper center'
    )
    '''

    fig_state_meas.align_ylabels([ax1, ax2, ax3])
    ax1.set_xlim((0., np.max(simulation_data['time'])))

    return fig_state_meas



# plot passivity related data
def plot_z_dot_z_and_beta(simulation_data, controller_sim_datasets, num_columns=3):

    vanilla_VIC_controller_sim_data = None
    for controller_sim_data in controller_sim_datasets:
        if (controller_sim_data['is_vanilla'] == True):
            vanilla_VIC_controller_sim_data = controller_sim_data
            break
    assert (vanilla_VIC_controller_sim_data is not None)
    label_nominal = vanilla_VIC_controller_sim_data['label']
    z_dot_vanilla = vanilla_VIC_controller_sim_data['z_dot']
    vanilla_z_dot_integral = vanilla_VIC_controller_sim_data['z_dot_integral']

    # ----------------------------------
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
    highlight_regions(simulation_data, ax1)
    highlight_regions(simulation_data, ax2)
    highlight_regions(simulation_data, ax3)

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
    for controller_sim_data in controller_sim_datasets:
        skip_this = (
            controller_sim_data['is_placeholder'] or
            (controller_sim_data['is_vanilla'] == True)
        )
        if (not skip_this):
            available_keys = \
                controller_sim_data['controller'].controller_log.keys()
            skip_this = skip_this or ('beta' not in available_keys)
            skip_this = skip_this or ('z_dot' not in available_keys)
            print(
                'Controller "'
                + controller_sim_data['label']
                + '" doesnt have the beta and/or z_dot field! Skipping'
            )
        if (skip_this):
            ax1.plot([], [], label = '_hh')
            ax2.plot([], [], label = '_hh')
            ax3.plot([], [], label = '_hh')
            continue

        # z_dot
        ax1.plot(
            simulation_data['time'],
            controller_sim_data['controller'].controller_log['z_dot'],
            label = controller_sim_data['label']
        )
        # Integral of z_dot
        z_key = 'z'
        if (not z_key in available_keys):
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
        ncol=num_columns,
        bbox_to_anchor=(0.5, 1.6),
        loc='upper center',
    )  # , framealpha=0.5)

    fig_z_z_dot_beta.align_ylabels([ax1, ax2, ax3])
    ax1.set_xlim((0., np.max(simulation_data['time'])))

    return fig_z_z_dot_beta


def plot_vic_tracking_errors(simulation_data, controller_sim_datasets, num_columns=3):
    # Create figure
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

    highlight_regions(simulation_data, ax1)
    highlight_regions(simulation_data, ax2)
    highlight_regions(simulation_data, ax3)
    highlight_regions(simulation_data, ax4)

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
    for controller_sim_data in controller_sim_datasets:
        skip_this = (
            controller_sim_data['is_placeholder'] or
            (controller_sim_data['is_vanilla'] == True)
        )

        if (skip_this):
            ax1.plot([], [], label = '_hh')
            ax2.plot([], [], label = '_hh')
            ax3.plot([], [], label = '_hh')
            ax4.plot([], [], label = '_hh')
            continue

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
        ncol=num_columns,
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

    return fig_vic_errors

# %%
import numpy as np
import matplotlib.pyplot as plt
from vic_controllers.plotting import multi_format_savefig, init_plt

init_plt(full_screen = False, scale = 1, use_latex=True)

Ts = 0.001
t = np.linspace(0, 10, int(10*1/Ts))

t_1 = 4
t_2 = 6

t1_idx = 1 + int(t_1*1/Ts)
t2_idx = 1 + int(t_2*1/Ts)

dissipation_rate = 0.14
tau_delay_z = 1
coef_filter_z = np.exp(-Ts/tau_delay_z)

z = np.zeros(t.shape)
z_bar = np.zeros(t.shape)

z[:t1_idx+1] = dissipation_rate * t[:t1_idx+1]

for idx in range(1, t1_idx+1):
    z_bar[idx] = coef_filter_z * z_bar[idx-1] + (1-coef_filter_z) * z[idx]

active_usage = z[t1_idx] - z_bar[t1_idx]

z[t1_idx:t2_idx+1] = z[t1_idx] - active_usage
z[t2_idx:] = z[t2_idx] + dissipation_rate * (t[t2_idx:] - t_2)


for idx in range(1, t.shape[0]):
    z_bar[idx] = coef_filter_z * z_bar[idx-1] + (1-coef_filter_z) * z[idx]


fig = plt.figure( figsize=(plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*0.4))
plt.plot(t, z, label = r'${z}(t)$')
plt.plot(t, z_bar, '--', label = r'$z_{min}(t) = \bar{z}(t)$')


plt.gca().fill_between(
    t, z, y2=z_bar,
    hatch='',  # '///',
    alpha=0.1,
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
)

plt.gca().fill_between(
    t, z_bar, y2=0,
    hatch='',  # '///',
    alpha=0.1,
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
)

plt.xlim((0, 10))
plt.ylim((0, 1 ))

plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')

plt.legend()

import os
export_figs_dir = "export_figures/illustrations"
if not os.path.exists(export_figs_dir):
    # Create a new directory because it does not exist
    os.makedirs(export_figs_dir)
    print(f"The directory {export_figs_dir} was created!")


multi_format_savefig(
    figure = fig,
    dir_name = export_figs_dir,
    fig_name = "illustration_z_min_adaptative"
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
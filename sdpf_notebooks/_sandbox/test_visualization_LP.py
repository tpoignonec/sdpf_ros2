# %%
# Special deps here:
#   pip install ipywidgets

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scipy_opt

from copy import deepcopy
from scipy.linalg import block_diag

from vic_controllers.plotting import multi_format_savefig, init_plt
init_plt(full_screen = False, scale = 1, use_latex=True)

%load_ext autoreload
%autoreload 2
from LPP_QPP_utils import *

LPP_in = {
    'err': - -0.19162826108213468,
    'err_dot': -0.07989518355874844,
    'f_ext': 2.0,
    'M': 5.,
    'D': 11.16512517,
    'K': 10.29951067,
    'M_d': np.array([[5.]]),
    'D_d': np.array([[11.21487945]]),
    'K_d': np.array([[172.25204068]]),
    'beta_max': 100.,
    'c': - np.array([1., 1., 1.]),
}
LPP_in['alpha'] = np.min(LPP_in['D_d'])/np.max(LPP_in['M_d'])

# ----------------------------------------------------------

LPP_out = build_LPP(LPP_in)
QPP_out = build_QPP(LPP_in, use_slack=False)

print('\n\n\n')
print('A_ub = ', LPP_out['A_ub'])
print('b_ub = ', LPP_out['b_ub'])

print('QPP sol = ', QPP_out['result']['x'])
print(
    'QPP passivity respected? ->',
    bool(LPP_out['A_ub'] @ np.array(LPP_out['result']['x'][:3]) <= LPP_out['b_ub'])
)

print('LPP sol = ', LPP_out['result']['x'])
print(
    'QPP passivity respected? ->',
    bool(LPP_out['A_ub'] @ np.array(QPP_out['result']['x'][:3]) <= LPP_out['b_ub'])
)
print('\n\n\n')

# ----------------------------------------------------------
range_beta, mesh_D, mesh_K = get_2D_beta_meshgrid(LPP_in)
_, _, mesh_M = get_2D_beta_meshgrid(LPP_in)

QP_cost_map_DK = np.zeros(mesh_D.shape)
for i in range(mesh_D.shape[0]):
    for j in range(mesh_D.shape[1]):
        QP_cost_map_DK[i, j] = QP_cost(LPP_out, 0.0, mesh_D[i, j], mesh_K[i, j])

QP_cost_map_DM = np.zeros(mesh_M.shape)
for i in range(mesh_D.shape[0]):
    for j in range(mesh_D.shape[1]):
        QP_cost_map_DM[i, j] = QP_cost(LPP_out, mesh_M[i, j], mesh_D[i, j], 0.0)

# ----------------------------------------------------------
# Solution and feasible region in D/K plane
# ----------------------------------------------------------

fig1 = plt.figure(
    figsize=(
        plt.rcParams["figure.figsize"][0],
        plt.rcParams["figure.figsize"][0] * 0.8
    )
)
ax1 = plt.gca()

plot_LPP_for_D_and_K(LPP_out, ax=ax1, with_LP_cost=False)
ax1.contour(
    mesh_D, mesh_K,
    LPP_out['c'][0, 1] * mesh_D + LPP_out['c'][0, 2] * mesh_K,
    30,
    colors='b',
    linestyles='solid',
    alpha=0.3,
)

ax1.plot(
    QPP_out['result']['x'][1],
    QPP_out['result']['x'][2],
    'g*',
    markersize=8,
    label='Optimal QP solution'
)

contours_KM = ax1.contour(
    mesh_D, mesh_K,
    QP_cost_map_DK,
    20,
    colors='g',
    linestyles='solid',
    alpha=0.3,
)

# plt.colorbar()
legend = ax1.legend(
    frameon=True,
    facecolor='white',
    framealpha=1,
    loc='upper center', bbox_to_anchor=(0.5, -0.15),
    ncol=2
)
# contours labels
ax1.clabel(
    contours_KM, contours_KM.levels,
    inline_spacing=15., inline=True, fontsize=6
)

# ----------------------------------------------------------
# Solution and feasible region in D/M plane
# ----------------------------------------------------------
fig2 = plt.figure(
    figsize=(
        plt.rcParams["figure.figsize"][0],
        plt.rcParams["figure.figsize"][0]*0.8
    )
)
ax2 = plt.gca()

plot_LPP_for_D_and_M(LPP_out, ax=ax2)

ax2.contour(
    mesh_D, mesh_M,
    LPP_out['c'][0, 1] * mesh_D + LPP_out['c'][0, 0] * mesh_M,
    30,
    colors='b',
    linestyles='solid',
    alpha=0.3,
)

ax2.plot(
    QPP_out['result']['x'][1],
    QPP_out['result']['x'][0],
    'g*',
    markersize=8,
    label='Optimal QP solution'
)

contours_DM = ax2.contour(
    mesh_D, mesh_M,
    QP_cost_map_DM,
    20,
    colors='g',
    linestyles='solid',
    alpha=0.3,
)

# contours labels
ax2.clabel(
    contours_DM, contours_DM.levels,
    inline_spacing=15., inline=True, fontsize=6
)

# plt.colorbar()
legend = ax2.legend(
    frameon=True,
    facecolor='white',
    framealpha=1,
    loc='upper center', bbox_to_anchor=(0.5, -0.15),
    ncol=2
)




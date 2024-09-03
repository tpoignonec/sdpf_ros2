import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from scipy.linalg import block_diag
import scipy.optimize as scipy_opt


# -----------------------------------------------------------------------------
# LP problem (LPP)
# -----------------------------------------------------------------------------

def build_LPP(LPP_in):
    err = np.asarray(LPP_in['err']).reshape(-1, 1)
    err_dot = np.asarray(LPP_in['err_dot']).reshape(-1, 1)
    f_ext = np.asarray(LPP_in['f_ext']).reshape(-1, 1)

    M = np.atleast_2d(np.asarray(LPP_in['M']))
    D = np.atleast_2d(np.asarray(LPP_in['D']))
    K = np.atleast_2d(np.asarray(LPP_in['K']))
    M_d = np.atleast_2d(np.asarray(LPP_in['M_d']))
    D_d = np.atleast_2d(np.asarray(LPP_in['D_d']))
    K_d = np.atleast_2d(np.asarray(LPP_in['K_d']))

    alpha = float(LPP_in['alpha'])
    beta_max = float(LPP_in['beta_max'])

    # linear constraints:
    # cst_rhs + cst_b_M * beta_M + cst_b_D * beta_D + cst_b_K * beta_K  >= 0
    # ==> -1 * cst_b_M * beta_M - cst_b_D * beta_D - cst_b_K * beta_K  <= cst_rhs
    cst_rhs = err_dot.reshape((1, -1))  @ (D - alpha * M) @ err_dot.reshape((-1, 1)) \
        + alpha * err.reshape((1, -1))  @ K @ err.reshape((-1, 1))
    cst_beta_M = - 0.5 *err_dot.reshape((1, -1))  @ (M_d - M) @ err_dot.reshape((-1, 1)) \
        - alpha *  err.reshape((1, -1)) @ (M_d - M) @ err_dot.reshape((-1, 1))
    cst_beta_D = - 0.5 * err.reshape((1, -1))  @ (D_d - D) @ err.reshape((-1, 1))
    cst_beta_K = - 0.5 * err.reshape((1, -1))  @ (K_d - K) @ err.reshape((-1, 1))

    # linprog LPP
    LPP_out = {}
    LPP_out['LPP_in'] = deepcopy(LPP_in)
    LPP_out['c'] = deepcopy(np.asarray(LPP_in['c'])).reshape((1, -1))
    LPP_out['lb'] = cst_beta_M

    LPP_out['lb'] = np.zeros((3,))
    LPP_out['ub'] = beta_max * np.ones((3,))
    LPP_out['A_ub'] = -1 * np.array([cst_beta_M, cst_beta_D, cst_beta_K]).reshape((1, -1))
    LPP_out['b_ub'] = np.asarray([cst_rhs])
    LPP_out['bounds'] = [(LPP_out['lb'][i], LPP_out['ub'][i]) for i in range(3)]

    if np.allclose(LPP_out['A_ub'][0, :], 0.0):
        assert (LPP_out['A_ub'].shape[0] == 1)
        LPP_out['result'] = scipy_opt.linprog(
            method='highs-ds',
            c=LPP_out['c'],
            bounds=LPP_out['bounds']
        )
    else :
        LPP_out['result'] = scipy_opt.linprog(
            method='highs-ds',
            c=LPP_out['c'],
            A_ub=LPP_out['A_ub'],
            b_ub=LPP_out['b_ub'],
            bounds=LPP_out['bounds']
        )

    LPP_out['augmented_A_ub'] = np.concatenate([LPP_out['A_ub'], np.eye(3), - np.eye(3)], axis=0)
    LPP_out['augmented_b_ub'] = np.concatenate([LPP_out['b_ub'].reshape((-1,)), LPP_out['ub'], -1 * LPP_out['lb']])

    return LPP_out

# -----------------------------------------------------------------------------
# QP problem (QPP)
# -----------------------------------------------------------------------------

def QP_cost_matrices(LPP_out, Ts = 0.005):
    """Get the H and g matrices for the QP cost function: 0.5 * x^T H x + g^T x
    """
    err = np.asarray(LPP_out['LPP_in']['err']).reshape((-1, 1))
    err_dot = np.asarray(LPP_out['LPP_in']['err_dot']).reshape((-1, 1))
    f_ext = np.asarray(LPP_out['LPP_in']['f_ext']).reshape((-1, 1))
    M = np.atleast_2d(LPP_out['LPP_in']['M'])
    D = np.atleast_2d(LPP_out['LPP_in']['D'])
    K = np.atleast_2d(LPP_out['LPP_in']['K'])
    M_d = np.atleast_2d(LPP_out['LPP_in']['M_d'])
    D_d = np.atleast_2d(LPP_out['LPP_in']['D_d'])
    K_d = np.atleast_2d(LPP_out['LPP_in']['K_d'])
    alpha = float(LPP_out['LPP_in']['alpha'])
    beta_max = float(LPP_out['LPP_in']['beta_max'])
    dim = err.shape[0]

    err_dot2_hat = np.linalg.inv(M) @ (-f_ext - D @ err_dot - K @ err)
    err_M = M_d - M
    err_D = D_d - D
    err_K = K_d - K
    R_mat = np.zeros((dim, 3))
    R_mat[:, 0] = - Ts * err_M @ err_dot2_hat
    R_mat[:, 1] = - Ts * err_D @ err_dot
    R_mat[:, 2] = - Ts * err_K @ err
    d_vect = - (err_M @ err_dot2_hat + err_D @ err_dot + err_K @ err)

    H_mat = R_mat.T @ R_mat
    g_vect = - R_mat.T @ d_vect

    #  add a small value to the diagonal to make the matrix positive definite
    #  and avoid numerical issues
    #  -> cost += eps * || IB - B_max ||^2
    # such that
    #     H += eps * I
    #     g += eps * B_max
    eps = 1. / beta_max**2
    vect_beta_values_max = beta_max * np.ones((3, 1))

    H_mat += eps * np.eye(3)
    g_vect += - eps * vect_beta_values_max

    return H_mat, g_vect

def QP_cost(LPP_out, beta_M, beta_D, beta_K, Ts = 0.005):
    """Get the cost of the QP problem
    """
    H_mat, g_vect = QP_cost_matrices(LPP_out, Ts)
    # QP
    vect_beta_values = np.array([beta_M, beta_D, beta_K]).reshape((3, 1))
    cost = 0.5 * vect_beta_values.T @ H_mat @ vect_beta_values \
        + g_vect.T @ vect_beta_values
    return cost

def build_QPP(LPP_in, use_slack = True, Ts = 0.005):
    import casadi as ca

    # Start by importing LPP
    LPP_out = build_LPP(LPP_in)
    H_mat_np, g_vect_np = QP_cost_matrices(LPP_out, Ts)

    # QP cost and constraints
    if use_slack:
        # QPP - with slack
        size_X = 4
        H_mat_np = block_diag(H_mat_np, 1e9 * np.eye(1))
        g_vect_np = np.concatenate([g_vect_np, np.array([[0.0]])], axis=0)

        constr_A_np = np.concatenate([LPP_out['A_ub'], np.array([[1.]])], axis=1)
        constr_uba_np = LPP_out['b_ub']
        lbx_np = np.concatenate([LPP_out['lb'], np.array([-np.inf])])
        ubx_np = np.concatenate([LPP_out['ub'], np.array([np.inf])])

        constr_A = ca.DM(constr_A_np)
        constr_uba = ca.DM(constr_uba_np)
        lbx = ca.DM(lbx_np)
        ubx = ca.DM(ubx_np)
    else:
        # QPP - no slack
        size_X = 3
        constr_A = ca.DM(LPP_out['A_ub'])
        constr_uba = ca.DM(LPP_out['b_ub'])
        lbx = ca.DM(LPP_out['lb'])
        ubx = ca.DM(LPP_out['ub'])
        '''
        constr_A = ca.DM(LPP_out['augmented_A_ub'])
        constr_lba = ca.DM(LPP_out['augmented_b_ub'])
        lbx = ca.DM(- np.inf * np.ones((3, 1)))
        ubx = ca.DM(np.inf * np.ones((3, 1)))
        '''
        if np.allclose(constr_A, 0.0, atol=1e-9):
            print('WARNING! QPP passivity constraint is ill-defined and will be disabled!')
            constr_A[0, :] = ca.DM(np.array([1., .0, .0]))
            constr_uba[0] = float(LPP_out['LPP_in']['beta_max'])

    # QP solver
    qp = {}
    H_mat = ca.DM(H_mat_np)
    g_vect = ca.DM(g_vect_np)
    qp['h'] = H_mat.sparsity()
    qp['a'] = constr_A.sparsity()
    opt = dict(
        verbose = True,
        print_in = True,
        print_out = True,
        print_problem = True,
        print_time = True
    )
    qp_solver = ca.conic('qp_solver_SDPF', 'qpoases', qp, opt)
    # Solve QP
    res = qp_solver(
        x0=np.zeros((size_X,)),
        h=H_mat,
        g=g_vect,
        a=constr_A,
        uba=constr_uba,
        lbx=lbx,
        ubx=ubx
    )

    '''
    from vic_controllers.controllers import SdpfController
    from vic_controllers.math import SpsdToolbox
    import casadi as ca

    controller_SDPF_independent_QP = SdpfController({
        'dim' : 1,
        'alpha' : alpha,
        'independent_beta_values' : True,
        'advanced_cost_function' : True,
        'filter_implementation' : 'QP',
        'beta_max' : beta_max,
        'verbose' : False,
        'N_logging' : None,
    })
    qp_solver = controller_SDPF_independent_QP.QP_solver

    # Solve QP "manually"
    params_solver = [
        alpha,
        beta_max,
        *err.flatten().tolist(),
        *err_dot.flatten().tolist(),
        *f_ext.flatten().tolist(),
        *M.flatten().tolist(),
        *D.flatten().tolist(),
        *K.flatten().tolist(),
        *M_d.flatten().tolist(),
        *D_d.flatten().tolist(),
        *K_d.flatten().tolist(),
    ]
    lbg = [0.0] * 3 + [0.0]
    ubg = [beta_max] * 3  + [1e9]

    result = qp_solver(
        x0=[0.0] * 3,
        p=params_solver,
        ubg=np.array(ubg).reshape((-1, 1)),
        lbg=np.array(lbg).reshape((-1, 1)),
    )
    '''
    # backup data
    QPP_out = {}
    QPP_out['qp_solver'] = deepcopy(qp_solver)
    QPP_out['result'] = deepcopy(res)
    return QPP_out


# -----------------------------------------------------------------------------
# Plotting utils
# -----------------------------------------------------------------------------

def get_feasibility_region(LPP_out, mesh_D, mesh_K):
    feasible_region = np.full((mesh_D.shape[0], mesh_D.shape[0]), True)

    for i in range(LPP_out['augmented_A_ub'].shape[0]):
        feasible_region &= (LPP_out['augmented_A_ub'][i, 1] * mesh_D
            + LPP_out['augmented_A_ub'][i, 2] * mesh_K) <= LPP_out['augmented_b_ub'][i]

    return feasible_region.astype(int)


def func_cstr_K_wrt_D(LPP_out):
    idx_row=0
    return lambda beta_D: \
        - (LPP_out['augmented_A_ub'][idx_row, 1] * beta_D - LPP_out['augmented_b_ub'][idx_row]) / LPP_out['augmented_A_ub'][idx_row, 2]

def func_cstr_M_wrt_D(LPP_out):
    idx_row=0
    return lambda beta_D: \
        - (LPP_out['augmented_A_ub'][idx_row, 1] * beta_D - LPP_out['augmented_b_ub'][idx_row]) / LPP_out['augmented_A_ub'][idx_row, 0]

def get_2D_beta_meshgrid(LPP_in):
    range_beta = np.linspace(- 1, LPP_in['beta_max'] + 9, 220)
    mesh_D, mesh_K = np.meshgrid(range_beta, range_beta)

    return range_beta, mesh_D, mesh_K

def plot_LPP_for_D_and_K(LPP_out, ax=None, with_LP_cost=True):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    range_beta, mesh_D, mesh_K = get_2D_beta_meshgrid(LPP_out['LPP_in'])

    if with_LP_cost:
        ax.contourf(
            mesh_D, mesh_K,
            LPP_out['c'][0, 1] * mesh_D + LPP_out['c'][0, 2] * mesh_K,
            30,
            cmap='winter',
            alpha=0.5,
            zorder=-2
        )

        clb = ax.colorbar()
        clb.ax.set_title(r'$c^T \beta$')

        unfeasible_region = get_feasibility_region(LPP_out, mesh_D, mesh_K)
        unfeasible_region = np.ma.masked_where(unfeasible_region == 1, unfeasible_region)

        plt.imshow(
            unfeasible_region,
            extent=(mesh_D.min(), mesh_D.max(), mesh_K.min(), mesh_K.max()),
            origin="lower",
            cmap="Greys",
            zorder=-1
        )

    if (LPP_out['augmented_A_ub'][0, 2] >= 1e-9):
        ax.plot(
            range_beta,
            func_cstr_K_wrt_D(LPP_out)(range_beta),
            '--r',
            label='Passivity constraint'
        )
    else:
        if (LPP_out['augmented_A_ub'][0, 1] >= 1e-9):
            ax.vlines(
                LPP_out['augmented_b_ub'][0]/LPP_out['augmented_A_ub'][0, 1],
                ymin=np.min(mesh_D),
                ymax=np.max(mesh_D),
                colors='red',
                linestyle='--',
                label='Passivity constraint'
            )
        else:
             ax.plot([], [], '--r', label='Passivity constraint')

    ax.plot(
        (0.0, LPP_out['LPP_in']['beta_max'], LPP_out['LPP_in']['beta_max'], 0.0, 0.0),
        (0.0, 0.0, LPP_out['LPP_in']['beta_max'], LPP_out['LPP_in']['beta_max'], 0.0),
        '-k',
        label=r'Bounds of $\beta$ variables'
    )

    ax.plot(
        LPP_out['result']['x'][1],
        LPP_out['result']['x'][2],
        'bo',
        # markersize=10,
        label='Optimal LP solution'
    )
    # plt.grid()
    ax.set_xlim(np.min(mesh_D), np.max(mesh_D))
    ax.set_ylim(np.min(mesh_K), np.max(mesh_K))
    ax.set_xlabel(r'$\beta_D$')
    ax.set_ylabel(r'$\beta_K$')
    # plt.legend()

    return ax

def plot_LPP_for_D_and_M(LPP_out, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    range_beta, mesh_D, mesh_M = get_2D_beta_meshgrid(LPP_out['LPP_in'])

    if (LPP_out['augmented_A_ub'][0, 0] >= 1e-9):
        ax.plot(
            range_beta,
            func_cstr_M_wrt_D(LPP_out)(range_beta),
            '--r',
            label='Passivity constraint'
        )
    else:
        if (LPP_out['augmented_A_ub'][0, 1] >= 1e-9):
            ax.vlines(
                LPP_out['augmented_b_ub'][0]/LPP_out['augmented_A_ub'][0, 1],
                ymin=np.min(mesh_D),
                ymax=np.max(mesh_D),
                colors='red',
                linestyle='--',
                label='Passivity constraint'
            )
        else:
             ax.plot([], [], '--r', label='Passivity constraint')

    ax.plot(
        (0.0, LPP_out['LPP_in']['beta_max'], LPP_out['LPP_in']['beta_max'], 0.0, 0.0),
        (0.0, 0.0, LPP_out['LPP_in']['beta_max'], LPP_out['LPP_in']['beta_max'], 0.0),
        '-k',
        label=r'Bounds of $\beta$ variables'
    )

    ax.plot(
        LPP_out['result']['x'][1],
        LPP_out['result']['x'][0],
        'bo',
        # markersize=10,
        label='Optimal LP solution'
    )
    # ax.grid()
    ax.set_xlim(np.min(mesh_D), np.max(mesh_D))
    ax.set_ylim(np.min(mesh_M), np.max(mesh_M))
    ax.set_xlabel(r'$\beta_D$')
    ax.set_ylabel(r'$\beta_M$')
    # ax.legend()

    return ax

def plot_LPP_for_3D(LPP_out):
    pass

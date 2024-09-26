import numpy as np
from copy import deepcopy


def make_simulation_data(scenario: str = 'scenario_1'):
    if (scenario == 'scenario_1'):
        return make_simulation_data_scenario_1()
    elif (scenario == 'scenario_variable_inertia'):
        return make_scenario_variable_inertia()
    elif (scenario == 'scenario_1_K_only'):
        sim_data = make_simulation_data_scenario_1()
        for i in range(1, sim_data['N']):
            sim_data['D_d'][i] = sim_data['D_d'][0]
            sim_data['D_d_dot'][i] = 0.0
        return sim_data
    else:
        raise NotImplementedError()


def make_simulation_data_scenario_1():
    sim_data = {}

    t_max = 24  # seconds
    sim_data['Ts'] = 0.001  # seconds
    sim_data['N'] = int(t_max/sim_data['Ts']) + 1
    sim_data['time'] = np.linspace(0, sim_data['N']*sim_data['Ts'], sim_data['N'])

    real_mass = 5.0
    sim_data['real_mass'] = real_mass
    sim_data['model_mass'] = deepcopy(real_mass)

    sim_data['simulation_phase'] = np.chararray((sim_data['N'],))
    sim_data['simulation_phase'][range(0, 6*int(1/sim_data['Ts']))] = b'a'
    sim_data['simulation_phase'][range(6*int(1/sim_data['Ts']), 12*int(1/sim_data['Ts']))] = b'a'
    sim_data['simulation_phase'][12*int(1/sim_data['Ts']):] = b'c'

    # Dummy reference trajectory
    # sin_A = 0.05
    # sin_w = 0.1
    sim_data['X_d'] = np.zeros((sim_data['N'], 3))  # [p, v, a]'
    sim_data['X_d'][:, 0] = 0.15  # sin_A*np.sin(sin_w*sim_data['time'])
    sim_data['X_d'][:, 1] = 0.0  # sin_A*sin_w*np.cos(sin_w*sim_data['time'])
    sim_data['X_d'][:, 2] = 0.0  # -sin_A*(sin_w**2)*np.sin(sin_w*sim_data['time'])

    # Dummy (variable) impedance profil
    K_min = 10
    K_max = 200
    K_w = (2 * np.pi) / (t_max / 6)  # 5*np.pi/10
    K_phi = -np.pi/2

    D_damp_ratio = 0.2
    D_min = 2*D_damp_ratio*np.sqrt(real_mass*K_min)
    D_max = 2*D_damp_ratio*np.sqrt(real_mass*K_max)
    D_w = K_w
    D_phi = K_phi

    func_K_d = lambda time: K_min + (K_max-K_min)*(1 + np.sin(K_w*time + K_phi))/2  # noqa: E731
    func_K_d_dot = lambda time: (K_max-K_min)*K_w*np.cos(K_w*time + K_phi)/2  # noqa: E731

    func_D_d = lambda time: D_min + (D_max-D_min)*(1 + np.sin(D_w*time + D_phi))/2  # noqa: E731
    # 2*np.sqrt(D_damp_ratio*real_mass*func_K_d(time))
    func_D_d_dot = lambda time: (D_max-D_min)*D_w*np.cos(D_w*time + D_phi)/2  # noqa: E731
    # (D_damp_ratio*real_mass*func_K_d_dot(time))/np.sqrt(D_damp_ratio*real_mass*func_K_d(time))

    sim_data['K_d'] = func_K_d(sim_data['time'])
    sim_data['K_d_dot'] = func_K_d_dot(sim_data['time'])
    sim_data['D_d'] = func_D_d(sim_data['time'])
    sim_data['D_d_dot'] = func_D_d_dot(sim_data['time'])
    sim_data['M_d'] = real_mass*np.ones(sim_data['time'].shape)
    sim_data['M_d_dot'] = np.zeros(sim_data['time'].shape)

    # Simulate a perturbation (traj. OR elastic link to X0)
    t_1 = 2 * (2 * np.pi) / K_w
    t_2 = 2 * t_1
    sim_data['t1'] = t_1
    sim_data['t2'] = t_2
    sim_data['stiffness_env'] = np.zeros((sim_data['N'],))
    sim_data['stiffness_env'][range(0, int(t_1/sim_data['Ts']))] = 50.0
    sim_data['f_ext_extra'] = np.zeros((sim_data['N'],))
    sim_data['f_ext_extra'] = np.array([2.0/(1 + np.exp(50 * (t_2 - sim_data['time'])))]).reshape((-1,))

    # Initial state (position and velocity of the 1D mass)
    position_at_t0 = sim_data['X_d'][0, 0] / (1 + sim_data['stiffness_env'][0] / sim_data['K_d'][0])
    sim_data['x0_mass_spring_damper'] = np.array([position_at_t0, 0.0])

    return sim_data


def make_scenario_variable_inertia():
    sim_data = {}

    t_max = 24  # seconds
    sim_data['Ts'] = 0.001  # seconds
    sim_data['N'] = int(t_max/sim_data['Ts']) + 1
    sim_data['time'] = np.linspace(0, sim_data['N']*sim_data['Ts'], sim_data['N'])

    real_mass = 5.0
    sim_data['real_mass'] = real_mass
    sim_data['model_mass'] = deepcopy(real_mass)

    sim_data['simulation_phase'] = np.chararray((sim_data['N'],))
    sim_data['simulation_phase'][range(0, 6*int(1/sim_data['Ts']))] = b'a'
    sim_data['simulation_phase'][range(6*int(1/sim_data['Ts']), 12*int(1/sim_data['Ts']))] = b'a'
    sim_data['simulation_phase'][12*int(1/sim_data['Ts']):] = b'c'

    # Dummy reference trajectory
    # sin_A = 0.05
    # sin_w = 0.1
    sim_data['X_d'] = np.zeros((sim_data['N'], 3))  # [p, v, a]'
    sim_data['X_d'][:, 0] = 0.15  # sin_A*np.sin(sin_w*sim_data['time'])
    sim_data['X_d'][:, 1] = 0.0  # sin_A*sin_w*np.cos(sin_w*sim_data['time'])
    sim_data['X_d'][:, 2] = 0.0  # -sin_A*(sin_w**2)*np.sin(sin_w*sim_data['time'])

    # Dummy (variable) impedance profil
    K_d = 200.0

    M_min = 1.0
    M_max = 10.0

    M_w = (2 * np.pi) / (t_max / 6)  # 5*np.pi/10
    M_phi = -np.pi/2

    D_damp_ratio = 0.2
    
    func_K_d = lambda time: K_d * np.ones(time.shape)  # noqa: E731
    func_K_d_dot = lambda time: K_d * np.ones(time.shape)  # noqa: E731

    func_M_d = lambda time: M_min + (M_max-M_min)*(1 + np.sin(M_w*time + M_phi))/2  # noqa: E731
    func_M_d_dot = lambda time: (M_max-M_min)*M_w*np.cos(M_w*time + M_phi)/2  # noqa: E731

    func_D_d = lambda time: 2*np.sqrt(D_damp_ratio*func_M_d(time)*func_K_d(time))
    func_D_d_dot = lambda time: D_damp_ratio * (
        func_M_d(time)*func_K_d_dot(time) + func_M_d_dot(time)*func_K_d(time)) \
            / (np.sqrt(D_damp_ratio*func_M_d(time)*func_K_d(time)))

    sim_data['K_d'] = func_K_d(sim_data['time'])
    sim_data['K_d_dot'] = func_K_d_dot(sim_data['time'])
    sim_data['D_d'] = func_D_d(sim_data['time'])
    sim_data['D_d_dot'] = func_D_d_dot(sim_data['time'])
    sim_data['M_d'] = func_M_d(sim_data['time'])
    sim_data['M_d_dot'] = func_M_d_dot(sim_data['time'])

    # Simulate a perturbation (traj. OR elastic link to X0)
    t_1 = 2 * (2 * np.pi) / M_w
    t_2 = 2 * t_1
    sim_data['t1'] = t_1
    sim_data['t2'] = t_2
    sim_data['stiffness_env'] = np.zeros((sim_data['N'],))
    sim_data['stiffness_env'][range(0, int(t_1/sim_data['Ts']))] = 50.0
    sim_data['f_ext_extra'] = np.zeros((sim_data['N'],))
    sim_data['f_ext_extra'] = np.array([2.0/(1 + np.exp(50 * (t_2 - sim_data['time'])))]).reshape((-1,))

    # Initial state (position and velocity of the 1D mass)
    position_at_t0 = sim_data['X_d'][0, 0] / (1 + sim_data['stiffness_env'][0] / sim_data['K_d'][0])
    sim_data['x0_mass_spring_damper'] = np.array([position_at_t0, 0.0])

    return sim_data

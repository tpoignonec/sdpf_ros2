# Copyright 2024 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)

from copy import deepcopy
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from tqdm import tqdm
# import typing

from vic_controllers.commons import MeasurementData
from vic_controllers.simulation import (
    build_simulator,
    export_linear_mass_model
)


# ----------------------------------------------------------------------------
#                      2D Obstacles (cercles, walls, etc.)
# ----------------------------------------------------------------------------


class Obstacle2D:
    def __init__(self):
        pass

    def get_force(self, p: np.ndarray, p_dot: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def plot(self, ax, **kwargs):
        raise NotImplementedError


class Wall2D(Obstacle2D):
    def __init__(self, p: np.ndarray, n: np.ndarray, k: float, d: float):
        """2D wall obstacle.

        :param p: point belonging to the wall
        :type p: np.ndarray
        :param n:  normal vector to the wall (towards the wall's outside)
        :type n: np.ndarray
        :param k: stiffness of the wall (N/m)
        :type k: float
        :param d: damping of the wall (N.s/m)
        :type d: float
        """
        self._p = p
        self._n = n
        self._k = k
        self._d = d

    def get_force(self, p: np.ndarray, p_dot: np.ndarray) -> np.ndarray:
        """Compute force applied by the wall on a point p."""

        # Compute distance to wall
        dist_to_wall = float(np.dot(p - self._p, self._n))
        vel_robot_along_n = float(np.dot(p_dot, self._n))

        # Compute force
        f_ext = np.zeros((2,))
        if (dist_to_wall < 0.0):
            f_ext = \
                - self._k * dist_to_wall * self._n \
                - self._d * vel_robot_along_n * self._n
        return f_ext

    def plot(self, ax, **kwargs):
        print("Warning: Wall2D.plot() not implemented yet.")
        pass


class ConstantForce2D(Obstacle2D):
    def __init__(self, f_ext: np.ndarray):
        """2D constant force.

        :param f_ext: constant force applied to the robot
        :type f_ext: np.ndarray
        """
        self._f_ext = f_ext

    def get_force(self, p: np.ndarray, p_dot: np.ndarray) -> np.ndarray:
        return self._f_ext

    def plot(self, ax, **kwargs):
        # Nothing to plot
        pass


class ViscosityField2D(Obstacle2D):
    def __init__(self, damping_x: float, damping_y: float):
        """2D viscosity field.

        :param damping_x: Damping along X axis (N.s/m)
        :type damping_x: float
        :param damping_y: Damping along Y axis (N.s/m)
        :type damping_y: float
        """
        self._d_x = damping_x
        self._d_y = damping_y

    def get_force(self, p: np.ndarray, p_dot: np.ndarray) -> np.ndarray:
        D = np.array([[self._d_x, 0.0], [0.0, self._d_y]])
        return - np.dot(D, p_dot)

    def plot(self, ax, **kwargs):
        # Nothing to plot
        pass


class Sphere2D(Obstacle2D):
    def __init__(
        self,
        p: np.ndarray,
        r: float,
        k: float,
        d: float
    ):
        """2D sphere obstacle.

        :param p: center of the sphere
        :type p: np.ndarray
        :param r: radius of the sphere
        :type r: float
        :param k: stiffness of the sphere (N/m)
        :type k: float
        :param d: damping of the sphere (N.s/m)
        :type d: float
        """
        self._p = p
        self._r = r
        self._k = k
        self._d = d

    def get_force(self, p: np.ndarray, p_dot: np.ndarray) -> np.ndarray:
        """Compute force applied by the sphere on a point p."""

        # Compute distance to sphere
        dist_center_to_p = np.linalg.norm(p - self._p)
        dist_to_sphere = dist_center_to_p - self._r
        normal_to_sphere = (p - self._p) / dist_center_to_p

        # Compute force
        f_ext = np.zeros((2,))
        if (dist_to_sphere < 0.0):
            f_ext = \
                - self._k * dist_to_sphere * normal_to_sphere \
                - self._d * p_dot
        return f_ext

    def plot(self, ax, **kwargs):
        alpha = kwargs.get('alpha', 1.0)
        circle_patch = plt.Circle(
            (self._p[0], self._p[1]), self._r, color='k', alpha=alpha)
        return ax.add_patch(circle_patch)


# ----------------------------------------------------------------------------
#                    2D Environment (robot, obstacles, etc.)
# ----------------------------------------------------------------------------


class Environment2D:
    def __init__(self, simulation_settings: dict = {}):
        self._p = deepcopy(simulation_settings['p_at_t0']).reshape((-1,))
        if 'p_dot_at_t0' in simulation_settings:
            self._p_dot = deepcopy(
                simulation_settings['p_dot_at_t0']).reshape((-1,))
        else:
            self._p_dot = np.zeros((2,))

        self._Ts = deepcopy(simulation_settings['Ts'])
        self._robot_inertia_real = deepcopy(
            simulation_settings['robot_inertia_real'])

        # Build simulator
        dyn_model = export_linear_mass_model(
            inertia=self._robot_inertia_real,
            f_ext_as_param=True,
            dim=2
        )
        self._dyn_model_integrator = build_simulator(
            dyn_model,
            time_step=self._Ts
        )
        self._dyn_model_integrator.set(
            "seed_adj", np.ones((dyn_model.x.size()[0], 1)))

        # Obstacles
        self._obstacles = []
        self.recompute_f_ext()

    def add_obstacle(self, obstacle: Obstacle2D):
        assert isinstance(obstacle, Obstacle2D)
        self._obstacles.append(obstacle)

    def compute_f_ext(self, p, p_dot) -> np.ndarray:
        f_ext = np.zeros((2,))
        for obs in self._obstacles:
            f_ext += obs.get_force(p, p_dot).reshape((-1,))
        return f_ext

    def recompute_f_ext(self):
        self._f_ext = self.compute_f_ext(self._p, self._p_dot)

    def get_measurements(self) -> MeasurementData:
        self.recompute_f_ext()
        # Package measurements
        measurements = MeasurementData(dimension=2)
        measurements.p = self._p
        measurements.p_dot = self._p_dot
        measurements.f_ext = self._f_ext
        return measurements

    def step(self, f_cmd: np.ndarray) -> MeasurementData:
        """Compute next measurements given current Cartesian command force.

        :param f_cmd: Cartesian command force (dim 2) at t_k
        :type f_cmd: np.ndarray

        :return: Next measurements (at t_k+1)
        :rtype: MeasurementData
        """
        assert (f_cmd.size == 2)

        # Compute forces from obstacles
        self.recompute_f_ext()

        # Simulate time step
        dyn_state = np.concatenate([
            self._p.reshape((-1,)), self._p_dot.reshape((-1,))])
        self._dyn_model_integrator.set("x", dyn_state)
        self._dyn_model_integrator.set("u", f_cmd.reshape((-1,)))
        self._dyn_model_integrator.set("p", self._f_ext)

        status = self._dyn_model_integrator.solve()
        if status != 0:
            raise Exception(
                f'2D env. simulator returned Acados status {status}.')
        X_next = self._dyn_model_integrator.get("x")
        self._p = X_next[:2]
        self._p_dot = X_next[2:]

        # Return measurements (AND compute f_ext in the process...)
        return self.get_measurements()

    # ----------------------------------------------------------------------------
    #                                Plotting
    # ----------------------------------------------------------------------------

    def create_plot(self, figsize=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(
            111, aspect='equal', autoscale_on=False,
            xlim=(-0.1, 1.0), ylim=(-0.5, 0.5)
        )
        ax.grid()
        return fig, ax

    def plot_desired_pose(self, ax, p_desired):
        return [ax.plot(
            p_desired[0], p_desired[1],
            'o',
            markersize=plt.rcParams["lines.markersize"]/4,
            color='green',
            label=r'$p^d$'
        )]

    def plot_desired_trajectory(self, ax, p_des_traj, **kwargs):
        return [ax.plot(
            p_des_traj[:, 0],
            p_des_traj[:, 1],
            '--',
            linewidth=plt.rcParams["lines.linewidth"]/2,
            color='red',
            label=r'$p^d$'
        )]

    def plot_robot(self, ax, **kwargs):
        # Plot robot
        ret = [ax.plot(
            self._p[0], self._p[1],
            'o',
            markersize=plt.rcParams["lines.markersize"]/4,
            color='blue',
            label=r'$p(t)$'
        )]

        # Plot f_ext
        if np.linalg.norm(self._f_ext) > 1e-6:
            ret += [ax.quiver(
                self._p[0], self._p[1],
                self._f_ext[0], self._f_ext[1],
                color='red',
                width=0.005,
                label=r'$f_{\text{ext}}$'
            )]
        return ret

    def plot_robot_trajectory(self, ax, p_traj, **kwargs):
        return ax.plot(
            p_traj[:, 0],
            p_traj[:, 1],
            '--',
            linewidth=plt.rcParams["lines.linewidth"]/2,
            color='blue',
            label=r'$p$'
        )

    def plot_obstacles(self, ax, **kwargs):
        # Plot obstacles
        ret = []
        for obs in self._obstacles:
            ret += [obs.plot(ax, **kwargs)]
        return ret

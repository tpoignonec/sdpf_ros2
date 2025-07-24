# sdpf_ros2
Ros2 nodes based on the [vic_controllers](https://github.com/tpoignonec/vic_python_controllers) python package.

# Dependencies

- Ubuntu 24.04;
- a working ROS2 Jazzy distribution;
- the `acados_template` vendor package [acados_vendor_ros2](https://github.com/tpoignonec/acados_vendor_ros2.git) (N.B., also shipped in [acados_controllers_ros2](https://github.com/tpoignonec/acados_controllers_ros2.git) below);
- the python package [vic_controllers](https://github.com/tpoignonec/vic_python_controllers.git) (shipped as a ros2 pkg by [vic_python_controllers_ros2](https://github.com/tpoignonec/vic_python_controllers_ros2.git));
- to launch the ros2 control-based scenario using Force Dimension devices, the [forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2.git) stack is required.
- (optional) to build the C++ examples using Acados NMPC solvers, the Acados ros2 stack [acados_solver_ros2](https://github.com/ICube-Robotics/acados_solver_ros2.git) is required;

# Installation

1) Prepare the ros2 workspace


```bash
# Go to/create ros2 workspace dir <ws>
export SDPF_WS=~/dev/ros2-jazzy/dev_sdpf_ros2
mkdir -p $SDPF_WS/src
cd $SDPF_WS/src

# Clone this repos
git clone https://github.com/tpoignonec/sdpf_ros2.git

# Clone dependencies
vcs import . < sdpf_ros2/sdpf_ros2.repos

cd ..
```

```bash
# Source ros2 distro
source /opt/ros/jazzy/setup.bash

# Install dependencies
PIP_BREAK_SYSTEM_PACKAGES=1 rosdep install --ignore-src --from-paths . -y -r

# Python retro-compatibility for acados
sudo apt install python3-pip -y
PIP_BREAK_SYSTEM_PACKAGES=1 pip install future-fstrings

# Manually install packages that don't have a rosdistro key
sudo apt install python3-scienceplots -y

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

```bash
# Source ros2 pkgs
source install/setup.bash
```

# How to run the simulations?

```bash
cd $SDPF_WS
source install/setup.bash

# Option 1: run one by one in interactive mode
code .  # launch vs code here
# - install Python3 and Jupyter extensions
# - open the file in vs code (see src/sdpf_ros2/sdpf_notebooks/**)
# - select Python env (ros2 python3 bin or local .venv if relevant)
# - run cell-by-cell

# Option 2: run all simulations + export figures

cd src/sdpf_ros2/sdpf_notebooks/simulations_SDPF

python3 01-A_simulation_1D_SIPF_vs_SDPF.py
python3 01-B_simulation_1D_variable_inertia.py
python3 02_illustration_z_min_adaptative.py
python3 03_simulation_1D_SIPF+_vs_SDPF_vs_SDPF-integrals.py
python3 04_simulation_1D_passivity_tank.py
python3 05_effect_of_z_max_and_tau_z_min.py
python3 06_effect_of_alpha_and_epsilon.py
```

# How to launch the experiment?

## Variable impedance control with Omega3 haptic interface

### 1) Launch the VIC controller

Connect the USB cable of the haptic device and setup rules.

> [!TIP]
> If using WSL2, bind the USB port first using `sudo usbip attach -r <host_ip> -b <bus_id>`.
> See [this Gist](https://gist.github.com/tpoignonec/762a108b25a460eb98e0d05412f4da18) for details about the binding procedure.

Setup the Ethercat Master:
- install the kernel module if needed (see the [documentation page]());
- start the master: `sudo /etc/init.d/ethercat start `;
- test that the bus is working: `ethercat slaves`.

Finally, launch the controller itself:

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup launch_fd_impedance_control.launch.py
```

### 2) Test the passive VIC filtering node


```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=impedance_ft_elastic trajectory_type:=static \
    pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

or with a circular trajectory

```bash
ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=impedance_ft_elastic trajectory_type:=circular \
    pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

### 3) Launch experiment and record data

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

export EXP_SERIES_NAME=<name_of_this_series_of_experiment>
# e.g., export EXP_SERIES_NAME=exp_october_12_2024

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=impedance_ft_elastic trajectory_type:=static \
    record_bags:=true \
    bag_path:=rosbags/$EXP_SERIES_NAME/ \
    pf_method:=SIPF

# CTRL + C at the end of the simulation (+- 15 seconds)

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=impedance_ft_elastic trajectory_type:=static \
    record_bags:=true \
    bag_path:=rosbags/$EXP_SERIES_NAME/ \
    pf_method:=SIPF+

# CTRL + C at the end of the simulation (+- 15 seconds)

# 3 more times with the other controllers: "SDPF", "SDPF-integral", and "SDPF-adaptive"

# You should have the following files:
# ws_sdpf_ros2/rosbags/<name_of_this_series_of_experiment>/SIPF/***
# ws_sdpf_ros2/rosbags/<name_of_this_series_of_experiment>/SIPF+/***
# ws_sdpf_ros2/rosbags/<name_of_this_series_of_experiment>/SDPF/***
# ws_sdpf_ros2/rosbags/<name_of_this_series_of_experiment>/SDPF-integral/***
# ws_sdpf_ros2/rosbags/<name_of_this_series_of_experiment>/SDPF-adaptive/***
```

## Variable admittance control with UR5 robot

### 1) Launch the VIC controller

Connect the robot and test the connection: `ping <robot_ip>`.

Connect the F/T sensor and setup the Ethercat Master:
- install the kernel module if needed (see the [documentation page]());
- start the master: `sudo /etc/init.d/ethercat start `;
- test that the bus is working: `ethercat slaves`.

Finally, launch the controller itself:

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup launch_ur5_admittance_control.launch.py
```

Alternatively, you can use a simulated robot:

```bash
ros2 launch sdpf_bringup launch_ur5_admittance_control.launch.py use_fake_hardware:=true
```

Although forces are not simulated, a dummy measurement can be provided manually as follows:

```bash
ros2 topic pub --rate 10 /dummy_ft_sensor_data geometry_msgs/Wrench "{force: {x: 0.0, y: 0., z: 0.0}}"
```

### 2) Test the passive VIC filtering node

> [!CAUTION]
> Make sure you have the E-stop near at hand!
> Also, check that the robot is near the starting desired position,
> no approach phase is implemented...

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=admittance_ur5_phri trajectory_type:=static \
    pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

or with a circular trajectory

```bash
ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=admittance_ur5_phri trajectory_type:=circular \
    pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

### 3) Launch experiment and record data

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

export EXP_SERIES_NAME=<name_of_this_series_of_experiment>
# e.g., export EXP_SERIES_NAME=exp_october_12_2024

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=admittance_ur5_phri trajectory_type:=circular \
    record_bags:=true \
    bag_path:=rosbags/$EXP_SERIES_NAME/ \
    pf_method:=SDPF

# CTRL + C at the end of the simulation (+- 15 seconds)

ros2 launch sdpf_bringup run_exp.launch.py \
    scenario:=admittance_ur5_phri trajectory_type:=circular \
    record_bags:=true \
    bag_path:=rosbags/$EXP_SERIES_NAME/ \
    pf_method:=SIPF

# CTRL + C at the end of the simulation (+- 15 seconds)

# 3 more times with the other controllers: "SDPF", "SDPF-integral", and "SDPF-adaptive"
```

## How to generate the figures?

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

# Go to SDPF notebooks package
cd src/sdpf_ros2/sdpf_notebooks

# Export figures for fd experiment
python3 plot_exp_1D_data.py --display-figs true --dataset <name_of_this_series_of_experiment>

# Or for UR5 experiment
python3 plot_exp_2D_data.py --display-figs true --dataset <name_of_this_series_of_experiment>

# Wait a bit, might take a few minutes...
# Then, you should have the figure files at
#   ~/dev/ros2_workspaces/ws_sdpf_ros2/src/sdpf_ros2/sdpf_notebooks/export_figures/exp_results-<name_of_this_series_of_experiment>/***
```

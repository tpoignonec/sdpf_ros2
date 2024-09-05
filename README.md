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
mkdir -p ~/dev/ros2_workspaces/ws_sdpf_ros2/src
cd ~/dev/ros2_workspaces/ws_sdpf_ros2/src

# Clone this repos
git clone https://github.com/tpoignonec/sdpf_ros2.git

# Clone dependencies
vcs import . < sdpf_ros2/sdpf_ros2.repos

cd ..
```

```bash
# Source ros2 distro
source /opt/ros/jazzy/setup.bash

sudo apt install ptython3-pip -y
# Install dependencies
export PIP_BREAK_SYSTEM_PACKAGES=1  # for casadi pip install
# !WARNING! Do not abuse PIP installs in this mode (unsafe...)!!!

rosdep install --ignore-src --from-paths . -y -r

pip install future-fstrings  # Python retro-compatibility for acados

# revert to "safe" pip policy
export PIP_BREAK_SYSTEM_PACKAGES=0

# Manually install packages that don't have a rosdistro key
sudo apt install ptython3-scienceplots -y

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

```bash
# Source ros2 pkgs
source install/setup.bash
```

# How to launch the experiment?

## 1) Launch the VIC controller

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup launch_impedance_control.launch.py
```

```bash
# If using WSL2, bind the USB port first
sudo usbip attach -r <host_ip> -b <bus_id>
```
__Note:__ see [this Gist](https://gist.github.com/tpoignonec/762a108b25a460eb98e0d05412f4da18) for details about the binding procedure.

## 2) Test the passive VIC node


```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup run_exp.launch.py pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

## 3) Launch experiment and record data

```bash
cd ~/dev/ros2_workspaces/ws_sdpf_ros2
source install/setup.bash

ros2 launch sdpf_bringup run_exp.launch.py \
    record_bags:=true \
    pf_method:=SIPF # SIPF / SIPF+ / SDPF / etc.
```

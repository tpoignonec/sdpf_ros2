# sdpf_ros2
Ros2 nodes based on the [vic_controllers](https://github.com/tpoignonec/vic_python_controllers) python package.

# Dependencies

- Ubuntu 24.04;
- a working ROS2 Jazzy distribution;
- the `acados_template` vendor package [acados_vendor_ros2](https://github.com/tpoignonec/acados_vendor_ros2.git) (N.B., also shipped in [acados_controllers_ros2](https://github.com/tpoignonec/acados_controllers_ros2.git) below);
- the python package [vic_controllers](https://github.com/tpoignonec/vic_python_controllers.git);
- to launch the ros2 control-based scenario using Force Dimension devices, the [forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2.git) stack is required.
- (optional) to build the C++ examples using Acados NMPC solvers, the Acados ros2 stack [acados_controllers_ros2](https://github.com/tpoignonec/acados_controllers_ros2.git) is required;

# Installation

1) Prepare the ros2 workspace

```bash
# Go to/create ros2 workspace dir <ws>
mkdir -p ~/dev/ros2_workspaces/ws_sdpf_controllers/src
cd ~/dev/ros2_workspaces/ws_sdpf_controllers/src

# Clone this repos
git clone https://github.com/tpoignonec/vic_controllers_ros2.git

# Clone dependencies
vcs import . < vic_controllers_ros2/vic_controllers_ros2.repos

# Make sure that colcon doesn’t try to build the pip packages
touch external-pip/COLCON_IGNORE
cd ..
```

2) Install the (ros2) dependencies and packages

```bash
# Source ros2 distro
source /opt/ros/jazzy/setup.bash

# Install dependencies
export PIP_BREAK_SYSTEM_PACKAGES=1  # !WARNING! Do not abuse PIP installs in this mode (unsafe...)!!!
rosdep install --ignore-src --from-paths . -y -r
pip install future-fstrings  # Python retro-compatibility

# revert to "safe" pip policy
export PIP_BREAK_SYSTEM_PACKAGES=0

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

3) Install the Python packages in a virtual environment

```bash
# Make a Python3 venv
sudo apt install virtualenv
virtualenv -p python3 .venv

# Make sure that colcon doesn’t try to build the venv later on
touch .venv/COLCON_IGNORE

# Source the env
source .venv/bin/activate  # or source .venv/local/bin/activate
```

```bash
# Source ros2 pkgs
source install/setup.bash

# Install requirements in venv
pip install setuptools
# pip install -r src/external-pip/vic_python_controllers/requirements.txt

# Install vic_python_controllers
pip install -e src/external-pip/vic_python_controllers

# install a ROS2 bags reader (nml_bag) used to plot experimental data (see notebooks)
pip install src/external-pip/nml_bag
```

# How to launch examples?

## Passive VIC

```bash
# Source venv and local ros2 setup
cd ~/dev/ros2_workspaces/ws_sdpf_controllers
source .venv/bin/activate
source install/setup.bash
```

```bash
# If using WSL2, bind the USB port
sudo usbip attach -r <host_ip> -b <bus_id>
```
__Note:__ see [this Gist](https://gist.github.com/tpoignonec/762a108b25a460eb98e0d05412f4da18) for details about the binding procedure.

import scipy
import numpy as np

"""Utils to extract data from the VIC and passivity filter ros2 msgs

How to use this module?
    1) Use nml_bag to transform the msg (MCAP only!!!) to a dict object

import nml_bag
reader = nml_bag.Reader(<path_to_bag>, topics=<list_of_topics_to_read>)
rosbag_data = reader.records

    2) Extract the data

vic_state_data = get_vic_state(rosbag_data, topic_name=<vic_state_topic_name>)

    3) use it :) See example "plot_exp_data.py" in this package.
"""

# Standard msg extraction utils

def unflatten_multiarray(flattened_multiarray, dims=None):
    """
    Unflatten a flattened 2D numpy array into a 2D numpy array given the dimensions.

    Args:
        flattened_multiarray (numpy.ndarray): The flattened 2D numpy array.
        dims (tuple, optional): The dimensions of the resulting 2D array. Defaults to None (assumes square!)

    Returns:
        numpy.ndarray: The unflattened 2D numpy array.
    """
    if (dims is None):
        rows = np.sqrt(flattened_multiarray.shape[0])
        assert (rows == int(rows))
        dims = (rows, rows)
    # TODO(tpoignonec): make sure the data is indeed row-major...
    return flattened_multiarray.reshape(dims[0], dims[1], order='C')

def extract_matrix(msg, dims=None):
    """
    Unflatten a flattened float multiarray ROS2 msg array into a 2D numpy array given the dimensions.

    Args:
        msg (dict): A FloatMultiArray message containing a 'data' key with a flattened 2D array.
        dims (tuple, optional): The dimensions of the resulting 2D array. Defaults to None (assumes square!)

    Returns:
        numpy.ndarray: The unflattened 2D numpy array.
    """
    return unflatten_multiarray(np.array(msg['data'], dims))

def extract_2D_matrix(msg):
    """ Specialization of extract_matrix() with dims = (2, 2) """
    return unflatten_multiarray(np.array(msg['data']), (2, 2))

def extract_3D_matrix(msg):
    """ Specialization of extract_matrix() with dims = (3, 3) """
    return unflatten_multiarray(np.array(msg['data']), (3, 3))

def extract_3D_matrix_from_6D(msg):
    """ Specialization of extract_matrix() with dims = (6, 6) . Return the upper left 3x3 block. """
    array_6x6 = unflatten_multiarray(np.array(msg['data']), (6, 6))
    return array_6x6[:3, :3]

def extract_pose_from_msg(msg):
    """
    Extract the position from a Pose ROS2 msg.

    Args:
        msg (dict): A Pose message containing a 'position' key with a 3D vector.

    Returns:
        numpy.ndarray: The position as a 3-element numpy array.
    """
    return np.array([
        msg['position']['x'],
        msg['position']['y'],
        msg['position']['z']
    ])

def extract_velocity_from_msg(msg):
    """
    Extract the velocity from a Twist ROS2 msg.

    Args:
        msg (dict): A Twist message containing a 'linear' key with a 3D vector.

    Returns:
        numpy.ndarray: The (linear) velocity as a 3-element numpy array.
    """
    return np.array([
        msg['linear']['x'],
        msg['linear']['y'],
        msg['linear']['z']
    ])

def extract_acc_from_msg(msg):
    # Using twist msg under the hood
    """
    Extract the acceleration from a Twist or Accel ROS2 msg.

    Args:
        msg (dict): A Twist / Accel message containing a 'linear' key with a 3D vector.

    Returns:
        numpy.ndarray: The (linear) acceleration as a 3-element numpy array.
    """
    return extract_velocity_from_msg(msg)

def extract_wrench_from_msg(msg):
    """
    Extract the wrench from a Wrench ROS2 msg.

    Args:
        msg (dict): A Wrench message containing a 'force' key with a 3D vector.

    Returns:
        numpy.ndarray: The force as a 3-element numpy array.
    """
    return np.array([
        msg['force']['x'],
        msg['force']['y'],
        msg['force']['z']
    ])

# Reader utils
def get_simulation_time(rosbag_raw_data, topic_name=None):
    data_dict = {
        'ros_time': [],
        'time': []
    }
    for msg_data in rosbag_raw_data:
        if (msg_data['topic'] == topic_name):
            # Retrieve ros timestamp
            data_dict['ros_time'] += [msg_data['time_ns']]
            data_dict['time'] += [msg_data['data']]

    data_dict['ros_time'] = np.array(data_dict['ros_time'])
    data_dict['time'] = np.array(data_dict['time'])
    return data_dict

def get_vic_state(rosbag_raw_data, topic_name=None):
    """
    Reads a rosbag and returns a dictionary containing the Vic controller state:
        - ros_time: The ROS timestamp of each message
        - position: The Cartesian position of the robot end-effector
        - velocity: The Cartesian velocity of the robot end-effector
        - acceleration: The Cartesian acceleration of the robot end-effector
        - wrench: The wrench (force and torque) exerted by the robot end-effector
        - natural_inertia: The natural inertia of the robot
        - desired_position: The desired Cartesian position of the robot end-effector
        - desired_velocity: The desired Cartesian velocity of the robot end-effector
        - desired_acceleration: The desired Cartesian acceleration of the robot end-effector
        - desired_wrench: The desired wrench (force and torque) exerted by the robot end-effector
        - stiffness: The rendered stiffness of the robot
        - damping: The rendered damping of the robot
        - inertia: The rendered inertia of the robot

    Note: message type is "cartesian_control_msgs::VicControllerState"

    Args:
        rosbag_raw_data (list): A list of dictionaries containing the ROS messages
        topic_name (str): The name of the topic to read from the rosbag

    Returns:
        dict: A dictionary containing the Vic controller state
    """
    data_dict = {
        'ros_time': [],
        'position': [],
        'velocity': [],
        'acceleration': [],
        'wrench': [],
        'natural_inertia': [],
        'desired_positon': [],
        'desired_velocity': [],
        'desired_acceleration': [],
        'desired_wrench': [],
        'stiffness': [],
        'damping': [],
        'inertia': []
    }
    for msg_data in rosbag_raw_data:
        if (msg_data['topic'] == topic_name):
            # Retrieve ros timestamp
            data_dict['ros_time'] += [msg_data['time_ns']]
            # Retrieve the robot state
            data_dict['position'] += [extract_pose_from_msg(msg_data['pose'])]
            data_dict['velocity'] += [extract_velocity_from_msg(msg_data['velocity'])]
            data_dict['acceleration'] += [extract_acc_from_msg(msg_data['acceleration'])]
            data_dict['wrench'] += [extract_wrench_from_msg(msg_data['wrench'])]
            data_dict['natural_inertia'] += [extract_3D_matrix_from_6D(msg_data['natural_inertia'])]
            # Retrieve the desired Cartesian trajectory
            data_dict['desired_position'] = extract_pose_from_msg(msg_data['desired_pose'])
            data_dict['desired_velocity'] = extract_velocity_from_msg(msg_data['desired_velocity'])
            data_dict['desired_acceleration'] = extract_acc_from_msg(msg_data['desired_acceleration'])
            data_dict['desired_wrench'] = extract_wrench_from_msg(msg_data['desired_wrench'])
            # Retrieve the rendered compliance
            data_dict['stiffness'] += [extract_3D_matrix_from_6D(msg_data['rendered_stiffness'])]
            data_dict['damping'] += [extract_3D_matrix_from_6D(msg_data['rendered_damping'])]
            data_dict['inertia'] += [extract_3D_matrix_from_6D(msg_data['rendered_inertia'])]

    data_dict['ros_time'] = np.array(data_dict['ros_time'])

    data_dict['position'] = np.array(data_dict['position'])
    data_dict['velocity'] = np.array(data_dict['velocity'])
    data_dict['acceleration'] = np.array(data_dict['acceleration'])
    data_dict['wrench'] = np.array(data_dict['wrench'])
    data_dict['natural_inertia'] = np.array(data_dict['natural_inertia'])

    data_dict['desired_position'] = np.array(data_dict['desired_position'])
    data_dict['desired_velocity'] = np.array(data_dict['desired_velocity'])
    data_dict['desired_acceleration'] = np.array(data_dict['desired_acceleration'])
    data_dict['desired_wrench'] = np.array(data_dict['desired_wrench'])

    data_dict['stiffness'] = np.array(data_dict['stiffness'])
    data_dict['damping'] = np.array(data_dict['damping'])
    data_dict['inertia'] = np.array(data_dict['inertia'])

    return data_dict

def get_reference_compliant_frame_trajectory(rosbag_raw_data, topic_name=None):
    """
    Reads a rosbag and returns a dictionary containing the desired Cartesian trajectory and the desired compliance sent to the VIC:
        - ros_time: The ROS timestamp of each message
        - desired_position: The desired Cartesian position of the robot end-effector
        - desired_velocity: The desired Cartesian velocity of the robot end-effector
        - desired_acceleration: The desired Cartesian acceleration of the robot end-effector
        - desired_wrench: The desired wrench (force and torque) exerted by the robot end-effector
        - desired_inertia: The desired inertia of the robot
        - desired_stiffness: The desired stiffness of the robot
        - desired_damping: The desired damping of the robot

    Note: message type is "cartesian_control_msgs::CompliantFrameTrajectory"

    Args:
        rosbag_raw_data (list): A list of dictionaries containing the ROS messages
        topic_name (str): The name of the topic to read from the rosbag

    Returns:
        dict: A dictionary containing the desired Cartesian trajectory and the desired compliance
    """
    data_dict = {
        'ros_time': [],
        'desired_position': [],
        'desired_velocity': [],
        'desired_acceleration': [],
        'desired_wrench': [],
        'desired_inertia': [],
        'desired_stiffness': [],
        'desired_damping': [],
    }
    for msg_data in rosbag_raw_data:
        if (msg_data['topic'] == topic_name):
            # Retrieve ros timestamp
            data_dict['ros_time'] += [msg_data['time_ns']]

            # Retrieve the desired Cartesian trajectory
            cartesian_trajectory_point = msg_data['cartesian_trajectory_points'][0]
            data_dict['desired_position'] += [extract_pose_from_msg(cartesian_trajectory_point['pose'])]
            data_dict['desired_velocity'] += [extract_velocity_from_msg(cartesian_trajectory_point['velocity'])]
            data_dict['desired_acceleration'] += [extract_acc_from_msg(cartesian_trajectory_point['acceleration'])]
            data_dict['desired_wrench'] += [extract_wrench_from_msg(cartesian_trajectory_point['wrench'])]

            # Retrieve the desired compliance
            compliance_at_point = msg_data['compliance_at_points'][0]
            data_dict['desired_inertia'] += [extract_3D_matrix_from_6D(compliance_at_point['inertia'])]
            data_dict['desired_stiffness'] += [extract_3D_matrix_from_6D(compliance_at_point['stiffness'])]
            data_dict['desired_damping'] += [extract_3D_matrix_from_6D(compliance_at_point['damping'])]

    data_dict['ros_time'] = np.array(data_dict['ros_time'])

    data_dict['desired_position'] = np.array(data_dict['desired_position'])
    data_dict['desired_velocity'] = np.array(data_dict['desired_velocity'])
    data_dict['desired_acceleration'] = np.array(data_dict['desired_acceleration'])
    data_dict['desired_wrench'] = np.array(data_dict['desired_wrench'])

    data_dict['desired_inertia'] = np.array(data_dict['desired_inertia'])
    data_dict['desired_stiffness'] = np.array(data_dict['desired_stiffness'])
    data_dict['desired_damping'] = np.array(data_dict['desired_damping'])

    return data_dict


def get_compliant_frame(rosbag_raw_data, topic_name=None):
    """
    Reads a rosbag and returns a dictionary containing the desired compliance:
        - ros_time: The ROS timestamp of each message
        - inertia: The desired inertia of the robot
        - stiffness: The desired stiffness of the robot
        - damping: The desired damping of the robot

    Note: message type is "cartesian_control_msgs::CartesianCompliant"

    Args:
        rosbag_raw_data (list): A list of dictionaries containing the ROS messages
        topic_name (str): The name of the topic to read from the rosbag

    Returns:
        dict: A dictionary containing the desired compliance
    """
    data_dict = {
        'ros_time': [],
        'inertia': [],
        'stiffness': [],
        'damping': [],
    }
    for msg_data in rosbag_raw_data:
        if (msg_data['topic'] == topic_name):
            # Retrieve ros timestamp
            data_dict['ros_time'] += [msg_data['time_ns']]
            # Retrieve the desired compliance
            data_dict['inertia'] += [extract_3D_matrix(msg_data['inertia'])]
            data_dict['stiffness'] += [extract_3D_matrix(msg_data['stiffness'])]
            data_dict['damping'] += [extract_3D_matrix(msg_data['damping'])]

    data_dict['ros_time'] = np.array(data_dict['ros_time'])

    data_dict['inertia'] = np.array(data_dict['inertia'])
    data_dict['stiffness'] = np.array(data_dict['stiffness'])
    data_dict['damping'] = np.array(data_dict['damping'])

    return data_dict


def get_diagnostic_data(rosbag_raw_data, topic_name=None):
    """
    Reads a rosbag and returns a dictionary containing the data from a diagnostic topic:
        - ros_time: The ROS timestamp of each message
        - <key>: The value of the diagnostic data with the corresponding key

    Note: message type is "cartesian_control_msgs::KeyValues"

    Args:
        rosbag_raw_data (list): A list of dictionaries containing the ROS messages
        topic_name (str): The name of the topic to read from the rosbag

    Returns:
        dict: A dictionary containing the data from the diagnostic topic
    """
    data_dict = {
        'ros_time': [],
    }
    is_first_itr = True
    for msg_data in rosbag_raw_data:
        if (msg_data['topic'] == topic_name):
            # Retrieve ros timestamp
            data_dict['ros_time'] += [msg_data['time_ns']]
            # Retrieve the actual data
            if is_first_itr:
                for value_, key_ in zip(msg_data['values'], msg_data['keys']):
                    data_dict[key_] = [value_]
                is_first_itr = False
            else:
                for value_, key_ in zip(msg_data['values'], msg_data['keys']):
                    data_dict[key_] += [value_]
    # Consolidate data as numpy arrays
    for key_, data_list_ in data_dict.items():
        data_dict[key_] = np.array(data_list_)

    return data_dict

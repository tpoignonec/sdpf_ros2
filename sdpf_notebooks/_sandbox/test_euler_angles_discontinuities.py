import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

radius = 0.1
height = 0.3


orientation_direction = np.linspace(-2 * np.pi, 2 * np.pi, 50)
angulation_propre_values = np.linspace(- np.pi / 2, np.pi / 2, 50)
# Ensure the angulation_propre_values are symmetric around zero
# by concatenating the values with their negative counterparts
# This creates a full range from -pi/2 to pi/2 and back to -pi/2
angulation_propre_values = np.concatenate(
    [angulation_propre_values, angulation_propre_values[::-1]])

orientations = []

for orientation in orientation_direction:
    for angulation in angulation_propre_values:
        new_orientation = R.from_rotvec(np.array([0, 0, angulation]))
        # Apply the actual orientation to the axis
        axis_rotation = R.from_rotvec(
            np.array([np.cos(orientation), np.sin(orientation), 0]))
        orientations.append(axis_rotation * new_orientation)

# Test the discontinuities in Euler angles
import itertools
repr_choices = [''.join(p) for p in itertools.permutations(['x', 'y', 'z'])]
repr_choices += \
    [''.join(p) for p in itertools.permutations(['X', 'Y', 'Z'])]
print("Representations of Euler angles:", repr_choices)

threshold = 180  # Define a threshold for discontinuity detection
repr_is_discontinuous = []
for repr_choice in repr_choices:
    repr_euler_angles = [
        orientation.as_euler(repr_choice, degrees=True)
        for orientation in orientations
    ]
    # Check for discontinuities
    is_discontinuous = False
    for i in range(1, len(repr_euler_angles)):
        if np.any(np.abs(repr_euler_angles[i] - repr_euler_angles[i - 1]) > threshold):
            is_discontinuous = True
    repr_is_discontinuous.append(is_discontinuous)

# Print the results
# Separate continuous and discontinuous representations
continuous_representations = [repr_choices[i] for i in range(len(repr_choices)) if not repr_is_discontinuous[i]]
discontinuous_representations = [repr_choices[i] for i in range(len(repr_choices)) if repr_is_discontinuous[i]]

print("Continuous representations:", continuous_representations)
print("DISCONTINUOUS representations:", discontinuous_representations)


# Plot the Euler angles for ONE representation
repr_choice = 'xyz'  # Representation of the Euler angles
repr_euler_angles = [
    orientation.as_euler(repr_choice, degrees=True) for orientation in orientations
]

fig_euler = plt.figure()
plt.title(f'Euler angles representation: {repr_choice}')
plt.plot(np.asarray(repr_euler_angles))


# Plot each frame of reference
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, orientation in enumerate(orientations):
    # Convert orientation to rotation matrix
    rotation_matrix = orientation.as_matrix()
    axis_length = 0.1  # Length of the axes
    # Compute the axes of the frame
    x_axis = 0.8 * axis_length * rotation_matrix[:, 0]  # x-axis
    y_axis = 0.8 * axis_length * rotation_matrix[:, 1]  # y-axis
    z_axis = axis_length * rotation_matrix[:, 2]  # z-axis

    # Plot the axes
    ax.plot([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]], color='red', label='X-axis' if i == 0 else "")
    ax.plot([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]], color='green', label='Y-axis' if i == 0 else "")
    ax.plot([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]], color='blue', label='Z-axis' if i == 0 else "")

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.signal import find_peaks

def ensure_dir_exists(dir_name):  # noqa:D103
    if not os.path.exists(dir_name):
        # Create a new directory because it does not exist
        os.makedirs(dir_name)
        print(f"The directory {dir_name} was created!")


def get_color_list():  # noqa:D103
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def set_linestyle_list(linestyle_list=['-', '-', '-', '-', '-.']):
    """Set the global matplotlib linestyle cycle to a custom list."""
    colors = get_color_list()
    linestyles = linestyle_list + ['-'] * (len(colors) - len(linestyle_list))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        linestyle=linestyles,
        color=colors
    )
    print(f"Set matplotlib linestyle cycle to: {linestyle_list}")


def get_linestyle_list():  # noqa:D103
    """Get the current matplotlib linestyle cycle."""
    return plt.rcParams['axes.prop_cycle'].by_key()['linestyle']



def flip(items, ncol):  # noqa:D103
    # https://stackoverflow.com/questions/10101141/matplotlib-legend-add-items-across-columns-instead-of-down
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def auto_adjust_ylim_for_xlim(ax, y_margin=0.05):
    """
    Automatically adjust y-limits based on current x-limits
    """
    xlim = ax.get_xlim()

    # Get all line data
    all_y_data = []
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # Filter data within current x-limits
        mask = (x_data >= xlim[0]) & (x_data <= xlim[1])
        if np.any(mask):
            visible_y = y_data[mask]
            all_y_data.extend(visible_y)

    if all_y_data:
        y_min, y_max = np.min(all_y_data), np.max(all_y_data)
        y_range = y_max - y_min
        margin = y_range * y_margin if y_range > 0 else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)


def highlight_regions(simulation_data, ax):  # noqa:D103
    alpha = 0.1
    # ax.axvspan(0.0, simulation_data['t2'], color='red', alpha=alpha, lw=0)
    ax.axvspan(simulation_data['t1'], simulation_data['t2'], color='green', alpha=alpha, lw=0)
    # ax.axvspan(simulation_data['t2'], np.max(simulation_data['time']), color='red', alpha=alpha, lw=0)

def annotate(ax, text, coord, bgc='white', extra_text_kwargs={}):  # noqa:D103
    t = ax.text(
        *coord,
        text,
        transform=ax.transAxes,
        **extra_text_kwargs
    )
    t.set_bbox(dict(facecolor=bgc, alpha=0.5, edgecolor='none'))
    return t


def annotate_regions(simulation_data, ax, relative_height):  # noqa:D103
    """Annotate a plot with (a), (b), (c) at relative_height.

    Note: relative_heights in ax's axes fraction coordinates.
    """
    annotate(ax, r'(a)', (1/3/2.25, relative_height))
    annotate(ax, r'(b)', (1/3 + 1/3/2.25, relative_height), bgc='none')
    annotate(ax, r'(c)', (2/3 + 1/3/2.25, relative_height))


def annotate_single_peak(
        ax, x, y, y_limit,
        offset_text, alignment, is_min, color
):
    """
    Annotate a single peak or valley on the plot.

    Parameters:
    - ax: The axis to annotate.
    - x: The x-coordinate of the peak/valley.
    - y: The y-coordinate of the peak/valley.
    - y_limit: The limit for annotation placement (e.g., y_min or y_max).
    - offset_text: Offset for text annotation.
    - alignment: Text alignment ('left' or 'right').
    - is_min: Whether the annotation is for a minimum ('Min') or maximum ('Max').
    - color: Color of the annotation text.
    """
    args_font = {
        'fontsize': 3,
        'color': color,
        'ha': alignment,
        'textcoords': "offset points",
        # 'bbox': dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white")
    }
    is_left = (alignment == 'left')
    format_text = \
        f"{'*' if is_left else ''}{y:.2f}{'*' if not is_left else ''}"
    ax.annotate(
        format_text,
        (x, y_limit),
        xytext=(-0.1 if is_left else -0.2, -offset_text if is_min else offset_text - 2),
        **args_font
    )


def annotate_peaks(
        ax, x_data, y_data,
        color='black',
        annotate_min=True,
        annotate_max=True,
        proximity_threshold=1.2):
    """
    Annotate peaks (min/max values) on a plot, dynamically adjusting alignment
    to avoid overlap if peaks are close.

    Parameters:
    - ax: The axis to annotate.
    - x_data: The x-axis data.
    - y_data: The y-axis data.
    - annotate_min: Whether to annotate minimum values.
    - annotate_max: Whether to annotate maximum values.
    - proximity_threshold: Minimum distance between peaks to avoid overlap.
    """
    y_min, y_max = ax.get_ylim()  # Get current axis limits
    offset_text = 3.0  # Offset for text annotation

    def find_peaks_in_range(data, threshold):
        """Find peaks in data that are above a certain threshold."""
        peaks, _ = find_peaks(data)
        return [peak for peak in peaks if data[peak] > threshold]

    if annotate_max:
        # Detect peaks (local maxima)
        peaks = find_peaks_in_range(y_data, y_max)
        for i, peak in enumerate(peaks):
            y_peak = y_data[peak]
            alignment = 'left'  # Default alignment
            if i < len(peaks) - 1 and abs(x_data[peak] - x_data[peaks[i + 1]]) < proximity_threshold:  # noqa:E501
                print("Proximity alert for max peack annotation!")
                alignment = 'right' if alignment == 'left' else 'left'
            if i == len(peaks):
                alignment = 'left'
            annotate_single_peak(
                ax, x_data[peak], y_peak,
                y_max, offset_text, alignment, False, color)

    if annotate_min:
        # Detect valleys (local minima)
        valleys = find_peaks_in_range(-y_data, -y_min)
        for i, valley in enumerate(valleys):
            y_valley = y_data[valley]
            alignment = 'left'  # Default alignment
            if i < len(valleys) - 1 and abs(x_data[valley] - x_data[valleys[i + 1]]) < proximity_threshold:  # noqa:E501
                print("Proximity alert for min valley annotation!")
                alignment = 'right' if alignment == 'left' else 'left'
            if i == len(valleys):
                alignment = 'left'
            annotate_single_peak(
                ax, x_data[valley], y_valley,
                y_min, offset_text, alignment, True, color)

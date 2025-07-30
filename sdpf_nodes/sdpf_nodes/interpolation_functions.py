"""Interpolation functions and utilities for smooth and step-like periodic signals.

This module provides sinusoidal, step, and smooth alternating step functions,
as well as tools for computing analytical and numerical derivatives, and
visualizing their behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def fct_sinus(
        time_point: float,
        period: float,
        delay: float = 0,
        derivative: int = 0):
    """Create a sinusoidal function."""
    omega = 2 * np.pi / period  # Convert to angular frequency
    if derivative == 0:
        return (1 + np.sin(omega*(time_point + delay)))/2
    elif derivative == 1:
        return omega*np.cos(omega*(time_point + delay))/2
    else:
        raise ValueError("Invalid derivative order")

def fct_cosinus(
        time_point: float,
        period: float,
        delay: float = 0,
        derivative: int = 0):
    """Create a cosinusoidal function that starts at 0."""
    omega = 2 * np.pi / period  # Convert to angular frequency
    if derivative == 0:
        return (1 - np.cos(omega*(time_point + delay)))/2
    elif derivative == 1:
        return omega*np.sin(omega*(time_point + delay))/2
    else:
        raise ValueError("Invalid derivative order")


def fct_step(
        time_point: float,
        period: float,
        delay: float = 0,
        derivative: int = 0):
    """Create a step function. NOT differentiable."""
    if derivative == 0:
        return np.where(np.mod(time_point + delay, period) >= period / 2, 1.0, 0.0)
    else:
        return np.nan * np.zeros_like(time_point)


def fct_tanh_alternating(
        time_point: float,
        period: float,
        delay: float = 0,
        derivative: int = 0):
    """Create a smooth alternating step function.

    Creates a smooth alternating step function that stays at 0 for half the period
    and 1 for the other half, with differentiable transitions.

    Args:
        time_point: Time at which to evaluate the function
        period: Period of the complete 0->1->0 cycle
        derivative: If 0, return the function value; if 1, return its derivative

    Returns:
        Function value or its derivative at the given time point
    """
    # Steepness controls how sharp the transition is (higher = closer to a step function)
    steepness = 200.0

    # Calculate normalized position within period (0 to 1)
    normalized_time = ((time_point + delay) % period) / period

    # Create a square wave with smooth transitions
    # Use tanh centered at 0.1 and 0.6 of the period
    if derivative == 0:
        # Transition from 0→1 at 0.1 of period, and from 1→0 at 0.6
        up_transition = \
            0.5 * (1 + np.tanh(steepness * (normalized_time - 0.1)))
        down_transition = \
            0.5 * (1 - np.tanh(steepness * (normalized_time - 0.6)))

        # Combine the transitions to get the alternating step pattern
        return np.minimum(up_transition, down_transition)
    elif derivative == 1:
        # Derivative of up transition (0→1)
        dup_dt = 0.5 * steepness * (1 / period) \
            * (1 / np.cosh(steepness * (normalized_time - 0.1))**2)

        # Derivative of down transition (1→0)
        ddown_dt = -0.5 * steepness * (1 / period) \
            * (1 / np.cosh(steepness * (normalized_time - 0.6))**2)

        # Only one transition is active at any time
        return np.where(normalized_time < 0.5, dup_dt, ddown_dt)
    else:
        raise ValueError('Invalid derivative order')


def compute_numerical_derivative(func, time_points, period, h=1e-6):
    """
    Compute the numerical derivative of a function using central differences.

    Args:
        func: Function to differentiate (taking time_point and period)
        time_points: Time points to evaluate derivative at
        period: Period of the function
        h: Step size for finite difference

    Returns:
        Numerical derivative at the given time points
    """
    derivatives = []

    for t in time_points:
        # Central difference approximation: (f(t+h) - f(t-h)) / (2*h)
        f_plus = func(t + h, period)
        f_minus = func(t - h, period)
        numerical_derivative = (f_plus - f_minus) / (2 * h)
        derivatives.append(numerical_derivative)

    return np.array(derivatives)


if __name__ == '__main__':
    # Test parameters
    period = 4.0  # seconds
    t_max = 3 * period  # Show 3 complete periods
    dt = 1e-3  # Time step for plotting

    # Create time points
    t = np.arange(0, t_max, dt)

    # Calculate function values
    function_values = fct_tanh_alternating(t, period, derivative=0)

    # Calculate analytical derivative
    analytical_derivative = fct_tanh_alternating(t, period, derivative=1)

    # Calculate numerical derivative
    numerical_derivative = compute_numerical_derivative(
        lambda t, p: fct_tanh_alternating(t, p, derivative=0),
        t, period
    )

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1])

    # Plot function values
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, function_values, 'b-', linewidth=2)
    ax1.set_title('Alternating Step Function (tanh-based)', fontsize=14)
    ax1.set_ylabel('Function Value')
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)

    # Add period markers
    for i in range(1, int(t_max / period) + 1):
        ax1.axvline(x=i*period, color='gray', linestyle='--', alpha=0.7)

    # Plot derivatives
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, analytical_derivative, 'r-', linewidth=2, label='Analytical Derivative')
    ax2.plot(t, numerical_derivative, 'g--', linewidth=1.5, label='Numerical Derivative')
    ax2.set_title('Derivative Comparison', fontsize=14)
    ax2.set_ylabel('Derivative Value')
    ax2.legend()
    ax2.grid(True)

    # Plot difference between analytical and numerical derivatives
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    diff = analytical_derivative - numerical_derivative
    ax3.plot(t, diff, 'k-', linewidth=1.5)
    ax3.set_title('Error: Analytical - Numerical', fontsize=14)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error')
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Print some statistics about the difference
    print(f"Maximum absolute error: {np.max(np.abs(diff)):.8e}")
    print(f"Mean absolute error: {np.mean(np.abs(diff)):.8e}")
    print(f"Root Mean Square Error: {np.sqrt(np.mean(diff**2)):.8e}")

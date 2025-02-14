import warnings

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, cross_val_score

from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer, mean_squared_error


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from pyDOE3 import lhs

import os
import multiprocessing as mp
import random
import time
import sys
import win32com.client as win32  # For interacting with Aspen Plus
import pythoncom  # Required for COM operations
from contextlib import redirect_stdout, redirect_stderr

np.set_printoptions(linewidth=120, suppress=True, formatter={'float_kind': '{:.3f}'.format})



def super_print(Fθ_iter, radii_iter, ite=None):
    """
    Prints metrics for the given iteration based on the Fθ_iter and radii_iter.

    :param Fθ_iter: List of (f, θ) pairs for each iteration.
    :param radii_iter: List of [Δ_, σ_] arrays for each iteration.
    :param ite: Current iteration index.
    """
    print('_' * 50)
    # Title
    if ite is not None:
        print(f'           METRICS AT ITERATION {ite + 1}            ')
    else:
        print('           METRICS            ')

    # Retrieve metrics for the current iteration
    if ite is not None:
        f, θ = Fθ_iter[ite]
        Δ_, σ_ = radii_iter[ite]
    else:
        f, θ = None, None
        Δ_, σ_ = None, None

    # Conditional prints for each metric
    if f is not None:
        print(f"Optimality (f): ${-f:,.2f}")
    if θ is not None:
        print(f"Infeasibility (θ): {θ:.2f}")
    if Δ_ is not None:
        print(f"Trust Region Radius (Δ_): {Δ_}")
    if σ_ is not None:
        print(f"Sampling Region Radius (σ_): {σ_}")

    # Footer
    print('_' * 50)




def obj_function(xk, rw):  # Penalties added for infeasibilities.
    if len(xk) != 9 or len(rw) != 9:
        raise ValueError(
            f"wk and rw should both contain 9 elements, but they contain {len(xk)} and {len(rw)} elements, respectively.")

    # Extract values from the rigorous model output (wk) and surrogate model output (rw)
    T, P, H2, CO, CO2, H2O, MEOH, ETOH, DME = xk
    TOUT, POUT, H2OUT, COOUT, CO2OUT, H2OOUT, MEOHOUT, ETOHOUT, DMEOUT = rw

    # Calculate the revenue based on methanol and ethanol production (add others as necessary)
    revenue = 50 * MEOHOUT + 8 * ETOHOUT  # Added ethanol revenue term


    raw_material_cost = 3 * (H2 + CO)


    # Raw material costs from revenue
    profit_before_penalties = revenue - raw_material_cost

    # Add penalties for unreacted outputs and environmental impact
    penalties = 10 * DMEOUT + 7 * CO2OUT  # + (sum(max(0, x - high) + max(0, low - x) for x, (low, high) in zip(xk, bounds)))

    operational_penalty = 10 * (POUT - P) + 0.1 * (TOUT - T)

    # Final objective function value (profit)
    profit = profit_before_penalties - penalties - operational_penalty

    return -profit




def Infeasibility(rw, dw):
    """
    Compute infeasibility (θ) using only black-box decision variables.

    :param rw: Surrogate model predictions (array-like or scalar).
    :param dw: Rigorous model predictions (array-like or scalar).
    :return: Feasibility metric (θ) as a scalar.
    """
    # Ensure rw and dw are NumPy arrays
    rw = np.array(rw, dtype=np.float64).ravel()  # Ensure rw is a 1D array
    dw = np.array(dw, dtype=np.float64).ravel()  # Ensure rw is a 1D array

    # Check for shape compatibility
    if rw.shape != dw.shape:
        raise ValueError(f"Shape mismatch: rw shape {rw.shape} and dw shape {dw.shape}")

    # Logarithmic normalization for black-box variables
    avg_predictions = (rw + dw) / 2 + 1e-6  # Avoid division by zero
    scale = 10 ** np.floor(np.log10(avg_predictions))

    # Compute L1-norm for black-box variables
    differences = np.abs(rw - dw)  # Element-wise differences
    normalized_differences = differences / scale  # Normalize differences
    theta = np.mean(normalized_differences)  # Average L1 norm

    return theta  # Return θ as a scalar




def fin_diff(func, xk, *args, epsilon=1e-3):
    """Compute numerical gradient of `func` at `xk` when `func` returns an array."""
    # Evaluate the base function at xk to determine the shape of the output
    f_base = func(xk, *args) if args else func(xk)

    # Handle scalar outputs of `func` (e.g., single profit value) or array outputs
    if np.isscalar(f_base):
        GRAD = np.zeros(len(xk))  # Scalar: Gradient w.r.t. each input
        for i in range(len(xk)):
            xk_plus = xk.copy()
            xk_minus = xk.copy()
            xk_plus[i] += epsilon
            xk_minus[i] -= epsilon

            f_plus = func(xk_plus, *args)
            f_minus = func(xk_minus, *args)

            GRAD[i] = (f_plus - f_minus) / (2 * epsilon)
    else:
        GRAD = np.zeros((len(f_base), len(xk)))  # Array: Gradient for each output dimension w.r.t. each input
        for i in range(len(xk)):
            xk_plus = xk.copy()
            xk_minus = xk.copy()
            xk_plus[i] += epsilon
            xk_minus[i] -= epsilon

            f_plus = func(xk_plus, *args)
            f_minus = func(xk_minus, *args)

            GRAD[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return GRAD


def process_and_plot_results(Fθ_iter, radii_iter, Chi_iter, time, surrogate_model,
                             rigorous_model, rigorous_placeholder, output_path="TRF_results.csv", save=False):
    """
    Processes TRF results, exports them to a CSV, and generates subplots for objective function, infeasibility, and radii.

    :param Fθ_iter: List of (f, θ) tuples for each iteration.
    :param radii_iter: List of [Δ_, σ_] arrays for each iteration.
    :param Chi_iter: List of χ values for each iteration.
    :param time: Total time taken for the optimization process.
    :param surrogate_model: Surrogate model description (string).
    :param rigorous_model: Rigorous model description (string).
    :param rigorous_placeholder: Rigorous placeholder description (string).
    :param output_path: Path to save the CSV file (default: "TRF_results.csv").
    """

    # Dynamically retrieve model names
    surrogate_model_name = surrogate_model.__class__.__name__ if surrogate_model else "None"
    rigorous_model_name = rigorous_model.__class__.__name__ if rigorous_model else "None"
    rigorous_placeholder_name = rigorous_placeholder.__class__.__name__ if rigorous_placeholder else "None"

    # Ensure all arrays have consistent lengths
    while len(Chi_iter) < len(Fθ_iter):
        Chi_iter.append(None)  # Fill missing values with None
    while len(radii_iter) < len(Fθ_iter):
        radii_iter.append([None, None])  # Fill missing radii
    while len(time) < len(Fθ_iter):
        time.append(None)  # Fill missing values with None

    # Prepare the data for the DataFrame
    data = {
        "Iteration": list(range(1, len(Fθ_iter) + 1)),  # Iteration numbers
        "Objective Function (f)": [-f for f, _ in Fθ_iter],  # Negated for maximization
        "Infeasibility (θ)": [θ for _, θ in Fθ_iter],
        "Trust Region Radius (Δ_)": [r[0] for r in radii_iter],
        "Sampling Region Radius (σ_)": [r[1] for r in radii_iter],
        "Chi (χ)": Chi_iter,
        "Time (s)": time,  # Repeat total time for all iterations
        "Surrogate Model": [surrogate_model_name] * len(Fθ_iter),
        "Rigorous Model": [rigorous_model_name] * len(Fθ_iter),
        "Rigorous Placeholder": [rigorous_placeholder_name] * len(Fθ_iter),
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    if save:
        # Export DataFrame to CSV
        df.to_csv(output_path, index=False)
        print(f"Data exported successfully to {output_path}")

    # # Optional: Display the DataFrame
    # print(df)

    # Extract values for plotting
    iterations = np.arange(1, len(Fθ_iter) + 1)  # Iteration indices
    f_values = [-f for f, _ in Fθ_iter]  # Objective function (negated for maximization)
    theta_values = [θ for _, θ in Fθ_iter]  # Infeasibility (θ)
    delta_values = [radii[0] for radii in radii_iter]  # Trust region radius (Δ_)
    sigma_values = [radii[1] for radii in radii_iter]  # Sampling region radius (σ_)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=True)

    # Plot f (Objective Function)
    axes[0].plot(iterations, f_values, marker='o', label='Objective Function (f)', color='blue')
    axes[0].set_ylabel('Profit ($)')
    axes[0].set_title('Objective Function (f) vs Iterations')
    axes[0].legend()
    axes[0].grid(True)

    # Plot θ (Infeasibility)
    axes[1].plot(iterations, theta_values, marker='o', label='Infeasibility (θ)', color='green')
    axes[1].set_ylabel('Infeasibility (θ)')
    axes[1].set_title('Infeasibility (θ) vs Iterations')
    axes[1].legend()
    axes[1].grid(True)

    # Plot radii (Δ_ and σ_)
    axes[2].plot(iterations, delta_values, marker='o', label='Trust Region Radius (Δ_)', color='red')
    axes[2].plot(iterations, sigma_values, marker='o', label='Sampling Region Radius (σ_)', color='orange')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Radii')
    axes[2].set_title('Radii (Δ_ and σ_) vs Iterations')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

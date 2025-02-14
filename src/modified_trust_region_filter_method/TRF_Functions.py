import warnings
import time

from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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




class Min:
    @staticmethod
    def infeasibility(item):
        """
        Find the minimum value from the second values of each list in a list of value pairs.

        :param item: List of lists, where each inner list contains two values [f, q].
        :return: The minimum value among the second values (q).
        """
        # Extract the second values and find the minimum
        second_values = [pair[1] for pair in item]
        return min(second_values)

    @staticmethod
    def trfradius(item):
        """
        Find the array with the smallest sum of elements from the second arrays in a list of array pairs.

        :param item: List of lists, where each inner list contains two arrays [array1, array2].
        :return: The second array with the minimum sum of elements.
        """
        # Extract the first arrays and find the one with the smallest sum
        second_arrays = [pair[0] for pair in item]
        min_sum_idx = np.argmin([np.sum(arr) for arr in second_arrays])
        return second_arrays[min_sum_idx]


class KfullyLinearization:
    def __init__(self, surrogate_model, rigorous_model, κf, κg, max_expansion=50):
        self.surrogate_model = surrogate_model
        self.rigorous_model = rigorous_model
        self.κf = κf
        self.κg = κg
        self.max_expansion = max_expansion

    @staticmethod
    def fin_diff_kproperty(func, xk, epsilon=1e-5):
        """
        Compute numerical gradient of `func` at `xk` when `func` returns an array or scalar.
        """
        f_base = func(xk)
        is_scalar = np.isscalar(f_base)

        if is_scalar:
            f_base = np.array([f_base])

        GRAD = np.zeros((len(f_base), len(xk)))

        for i in range(len(xk)):
            xk_plus = xk.copy()
            xk_minus = xk.copy()
            xk_plus[i] += epsilon
            xk_minus[i] -= epsilon

            f_plus = func(xk_plus)
            f_minus = func(xk_minus)

            if is_scalar:
                f_plus = [f_plus]
                f_minus = [f_minus]

            GRAD[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * epsilon)

        return GRAD if not is_scalar else GRAD.flatten()

    def is_κ_fully_linear(self, xk_, Δk, blackbox_idx):
        """
        Check if the surrogate model is κ-fully linear in the sampling region (only for black-box variables).
        """
        # Predict surrogate and rigorous outputs
        rw_pred = self.surrogate_model.predict(xk_)
        dw_pred = self.rigorous_model.predict(xk_)

        # Check and filter dimensions
        if len(rw_pred) != len(blackbox_idx):
            rw_pred = rw_pred[blackbox_idx]

        if len(dw_pred) != len(blackbox_idx):
            dw_pred = dw_pred[blackbox_idx]

        # # Debugging output
        # print(f"Filtered rw_pred: {rw_pred}")
        # print(f"Filtered dw_pred: {dw_pred}")

        # Compute gradients
        rw_GRAD = self.fin_diff_kproperty(self.surrogate_model.predict, xk_)
        dw_GRAD = self.fin_diff_kproperty(self.rigorous_model.predict, xk_)

        # Check and filter gradients
        if rw_GRAD.shape[1] != len(blackbox_idx):
            rw_GRAD = rw_GRAD[:, blackbox_idx]

        if dw_GRAD.shape[1] != len(blackbox_idx):
            dw_GRAD = dw_GRAD[blackbox_idx, blackbox_idx]

        # # Debugging output
        # print(f"Filtered rw_GRAD: {rw_GRAD}")
        # print(f"Filtered dw_GRAD: {dw_GRAD}")

        # Restrict Δk to black-box variables
        if len(Δk) != len(blackbox_idx):
            Δk_blackbox = Δk[blackbox_idx]
        else:
            Δk_blackbox = Δk

        # κ-fully linearity conditions
        cond1 = np.all(np.abs(rw_pred - dw_pred) <= self.κf * Δk_blackbox ** 2)
        cond2 = np.all(np.linalg.norm(rw_GRAD - dw_GRAD, axis=1) <= self.κg * Δk_blackbox)

        return cond1 and cond2

    # MinFeasRadiiFinder: Simpler and faster for homogeneous systems.
    def MinFeasRadiiFinder(self, xk_, Δk, blackbox_idx, max_iter=20, tolerance=1e-6, verbose=False):
        """
        Adjust sampling region radii (σ_) for κ-fully linearity with safeguards to prevent over-shrinking.

        :param xk_: Current iterate for decision variables.
        :param Δk: Trust region radius for blackbox variables (already filtered).
        :param blackbox_idx: Indices of blackbox variables (informational only).
        :param max_iter: Maximum iterations for bisection.
        :param tolerance: Tolerance for convergence.
        :param verbose: If True, prints debugging information.
        :return: Final σ_ (sampling region radii) as a list and adjusted Δk.
        """
        # Initialize bounds for blackbox variables
        lower_bound = np.zeros_like(Δk)
        upper_bound = Δk.copy()
        σ_ = Δk.copy()
        min_radius = 1e-3  # Minimum allowable radius

        for iteration in range(max_iter):
            # Check κ-full linearity
            is_fully_linear = self.is_κ_fully_linear(xk_, σ_, blackbox_idx)

            if is_fully_linear:
                upper_bound = σ_.copy()
            else:
                lower_bound = σ_.copy()

            # Update σ_ as the midpoint of lower and upper bounds
            σ_ = (lower_bound + upper_bound) / 2

            # Apply safeguards
            σ_ = np.clip(σ_, min_radius, Δk)

            # if verbose:
            #     print(f"Iteration {iteration + 1}: σ_ = {σ_}, lower_bound = {lower_bound}, upper_bound = {upper_bound}")

            # Check convergence
            if np.linalg.norm(upper_bound - lower_bound) < tolerance:
                break

        # Final safeguard to ensure radii are reasonable
        σ_ = np.clip(σ_, min_radius, Δk)

        # if verbose:
        #     print(f"Final σ_: {σ_}")

        return σ_, Δk




def compatibility_check(xk, yk, surrogate_model, blackbox_idx, Δ, κΔ, κμ, μ):
    """
    Perform compatibility check for the current trust region subproblem (TRSP).
    This function attempts to find a correction term `nk` such that compatibility is achieved.

    :param xk: Current iterate for the input variables.
    :param yk: Current output from the rigorous model.
    :param surrogate_model: The surrogate model to use for prediction.
    :param Δ: Current trust region size.
    :param κΔ, κμ, μ: Parameters to define the adjustment factor and constraints.
    :return: Updated xk (if compatible), radii update, compatibility result (β), success flag.
    """
    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]

    # Compute the trust region bounds for blackbox variables
    adjustment_factor = κΔ * Δ * np.minimum(1, κμ * Δ ** μ)
    bounds = [(xk[idx] - adjustment_factor[i], xk[idx] + adjustment_factor[i]) for i, idx in enumerate(blackbox_idx)]

    # Add dummy bounds for graybox variables (no restriction)
    bounds_full = [(0, 1e4)] * len(xk)
    for i, idx in enumerate(blackbox_idx):
        bounds_full[idx] = bounds[i]  # Overwrite bounds for blackbox variables

    # Define the objective function
    def compatibility_obj(nk):
        # Split nk into blackbox and graybox components
        xk_adjusted = xk + nk  # Adjust xk by the perturbation
        rk_adjusted = surrogate_model.predict(xk_adjusted)
        return np.linalg.norm(yk - rk_adjusted)  # Minimize the residual norm

    # Constraints for non-negativity and upper bounds
    constraints = [
        {'type': 'ineq', 'fun': lambda nk: xk + nk},  # x + nk >= 0
        {'type': 'ineq', 'fun': lambda nk: 1e3 - (xk + nk)},  # x + nk <= 1e3
    ]

    # Initial guess for nk
    nk_initial = np.zeros(len(xk))

    # Run the optimization
    result = minimize(
        compatibility_obj,
        nk_initial,
        bounds=bounds_full,
        constraints=constraints,
        method='trust-constr'
    )

    if result.success:
        nk = result.x
        xk_minimizer = xk + nk
        Δ_comp = np.zeros_like(xk)  # Initialize Δ_comp for all variables

        # Apply Δ only to blackbox variables
        if len(Δ) != len(blackbox_idx):
            Δ_comp[blackbox_idx] = Δ[blackbox_idx]
        else:
            Δ_comp[blackbox_idx] = Δ
        # print("Compatibility solver is successful.")
        β = result.fun  # Compatibility result from optimization
        return xk_minimizer, Δ_comp, β, True
    else:
        # print("Compatibility solver is failed.")
        return xk, np.zeros_like(xk), None, False




def criticality_check(xk, drw, objective_function, surrogate_model, blackbox_idx):
    grad_f = fin_diff(objective_function, xk, drw, epsilon=1).flatten()
    grad_rw = fin_diff(surrogate_model.predict, xk, epsilon=1)

    # grad_f /= np.linalg.norm(grad_f) + 1e-8  # Normalize gradient
    # grad_rw /= np.linalg.norm(grad_rw, axis=0) + 1e-8

    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]

    def criticality_obj(v):
        return np.abs(np.dot(grad_f, v))

    def eq_constraint_h(v):
        return np.dot(grad_rw[:, graybox_idx], v[graybox_idx])

    def eq_constraint_r(v):
        v_y = v[blackbox_idx]
        v_w = v[graybox_idx]
        return v_y - np.dot(grad_rw[:, graybox_idx], v_w)

    def ineq_constraint_lower(v):
        return v + xk  # Ensure xk + v >= 0

    def ineq_constraint_upper(v):
        return 1e3 - (xk + v)  # Ensure xk + v <= 1e3

    constraints = [
        {'type': 'eq', 'fun': eq_constraint_h},
        {'type': 'eq', 'fun': eq_constraint_r},
        {'type': 'ineq', 'fun': ineq_constraint_lower},
        {'type': 'ineq', 'fun': ineq_constraint_upper},
    ]

    bounds = [(-1, 1) for _ in range(len(xk))]

    result = minimize(
        criticality_obj,
        np.zeros_like(xk),
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        # print("Constraint evaluations at criticality solution:")
        # for constraint in constraints:
        #     print(constraint['fun'](result.x))

        Chi_k = result.fun * 1e7
        xk_critical = result.x
        #print("Criticality condition satisfied. Adjusted xk:", xk_critical)
        return xk_critical, Chi_k, True
    else:
        # print("Criticality condition failed.")
        return xk, 1e5, False




def adaptive_TRSP_solver(xk, Δ_, surrogate_model, rigorous_model, blackbox_idx, obj_function, tolerance=2e-1):
    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]

    def TRSPk(xk):
        rw = surrogate_model.predict(xk)
        # yk = rigorous_model.predict(xk)[graybox_idx]
        yk = rigorous_model.predict(xk)[graybox_idx]
        drw = np.zeros(len(xk), dtype=float)
        drw[blackbox_idx] = rw
        drw[graybox_idx] = yk
        return obj_function(xk, drw)

    def TRSPconstraints(xk, Δ_):
        constraints = [
            # Trust region constraint for blackbox variables only
            {'type': 'ineq', 'fun': lambda x: Δ_[0] - np.linalg.norm([x[i] - xk[i] for i in blackbox_idx])},
            # {'type': 'ineq', 'fun': lambda x: Δ_[0] - np.linalg.norm(x - xk)},  # Trust region as norm ball
            {'type': 'ineq', 'fun': lambda x: x},  # Non-negativity
            {'type': 'ineq', 'fun': lambda x: 1e3 - x},  # Upper bound
        ]
        return constraints

    bounds = [
        (max(0, xk[i] - Δ_[0]), min(1e3, xk[i] + Δ_[0])) for i in range(len(xk))
    ]

    for i, (lb, ub) in enumerate(bounds):
        if lb > ub:
            # print(f"Warning: Adjusting invalid bounds for x[{i}]: lower = {lb}, upper = {ub}")
            bounds[i] = (ub, ub)  # Set bounds to valid range if they collapse

    result = minimize(
        TRSPk,
        xk,
        method='trust-constr',
        bounds=bounds,
        constraints=TRSPconstraints(xk, Δ_),
        options={'disp': False, 'maxiter': 10, 'gtol': tolerance},
    )

    if result.success:
        f = result.fun
        xk_trspk = result.x
        sk = xk_trspk - xk
        # print("Constraint evaluations at TRSPk solution:")
        # for constraint in TRSPconstraints(xk, Δ_):
        #     print(constraint['fun'](result.x))
        return f, xk_trspk, sk, Δ_, True
    else:
        print("TRSPk failed to converge.")
        return np.inf, xk, np.zeros_like(xk), Δ_, False




def restoration(xk, Δ_, surrogate_model, rigorous_model, θmin, fmin, blackbox_idx, TRFparams, Fθ, radii):
    """
    Restoration Procedure with Termination Based on Filter Acceptability.
    """
    global rw_global, dw_global
    max_retries = 80
    retry_count = 0
    step_size = 0.1  # Initial step size for directional steps
    radius_validator = KfullyLinearization(surrogate_model, rigorous_model, TRFparams['κf'], TRFparams['κg'])
    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]

    # Expand Δ_ to match the size of xk
    expanded_Δ = np.zeros_like(xk)
    expanded_Δ[blackbox_idx] = Δ_

    # Define bounds based on expanded trust region radius
    lower_bounds = xk - expanded_Δ
    upper_bounds = xk + expanded_Δ

    # Initialize sampling region radius (σ_)
    σ_ = np.clip(expanded_Δ * 0.5, 1e-3, np.max(expanded_Δ))

    min_f = np.inf
    min_θ = np.inf
    min_xk_attempt = xk.copy()

    while retry_count < max_retries:
        count = 0
        max = 20  # Maximum retries for the inner loop
        # Ensure κ-full linearity in the inner loop
        while count < max:
            if not radius_validator.is_κ_fully_linear(xk, σ_, blackbox_idx):
                # print(f"Trust region with Δ_ = {Δ_} is not κ-fully linear. Adjusting.")
                σ_ *= 10  # Adjust the sampling region
                count += 1
            else:
                # print(f"Trust region with Δ_ = {Δ_} is κ-fully linear at {count+1}. Step.")
                break  # Exit the inner loop if κ-full linearity is achieved

        # Generate candidate points
        step_direction = np.random.uniform(-1, 1, size=xk.shape)
        step_direction /= np.linalg.norm(step_direction)  # Normalize direction vector
        xk_attempt = xk + step_direction * step_size
        xk_attempt = np.clip(xk_attempt, lower_bounds, upper_bounds)

        # Model predictions
        rw_global = surrogate_model.predict(xk_attempt)
        dw_global = rigorous_model.predict(xk_attempt)[blackbox_idx]
        yk_global = rigorous_model.predict(xk_attempt)[graybox_idx]

        drw = np.zeros(len(xk), dtype=float)
        drw[blackbox_idx] = rw_global
        for idx, val in zip(graybox_idx, yk_global):
            drw[idx] = val

        # Calculate infeasibility (θk) and objective function (fk)
        θk = Infeasibility(rw_global, dw_global)
        fk = obj_function(xk_attempt, drw)

        # Filter compatibility checks
        Cf = fk <= fmin - TRFparams['γf'] * θmin
        Cθ = θk <= (1 - TRFparams['γθ']) * θmin

        if Cf or Cθ:
            expanded_Δ = expanded_Δ[blackbox_idx]
            σ_ = σ_[blackbox_idx]
            Fθ.append([fk, θk])
            radii.append([expanded_Δ, σ_])
            if fk < min_f:
                min_f = fk
                min_xk_attempt = xk_attempt.copy()
            print(f"Restoration succeeded: fk = {fk:.3f}, θk = {θk:.3f}")
            return xk_attempt, expanded_Δ, σ_, fk, θk, True

        # Update sampling region and step size
        step_size *= 0.9 if θk < min_θ else 1.1
        Δ_ *= TRFparams['γc'] if θk < min_θ else TRFparams['γe']
        σ_ *= TRFparams['γc'] if θk < min_θ else TRFparams['γe']
        min_θ = min(min_θ, θk)
        retry_count += 1

        # Dynamic exploration after failures
        if retry_count % 10 == 0:
            print("Exploring broader domain.")
            # expanded_Δ *= TRFparams['γe']
            # σ_ = np.clip(expanded_Δ*0.9, 1e-3, np.max(expanded_Δ))
            expanded_Δ *= 5
            σ_ *= 4.5
            lower_bounds = xk - expanded_Δ
            upper_bounds = xk + expanded_Δ

    print("Restoration failed: No filter-compatible point found.")
    return min_xk_attempt, expanded_Δ, σ_, min_f, min_θ, False




def evaluate_filter_conditions(f_trspk, θ_trspk, fmin, θmin, fk, θk, xk_trspk, xk_, sk, Δ_, σ_, Fθ, radii, TRFparams, k):
    global blackbox_idx
    """
    Evaluate filter conditions and update trust region parameters.

    :param f_trspk: Objective function value from TRSPk.
    :param θ_trspk: Infeasibility measure from TRSPk.
    :param fmin: Current filter objective function value.
    :param θmin: Current filter infeasibility value.
    :param xk_trspk: Proposed solution from TRSPk.
    :param sk: Step taken in the decision variables.
    :param Δ_trspk: Proposed trust region size from TRSPk.
    :param σ_trspk: Sampling region size from TRSPk.
    :param Fθ: Filter set (list of tuples with f and θ values).
    :param radii: List of current trust region radii.
    :param TRFparams: Dictionary of trust region filter parameters.
    :param blackbox_idx: Indices of black-box decision variables.
    :param xk_: Current decision variable vector.
    :param θ: Current infeasibility value.
    :param f: Current objective function value.
    :param Δ_: Current trust region radii for black-box variables.
    :param k: Current iteration index.
    :return: Updated xk_, θ, f, Δ_, Fθ, radii, k.
    """
    step_acceptance = False
    step_type = None

    # Check filter conditions
    Cf = f_trspk <= fmin - TRFparams['γf'] * θmin
    Cθ = θ_trspk <= (1 - TRFparams['γθ']) * θmin

    # print(f"before sk: {sk}")
    # print(f"before Δ_: {Δ_}")

    # sk = sk[blackbox_idx]  # Filter step change vector for blackbox variables
    # sk = sk * 1e3 # Increasing magnitude effect due to low impact to trust region
    # sk = np.clip(sk, -10, 10)

    # Min-Max normalization while retaining the sign
    sk_array = np.array(sk)
    min_sk, max_sk = np.min(sk_array), np.max(sk_array)
    sk_normalized = ((sk_array - min_sk) / (max_sk - min_sk))  # Normalize to [0, 1]
    # Scale to [-1, 1] while retaining the original signs
    sk_signed_normalized = sk_normalized * 2 - 1
    sk_scaled = sk_signed_normalized * 5
    print(f"sk_scaled: {sk_scaled}")
    sk_blackbox = sk_scaled[blackbox_idx]
    Δ_ = Δ_[blackbox_idx]  # Filter trust region size for blackbox variables
    σ_ = σ_[blackbox_idx]

    # print(f"after sk: {sk}")
    # print(f"after Δ_: {Δ_}")

    if Cf or Cθ:
        step_acceptance = True
        print("Step is accepted.")

        switching_condition = [fk - f_trspk >= TRFparams['κθ'] * θk ** TRFparams['γs']]

        if switching_condition and θ_trspk <= θmin:
            step_type = 'F-Type'
            print("F-type Step.")
            # print(f"Δ_: {Δ_}")
            # print(f"TRFparams[γe] * np.abs(sk_blackbox): {TRFparams['γe'] * np.abs(sk_blackbox)}")
            Δ_ = np.maximum(TRFparams['γe'] * np.abs(sk_blackbox), Δ_)  # Expand trust region
            # Δ_ = np.maximum(TRFparams['γe'] * np.max(np.abs(sk)), Δ_)  # Expand trust region
            σ_ = σ_
        else:
            step_type = 'θ-Type'
            print("θ-type Step.")
            # Step quality ratio for derivative-free optimization
            ρ = (θk - θ_trspk + TRFparams['ϵθ']) / max(θ_trspk, TRFparams['ϵθ'])
            print(f"ρ: {ρ}")
            if ρ < TRFparams['η1']:
                status = 'θtypeShrinkage'
                # print(f"TRFparams[γc] * np.abs(sk_blackbox)): {TRFparams['γc'] * np.abs(sk_blackbox)}")
                Δ_ = TRFparams['γc'] * np.abs(sk_blackbox)  # Shrink trust region
                # Δ_ = TRFparams['γc'] * np.max(np.abs(sk))  # Shrink trust region
                print("Shrinking filter size.")
            elif ρ > TRFparams['η2']:
                status = 'θtypeExpansion'
                # print(f"Δ_: {Δ_}")
                # print(f"TRFparams[γe] * np.abs(sk_blackbox): {TRFparams['γe'] * np.abs(sk_blackbox)}")
                Δ_ = np.maximum(TRFparams['γe'] * np.abs(sk_blackbox), Δ_)  # Expand trust region
                # Δ_ = np.maximum(TRFparams['γe'] * np.max(np.abs(sk)), Δ_)  # Expand trust region
                print("Expanding filter size.")
            else:
                print("No filter size update.")
                Δ_ = Δ_
                status = 'θtypeNoUpdate'
            σ_ = np.minimum(σ_, TRFparams["Ψ"] * Δ_)
        θk = θ_trspk
        fk = f_trspk
        xk_ += sk_scaled
        Fθ.append([fk, θk])
        radii.append([Δ_, σ_])
    else:
        # Step is not acceptable; shrink trust region
        print("The step is not accepted. Shrinking trust region.")
        xk_ = xk_
        θk = θk
        fk = fk
        # print(f"TRFparams[γc] * np.abs(sk_blackbox)): {TRFparams['γc'] * np.abs(sk_blackbox)}")
        Δ_ = TRFparams['γc'] * np.abs(sk_blackbox)  # Shrink trust region
        # Δ_ = TRFparams['γc'] * np.max(np.abs(sk))  # Shrink trust region
        σ_ = np.minimum(σ_, TRFparams["Ψ"] * Δ_)
        # print(f"σ_: {σ_}")
        # print(f"Δ_: {Δ_}")

    # Δ = Δ_ / xk_[blackbox_idx]

    # print(f"all after Δ_: {Δ_}")

    return fk, θk, xk_, Δ_, σ_, Fθ, radii, k




def adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function, surrogate_model, rigorous_model,
                               Infeasibility, radius_validator, blackbox_idx, graybox_idx, TRFparams):
    global restart_count, max_restart

    if restart_count >= max_restart:
        print(f"Maximum Restart Limit is Reached {restart_count+1}.")
        σ_ = Δ_.copy()
        return 1e10, 1e10, xk_, Δ_, σ_

    print("Adaptive Restart Mechanism Called...")
    restart_count += 1
    # Restart with respect to Objective Gradient
    print("Restart by Optimality Gradient.")
    grad_f = fin_diff(obj_function, xk_, drw, epsilon=1e-5).flatten()
    # print(f"Objective Function Gradient: {grad_f}")

    # Propose a new point along the negative gradient direction of the objective function
    lr = 0.1  # Step size for gradient
    drw_proposed = np.zeros(len(xk_))

    xk_proposed = xk_ - lr * grad_f
    rw_proposed = surrogate_model.predict(xk_proposed)
    dw_proposed = rigorous_model.predict(xk_proposed)[blackbox_idx]
    yk_proposed = rigorous_model.predict(xk_proposed)[graybox_idx]
    drw_proposed[blackbox_idx] = rw_proposed
    drw_proposed[graybox_idx] = yk_proposed
    fk_proposed = obj_function(xk_proposed, drw_proposed)
    θ_proposed = Infeasibility(rw_proposed, dw_proposed)

    # Check feasibility
    if θ_proposed <= TRFparams["ϵθ"]:
        print("Proposed point is feasible.")
        σ_, Δ_ = radius_validator.MinFeasRadiiFinder(xk_proposed, Δ_, blackbox_idx)
        return fk_proposed, θ_proposed, xk_proposed, Δ_, σ_

    # Restart with respect to Infeasibility Gradient
    print("Failed. Restart by Infeasibility Gradient.")
    drw_proposed2 = np.zeros(len(xk_))
    grad_θ = fin_diff(lambda x: Infeasibility(surrogate_model.predict(x).ravel(), rigorous_model.predict(x)[blackbox_idx].ravel()), xk_,
                      epsilon=1e-5).flatten()
    # print(f"Infeasibility Gradient: {grad_θ}")

    xk_proposed2 = xk_ - lr * grad_θ
    rw_proposed2 = surrogate_model.predict(xk_proposed2)
    dw_proposed2 = rigorous_model.predict(xk_proposed2)[blackbox_idx]
    yk_proposed2 = rigorous_model.predict(xk_proposed2)[graybox_idx]
    drw_proposed2[blackbox_idx] = rw_proposed2
    drw_proposed2[graybox_idx] = yk_proposed2
    fk_proposed2 = obj_function(xk_proposed2, drw_proposed2)
    θ_proposed2 = Infeasibility(rw_proposed2, dw_proposed2)

    # Check feasibility
    if θ_proposed2 <= TRFparams["ϵθ"]:
        print("Proposed point is feasible.")
        σ_, Δ_ = radius_validator.MinFeasRadiiFinder(xk_proposed2, Δ_, blackbox_idx)
        return fk_proposed2, θ_proposed2, xk_proposed2, Δ_, σ_

    # Historical Fallback
    print("Gradient-enforced failed. Resorting to historical knowledge.")
    if memory:
        x_restart = min(memory, key=lambda x: np.linalg.norm(x - xk_))
        print(f"Restarting with historical point: {x_restart}")
    else:
        x_restart = xk_  # Default to current point if no memory available
        print("No feasible memory available. Restarting from current point.")

    drw_restart = np.zeros(len(xk_))
    xk_restart = xk_ - lr * grad_θ
    rw_restart = surrogate_model.predict(x_restart)
    dw_restart = rigorous_model.predict(x_restart)[blackbox_idx]
    yk_restart = rigorous_model.predict(x_restart)[graybox_idx]
    drw_restart[blackbox_idx] = rw_restart
    drw_restart[graybox_idx] = yk_restart
    fk_restart = obj_function(x_restart, drw_restart)
    θ_restart = Infeasibility(rw_restart, dw_restart)
    # Adjust trust region parameters
    σ_, Δ_ = radius_validator.MinFeasRadiiFinder(x_restart, Δ_, blackbox_idx)

    return fk_restart, θ_restart, xk_restart, Δ_, σ_




def check_stagnation(Fθ_iter, fk, θk, TRFparams, stagnation, max_stagnation=3, window=5):

    if len(Fθ_iter) == 0:
        print("Filter is empty, adding the first point.")
        Fθ_iter.append([np.inf, np.inf])
    # Historical averages over the last `window` iterations
    elif len(Fθ_iter) >= window:
        historical_avg_fk = np.mean([entry[0] for entry in Fθ_iter[-window:]])
        historical_avg_θk = np.mean([entry[1] for entry in Fθ_iter[-window:]])
    else:
        historical_avg_fk, historical_avg_θk = Fθ_iter[-1][0], Fθ_iter[-1][1]

    # Check for improvement
    if abs(fk - historical_avg_fk) < TRFparams['ϵf'] and abs(θk - historical_avg_θk) < TRFparams['ϵθ']:
        stagnation += 1
    else:
        stagnation = 0  # Reset stagnation counter on significant improvement

    # Check if stagnation threshold is met
    if stagnation >= max_stagnation:
        print("Stagnation detected.")
        return True, stagnation
    return False, stagnation

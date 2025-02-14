import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from scipy.optimize import minimize
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
        if callable(self.rigorous_model):
            dw_pred = self.rigorous_model(xk_)
        else:
            dw_pred = self.rigorous_model.predict(xk_)

        # Check and filter dimensions
        if len(rw_pred) != len(blackbox_idx):
            rw_pred = rw_pred[blackbox_idx]

        # # Debugging output
        # print(f"Filtered rw_pred: {rw_pred}")
        # print(f"Filtered dw_pred: {dw_pred}")

        if isinstance(dw_pred, tuple):
            dw_pred = np.array(dw_pred)

        # Ensure `blackbox_idx` is a NumPy array
        blackbox_idx = np.array(blackbox_idx)

        # Apply indexing
        if len(dw_pred) != len(blackbox_idx):
            dw_pred = dw_pred[blackbox_idx]

        # Compute gradients
        # rw_GRAD = self.fin_diff_kproperty(self.surrogate_model.predict, xk_)
        # dw_GRAD = self.fin_diff_kproperty(self.rigorous_model.predict, xk_)
        # Compute gradients
        # rw_GRAD = self.fin_diff_kproperty(self.surrogate_model.predict, xk_)
        rw_GRAD = self.fin_diff_kproperty(lambda x: self.surrogate_model.predict(x), xk_)
        # dw_GRAD = self.fin_diff_kproperty(self.rigorous_model.predict, xk_)
        dw_GRAD = self.fin_diff_kproperty(lambda x: self.rigorous_model.predict(x), xk_)

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
    def MinFeasRadiiFinder(self, xk_, Δk, blackbox_idx, max_iter=50, tolerance=1e-6, verbose=False):
        """
        :param xk_: Current iterate for decision variables.
        :param Δk: Trust region radius (already for blackbox variables).
        :param blackbox_idx: Indices of blackbox variables.
        :param max_iter: Maximum iterations for bisection.
        :param tolerance: Tolerance for convergence.
        :param verbose: If True, prints debugging information.
        :return: Final σ_ (sampling region radii) and adjusted Δk.
        """
        # Ensure Δk contains only blackbox variables
        lower_bound = np.zeros_like(Δk)
        upper_bound = 2 * Δk

        for iteration in range(max_iter):
            is_fully_linear = self.is_κ_fully_linear(xk_, Δk, blackbox_idx)

            if is_fully_linear:
                upper_bound = Δk.copy()
            else:
                lower_bound = Δk.copy()

            # Update σ_ as the midpoint of lower and upper bounds
            Δk = (lower_bound + upper_bound) / 2

            if verbose:
                print(f"Iteration {iteration + 1}: Δk = {Δk}, lower_bound = {lower_bound}, upper_bound = {upper_bound}")

            # Check convergence
            if np.linalg.norm(upper_bound - lower_bound) < tolerance:
                break

        # Clip for numerical stability
        Δk = np.clip(Δk, 1e-10, 1e5)

        if verbose:
            print(f"Final σ_: {Δk}")

        return Δk


def compatibility_check(xk, yk, surrogate_model, real_model, blackbox_idx, Δ, κΔ, κμ, μ):
    """
    Perform compatibility check for the current trust region subproblem (TRSP).
    This function attempts to find a correction term `nk` such that compatibility is achieved,
    with the objective function normalized to a range of [0, 5].

    :param xk: Current iterate for the input variables.
    :param yk: Current output from the rigorous model.
    :param surrogate_model: The surrogate model to use for prediction.
    :param blackbox_idx: Indices of the blackbox variables.
    :param Δ: Trust region size array for the blackbox variables.
    :param κΔ: Scaling factor for trust region adjustment.
    :param κμ: Scaling factor for criticality adjustment.
    :param μ: Trade-off parameter for criticality adjustment.
    :return: Updated xk (if compatible), radii update, compatibility result (β), success flag.
    """
    # Define min and max limits
    min_limits = np.array([60, 70, 50, 50, 50, 0.00001, 0.00001, 0.00001, 0.00001])
    max_limits = np.array([240, 150, 1000, 600, 600, 1, 30, 1, 1])

    # Compute adjustment factor
    adjustment_factor = κΔ * Δ * np.minimum(1, κμ * Δ ** μ)

    # Debugging adjustment factor
    print("Adjustment factors for blackbox variables:")
    for idx, adj in zip(blackbox_idx, adjustment_factor):
        print(f"Variable {idx}: Adjustment factor = {adj:.6f}")

    # Define bounds for blackbox variables
    bounds = [
        (max(min_limits[idx], xk[idx] - adjustment_factor[blackbox_idx.index(idx)]),
         min(max_limits[idx], xk[idx] + adjustment_factor[blackbox_idx.index(idx)]))
        for idx in blackbox_idx
    ]

    # Extend bounds for graybox variables
    bounds_full = [
        (min_limits[i], max_limits[i]) if i not in blackbox_idx else bounds[blackbox_idx.index(i)]
        for i in range(len(xk))
    ]

    # Debugging bounds
    print("Bounds for each variable during compatibility check:")
    for i, (lb, ub) in enumerate(bounds_full):
        print(f"x[{i}]: Lower = {lb:.6f}, Upper = {ub:.6f}")
        if lb > ub:
            print(f"Warning: Invalid bounds for x[{i}]. Adjusting to lb = ub = {lb:.6f}")
            bounds_full[i] = (lb, lb)  # Fix bounds

    # Objective function
    def compatibility_obj(nk):
        xk_adjusted = xk + nk
        rk_adjusted = surrogate_model.predict(xk_adjusted)
        yk = real_model.predict(xk_adjusted)[blackbox_idx]
        residual = np.linalg.norm(yk - rk_adjusted)
        # print(f'xk_adjusted: {xk_adjusted}')
        # print(f'rk_adjusted: {rk_adjusted}')
        # print(f'yk: {yk}')

        # # Normalize the residual
        # min_residual, max_residual = 0, 100  # Adjust based on expected residual range
        # normalized_residual = 5 * (residual - min_residual) / (max_residual - min_residual)
        # normalized_residual = np.clip(normalized_residual, 0, 5)  # Clip to the range [0, 5]

        # Debugging residual
        # print(f"Objective residual at nk = {nk}: {normalized_residual:.6f} (normalized)")
        return residual

    # # Constraints
    # constraints = [
    #     {'type': 'ineq', 'fun': lambda nk: xk + nk - min_limits},  # x + nk >= min_limits
    #     {'type': 'ineq', 'fun': lambda nk: max_limits - (xk + nk)}  # x + nk <= max_limits
    # ]
    constraints = []
    # Constraint for trust-region bounds on blackbox variables
    for i in blackbox_idx:
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: Δ_[i] - abs(x[i] - xk[i])})
    # Constraints for graybox variables: Ensure within min/max limits
    for i in range(len(xk)):
        if i not in blackbox_idx:
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - min_limits[i]})  # x[i] >= min_limits[i]
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_limits[i] - x[i]})  # x[i] <= max_limits[i]

    # Initial guess for nk
    nk_initial = np.zeros_like(xk)

    # Debugging initial guess
    # print("Initial guess for nk:", nk_initial)

    # Optimization
    print("Starting compatibility optimization...")
    result = minimize(
        compatibility_obj,
        nk_initial,
        bounds=bounds_full,
        constraints=constraints,
        method='Nelder-Mead',
        # options={'disp': True, 'maxiter': 200, 'gtol': 1e-1},
        options={'disp': True, 'maxiter': 200, 'fatol': 1e-3}
    )

    if result.success:
        nk = result.x
        xk_minimizer = xk + nk
        Δ_comp = np.zeros_like(xk)

        # Update Δ only for blackbox variables
        for idx in blackbox_idx:
            Δ_comp[idx] = adjustment_factor[blackbox_idx.index(idx)]

        β = result.fun  # Compatibility result

        # Debugging successful result
        print("Compatibility optimization successful.")
        print("Resulting nk:", nk)
        print("Updated xk:", xk_minimizer)
        print("Residual (β):", β)

        return xk_minimizer, Δ_comp, β, True
    else:
        # Debugging failed result
        print("Compatibility optimization failed.")
        print("Residual at failure:", result.fun if 'fun' in result else "Not computed")
        return xk, np.zeros_like(xk), None, False




def criticality_check(xk, drw, objective_function, surrogate_model, blackbox_idx):

    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]
    # Construct `drw` with predictions from both models
    drw = np.zeros_like(xk)
    drw[blackbox_idx] = surrogate_model.predict(xk)  # Assign surrogate predictions
    drw[graybox_idx] = placeholder.predict(xk)[graybox_idx]  # Assign placeholder predictions

    # Compute gradient of the objective function using dynamic `drw`
    grad_f = fin_diff(
        lambda x: obj_function(
            x,
            np.concatenate([
                surrogate_model.predict(x),        # Surrogate predictions
                placeholder.predict(x)[graybox_idx]  # Placeholder predictions for graybox_idx
            ])
        ),
        xk,
        epsilon=1e-5
    ).flatten()

    # Compute gradient of surrogate model predictions (`grad_rw`)
    grad_rw = fin_diff(lambda x: surrogate_model.predict(x), xk, epsilon=1e-5)

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


def TRSP_solver(xk, Δ_, surrogate_model, rigorous_model, blackbox_idx, obj_function, tolerance=1e-1):
    # Define variable-specific bounds
    min_limits = np.array([60, 70, 50, 50, 50, 0.00001, 0.00001, 0.00001, 0.00001])
    max_limits = np.array([240, 150, 1000, 600, 600, 1, 30, 1, 1])

    graybox_idx = [i for i in range(len(xk)) if i not in blackbox_idx]

    def TRSPk(xk):
        rw = surrogate_model.predict(xk)
        yk = rigorous_model.predict(xk)[graybox_idx]
        drw = np.zeros(len(xk), dtype=float)
        drw[blackbox_idx] = rw
        drw[graybox_idx] = yk
        # print(f'rw: {rw}')
        # print(f'yk: {yk}')
        # print(f'drw: {drw}')
        return obj_function(xk, drw)

    # Define bounds for the optimization
    bounds = []
    for i in range(len(xk)):
        if i in blackbox_idx:
            lb = max(min_limits[i], xk[i] - Δ_[i])
            ub = min(max_limits[i], xk[i] + Δ_[i])
            # print(f"Δ_[i]:{Δ_[i]}")
            # print(f"max(min_limits[i], xk[i] -  Δ_[i]):{max(min_limits[i], xk[i] -  Δ_[i])}")
            # print(f"min(max_limits[i], xk[i] +  Δ_[i]):{min(max_limits[i], xk[i] +  Δ_[i])}")
            # print(f"xk[i] -  Δ_[i]:{xk[i] -  Δ_[i]}")
            # print(f"xk[i] +  Δ_[i]:{xk[i] +  Δ_[i]}")

        else:
            lb = min_limits[i]
            ub = max_limits[i]
        bounds.append((lb, ub))

    # Debugging bounds
    print("Variable bounds:")
    for i, (lb, ub) in enumerate(bounds):
        # print(f"x[{i}]: Lower = {lb}, Upper = {ub}")
        if lb > ub:
            print(f"Warning: Invalid bounds detected for x[{i}]. Adjusting to lb = ub = {lb}")
            bounds[i] = (lb, lb)  # Collapse invalid bounds


    result = minimize(
        TRSPk,
        xk,
        bounds=bounds,
        method='trust-constr',
        options={
            'disp': True,
            'maxiter': 500,
            'gtol': tolerance,
            'xtol': 1e-6,  # Add tighter control on step sizes
        }
    )

    if not result.success:
        # Retry with a larger trust region or fallback to a global optimizer
        Δ_ *= 1.5  # Expand trust region for next iteration
        result = differential_evolution(TRSPk, bounds, strategy='best1bin', maxiter=300)

    if result.success:
        f = result.fun
        xk_trspk = result.x
        sk = xk_trspk - xk

        # Debug step sizes
        print("Step sizes (sk) for each variable:")
        for i, step in enumerate(sk):
            print(f"x[{i}]: Step size = {step:.6f}")

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

    min_f = np.inf
    min_θ = np.inf
    min_xk_attempt = xk.copy()

    while retry_count < max_retries:
        count = 0
        max = 20  # Maximum retries for the inner loop
        # Ensure κ-full linearity in the inner loop
        while count < max:
            if not radius_validator.is_κ_fully_linear(xk, Δ_, blackbox_idx):
                # print(f"Trust region with Δ_ = {Δ_} is not κ-fully linear. Adjusting.")
                Δ_ *= 10  # Adjust the sampling region
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
            Fθ.append([fk, θk])
            radii.append([expanded_Δ])
            if fk < min_f:
                min_f = fk
                min_xk_attempt = xk_attempt.copy()
            print(f"Restoration succeeded: fk = {fk:.3f}, θk = {θk:.3f}")
            return xk_attempt, expanded_Δ, fk, θk, True

        # Update sampling region and step size
        step_size *= 0.9 if θk < min_θ else 1.1
        Δ_ *= TRFparams['γc'] if θk < min_θ else TRFparams['γe']
        min_θ = min(min_θ, θk)
        retry_count += 1

        # Dynamic exploration after failures
        if retry_count % 10 == 0:
            print("Exploring broader domain.")
            # expanded_Δ *= TRFparams['γe']
            # σ_ = np.clip(expanded_Δ*0.9, 1e-3, np.max(expanded_Δ))
            expanded_Δ *= 5
            lower_bounds = xk - expanded_Δ
            upper_bounds = xk + expanded_Δ

    print("Restoration failed: No filter-compatible point found.")
    return min_xk_attempt, expanded_Δ, min_f, min_θ, False





def check_stagnation(Fθ_iter, fk, θk, TRFparams, stagnation, max_stagnation=3, window=2):
    # Filter out invalid entries
    Fθ_iter = [entry for entry in Fθ_iter if entry[0] is not None and entry[1] is not None]

    # Handle empty or invalid `Fθ_iter`
    if len(Fθ_iter) == 0:
        print("Filter is empty or invalid, adding the first point.")
        Fθ_iter.append([np.inf, np.inf])

    # Validate `fk` and `θk`
    if fk is None or θk is None:
        print("Invalid `fk` or `θk`. Skipping stagnation check.")
        return False, stagnation

    # Historical averages over the last `window` iterations
    if len(Fθ_iter) >= window:
        historical_avg_fk = np.mean([entry[0] for entry in Fθ_iter[-window:]])
        historical_avg_θk = np.mean([entry[1] for entry in Fθ_iter[-window:]])
    else:
        historical_avg_fk, historical_avg_θk = Fθ_iter[-1][0], Fθ_iter[-1][1]

    # Handle cases where historical averages might be invalid
    if historical_avg_fk is None or historical_avg_θk is None:
        print("Invalid historical averages. Skipping stagnation check.")
        return False, stagnation

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




def evaluate_filter_conditions(f_trspk, θ_trspk, fmin, θmin, fk, θk, xk_trspk, xk_, sk, Δ_, Fθ, radii, TRFparams, k):
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

    # Min-Max normalization while retaining the sign
    sk_array = np.array(sk)
    min_sk, max_sk = np.min(sk_array), np.max(sk_array)
    sk_normalized = ((sk_array - min_sk) / (max_sk - min_sk))  # Normalize to [0, 1]
    # Scale to [-1, 1] while retaining the original signs
    sk_signed_normalized = sk_normalized * 2 - 1
    sk_scaled = sk_signed_normalized * 5
    print(f'sk_scaled: {sk_scaled}')
    sk_blackbox = sk_scaled[blackbox_idx]
    Δ_ = Δ_[blackbox_idx]  # Filter trust region size for blackbox variables

    # print(f"after sk: {sk}")
    # print(f"after Δ_: {Δ_}")

    if Cf or Cθ:
        step_acceptance = True
        print("Step is accepted.")

        switching_condition = [fk - f_trspk >= TRFparams['κθ'] * θk ** TRFparams['γs']]

        if switching_condition:
            step_type = 'F-Type'
            print("F-type Step.")
            Δ_ *= TRFparams['γe']  # Expand trust region
        else:
            step_type = 'θ-Type'
            print("θ-type Step.")
            # Step quality ratio for derivative-free optimization
            ρ = 1 - θ_trspk/θk
            print(f"ρ: {ρ}")
            if ρ < TRFparams['η1']:
                status = 'θtypeShrinkage'
                Δ_ *= TRFparams['γc']  # Shrink trust region
                print("Shrinking filter size.")
            elif ρ > TRFparams['η2']:
                status = 'θtypeExpansion'
                Δ_ *= TRFparams['γe']  # Expand trust region
                print("Expanding filter size.")
            else:
                print("No filter size update.")
                Δ_ = Δ_
                status = 'θtypeNoUpdate'
        θk = θ_trspk
        fk = f_trspk
        xk_ += sk_scaled
        Fθ.append([fk, θk])
        radii.append([Δ_])
    else:
        # Step is not acceptable; shrink trust region
        print("The step is not accepted. Shrinking trust region.")
        xk_ = xk_
        θk = θk
        fk = fk
        Δ_ *= TRFparams['γc']  # Shrink trust region
        # print(f"Δ_: {Δ_}")

    Δ = Δ_ / xk_[blackbox_idx]

    # print(f"all after Δ_: {Δ_}")

    return fk, θk, xk_, Δ_, Fθ, radii, k, step_acceptance




def adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function, surrogate_model, rigorous_model,
                               Infeasibility, blackbox_idx, graybox_idx, ϵθ):
    global restart_count, max_restart

    if restart_count >= max_restart:
        print(f"Maximum Restart Limit is Reached {restart_count+1}.")
        return 1e10, 1e10, xk_, Δ_

    print("Adaptive Restart Mechanism Called...")
    restart_count += 1
    # Restart with respect to Objective Gradient
    print("Restart wrt. Optimality Gradient.")
    drw = np.zeros_like(xk_)
    drw[blackbox_idx] = surrogate_model.predict(xk_)  # Assign surrogate predictions
    drw[graybox_idx] = rigorous_model.predict(xk_)[graybox_idx]  # Assign placeholder predictions

    # Compute gradient of the objective function using dynamic `drw`
    grad_f = fin_diff(
        lambda x: obj_function(
            x,
            np.concatenate([
                surrogate_model.predict(x),        # Surrogate predictions
                rigorous_model.predict(x)[graybox_idx]  # Placeholder predictions for graybox_idx
            ])
        ),
        xk_,
        epsilon=1e-5
    ).flatten()
    print(f"Objective Function Gradient: {grad_f}")

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
    if θ_proposed <= ϵθ:
        print("Proposed point is feasible.")
        return fk_proposed, θ_proposed, xk_proposed, Δ_

    # Restart with respect to Infeasibility Gradient
    print("Failed. Restart wrt. Infeasibility Gradient.")
    drw_proposed2 = np.zeros(len(xk_))
    grad_θ = fin_diff(lambda x: Infeasibility(surrogate_model.predict(x).ravel(), rigorous_model.predict(x)[blackbox_idx].ravel()), xk_,
                      epsilon=1e-5).flatten()
    print(f"Infeasibility Gradient: {grad_θ}")

    xk_proposed2 = xk_ - lr * grad_θ
    rw_proposed2 = surrogate_model.predict(xk_proposed2)
    dw_proposed2 = rigorous_model.predict(xk_proposed2)[blackbox_idx]
    yk_proposed2 = rigorous_model.predict(xk_proposed2)[graybox_idx]
    drw_proposed2[blackbox_idx] = rw_proposed2
    drw_proposed2[graybox_idx] = yk_proposed2
    fk_proposed2 = obj_function(xk_proposed2, drw_proposed2)
    θ_proposed2 = Infeasibility(rw_proposed2, dw_proposed2)

    # Check feasibility
    if θ_proposed2 <= ϵθ:
        print("Proposed point is feasible.")
        return fk_proposed2, θ_proposed2, xk_proposed2, Δ_

    # Historical Fallback
    print("Gradient-enforced failed. Resorting to historical knowledge.")
    if memory:
        x_restart = min(memory, key=lambda x: np.linalg.norm(x - xk_))
        print(f"Restarting with historical point: {x_restart}")
    else:
        x_restart = xk_  # Default to current point if no memory available
        print("No feasible memory available. Restarting from current point.")

    drw_restart = np.zeros(len(xk_))
    rw_restart = surrogate_model.predict(x_restart)
    dw_restart = rigorous_model.predict(x_restart)[blackbox_idx]
    yk_restart = rigorous_model.predict(x_restart)[graybox_idx]
    drw_restart[blackbox_idx] = rw_restart
    drw_restart[graybox_idx] = yk_restart
    fk_restart = obj_function(x_restart, drw_restart)
    θ_restart = Infeasibility(rw_restart, dw_restart)
    return fk_restart, θ_restart, x_restart, Δ_




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

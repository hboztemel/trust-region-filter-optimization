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




def modified_TRF_model(xk_initial, surrogate_model, rigorous_model, rigorous_placeholder, blackbox_idx,
                       iterations=50, TRFparams=None):
    global rw_global, dw_global, Fθ, radii, Fθ_iter, radii_iter, Chi_iter, restart_count, max_restart

    xk_ = xk_initial.copy()
    graybox_idx = [i for i in range(len(xk_)) if i not in blackbox_idx]
    Fθ = []  # Initial filter set
    radii = []  # Radii setup
    Fθ_iter = []
    radii_iter = []
    Chi_iter = []
    memory = []
    iter_time = []
    k = 0
    restoration_count = 0
    max_restoration = 5
    restart_count = 0
    max_restart = 8

    rw_global, dw_global = None, None

    radius_validator = KfullyLinearization(surrogate_model, rigorous_placeholder, TRFparams['κf'], TRFparams['κg'])
    Δ = TRFparams['Δ'] if TRFparams and 'Δ' in TRFparams else 1.0
    Δ_ = xk_[blackbox_idx] * Δ
    σ_, Δ_ = radius_validator.MinFeasRadiiFinder(xk_, Δ_, blackbox_idx)
    σ = σ_ / xk_[blackbox_idx]

    best_point = {"xk": None, "fk": np.inf, "θk": np.inf, "Δ_": None, "σ_": None}
    stagnation = 0

    for ite in range(iterations):
        start_time = time.time()
        print(f"\nIteration {ite + 1}")
        if np.any(np.isinf(xk_)) or np.any(np.isnan(xk_)):
            print("Invalid xk_ values detected. Terminating loop.")
            stop_time = time.time()  # Stop timing the iteration
            time.append(stop_time - start_time)
            break

        xk_ = np.clip(xk_, 0, 1e3)
        Δ_ = np.clip(Δ_, 0, 1e3)
        if np.any(Δ_ < σ_):  # Compare each element independently
            print("The update is made for σ ≤ Δ.")
            σ_, Δ_ = radius_validator.MinFeasRadiiFinder(xk_, Δ_, blackbox_idx)

        if len(Δ_) != len(blackbox_idx):
            Δ_ = Δ_[blackbox_idx]

        if len(σ_) != len(blackbox_idx):
            σ_ = σ_[blackbox_idx]

        print(f"Current point: {xk_}")
        print(f"Trust Region Radius (Δ): {Δ_}")
        print(f"Sampling Region Radius (σ): {σ_}")

        rw = surrogate_model.predict(xk_)
        print(f'Surrogate Model Output: {rw}')

        aspen_plus, sim_stat1 = rigorous_model(xk_)
        # aspen_plus = rigorous_model(xk_)
        # sim_stat1 = 1
        dw = aspen_plus[blackbox_idx]
        yk = aspen_plus[graybox_idx]
        print(f'Aspen Plus Simulation Output: {dw}')

        drw = np.zeros(len(xk_))
        drw[blackbox_idx] = rw
        drw[graybox_idx] = yk

        fk = obj_function(xk_, drw)
        θk = Infeasibility(rw, dw) if sim_stat1 else np.Inf

        if θk <= TRFparams["ϵθ"]:
            memory.append(xk_)

        if ite == 0:
            Fθ.append([fk, θk])
            radii.append([Δ_, σ_])
            Fθ_iter.append([fk, θk])
            radii_iter.append([Δ_, σ_])

        θmin = min(Fθ, key=lambda x: x[1])[1] if ite > 0 else 1e10
        fmin = min(Fθ, key=lambda x: x[0])[0] if ite > 0 else 1e10

        print(f"Objective Function (Profit, f): ${-fk:,.2f} at Step {ite + 1}.")
        print(f"Infeasibility (θ): {θk:.2f} at Step {ite + 1}.")

        # Criticality Check
        print('Criticality check starts.')
        xk_critical, Chi_k, crit_success = criticality_check(xk_, drw, obj_function, surrogate_model, blackbox_idx)
        Chi_iter.append(Chi_k)

        # print(f'xk_critical: {xk_critical}')
        # print(f'Chi_k: {Chi_k}')
        # print(f'TRFparams[ξ] * Δ_ / xk_[blackbox_idx]: {TRFparams["ξ"] * Δ_ / xk_[blackbox_idx]}')
        if crit_success:
            print('Solver is successful.')

            Δmin = Min.trfradius(radii)

            if k > 0:
                ϵΔ = np.maximum(np.full(len(blackbox_idx), TRFparams['ϵΔ']), Δmin)  # ϵΔ >= Δmin
                if θk <= TRFparams['ϵθ'] and Chi_k <= TRFparams['ϵχ'] and (σ_ <= ϵΔ).all():
                    print("First-order critical point has been found! Solution is feasible.")
                    super_print(Fθ_iter, radii_iter, ite=ite+1)
                    Fθ_iter.append([fk, θk])
                    radii_iter.append([Δ_, σ_])
                    stop_time = time.time()  # Stop timing the iteration
                    iter_time.append(stop_time - start_time)
                    break
                elif ((Δ_ <= Δmin).all() and (radii_iter[-1][0] <= Δmin).all() and θk <= TRFparams['ϵθ'] and
                      Fθ_iter[-1][1] <=
                      TRFparams['ϵθ']):
                    print("Feasible solution is found but global optimality is not guaranteed.")
                    super_print(Fθ_iter, radii_iter, ite=ite+1)
                    Fθ_iter.append([fk, θk])
                    radii_iter.append([Δ_, σ_])
                    stop_time = time.time()  # Stop timing the iteration
                    iter_time.append(stop_time - start_time)
                    break
            if (Chi_k < TRFparams['ξ'] * Δ_ / xk_[blackbox_idx]).all():
                σ_ = np.maximum(np.minimum(radii[-1][1], Chi_k / TRFparams["ξ"]), Δmin)  # Min.trfradius(radii) = Δmin
                # print(f"np.minimum(radii[-1][1]: {radii[-1][1]}")
                # print(f"(Chi_k / TRFparams[ξ]: {Chi_k / TRFparams["ξ"]}")
                # print(f"Δmin: {Δmin}")
                print("The TRSPk is in critical region. Sampling region shrinkage.")
                is_critical = True
            else:
                print("TRSPk is not in a critical region.")

            if ite > iterations * 0.25:
                if fk < Fθ[-1][0] * 0.95:
                    print("Adaptive Trust Region Update: Shrinkage.")
                    Δ_ *= TRFparams['γc']
                else:
                    print("Adaptive Trust Region Update: Expansion.")
                    Δ_ *= TRFparams['γe']
                if θk < Fθ[-1][1] * 0.95:
                    print("Adaptive Sampling Region Update: Shrinkage.")
                    σ_ *= TRFparams['γc']
                else:
                    print("Adaptive Sampling Region Update: Expansion.")
                    σ_ *= TRFparams['γe']
        else:
            if ite > 1:
                # print("Criticality failed. Terminating loop.")
                # find_best_point(Fθ_iter, radii_iter)
                # # break
                print('Solver failed. Restart Mechanism Starts...')
                # print("Failure in Criticality Solver. Restart Mechanism Starts...")
                fk, θk, xk_, Δ_, σ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                 surrogate_model, rigorous_placeholder, Infeasibility,
                                                                 radius_validator, blackbox_idx, graybox_idx, TRFparams)

        # Compatibility Check
        print('Compatibility check starts.')
        xk_minimizer, Δ_comp, β, compat_succ = compatibility_check(
            xk_, dw, surrogate_model, blackbox_idx, Δ_,
            TRFparams["κΔ"], TRFparams["κμ"], TRFparams["μ"])

        if compat_succ:
            print('Solver is successful.')
            print(f"β (Compatibility result): {β}")

            if β < TRFparams["ϵc"]:
                print(f'TRSPk is compatible.')

                # kth Trust Region Sub-Problem (TRSPk)
                print(f"Solving Trust Region Sub-Problem ({ite + 1})...")
                f_trspk, xk_trspk, sk, Δ_trspk, success = adaptive_TRSP_solver(xk_, Δ_comp,
                                                                               surrogate_model, rigorous_placeholder,
                                                                               blackbox_idx, obj_function)
                if success:
                    print("TRSPk found a solution.")
                    rw_trspk = surrogate_model.predict(xk_trspk)
                    aspen_plus, sim_stat2 = rigorous_model(xk_trspk)
                    # aspen_plus = rigorous_model(xk_trspk)
                    # sim_stat2 = 1
                    dw_trspk = aspen_plus[blackbox_idx]
                    yk_trspk = aspen_plus[graybox_idx]
                    θ_trspk = Infeasibility(rw_trspk, dw_trspk) if sim_stat2 else np.Inf
                    σ_trspk, Δ_trspk = radius_validator.MinFeasRadiiFinder(xk_trspk, Δ_trspk, blackbox_idx)

                    print("Filter Conditions:")
                    step_acceptance = False
                    step_type = False

                    fk, θk, xk_, Δ_, σ_, Fθ, radii, k = evaluate_filter_conditions(f_trspk, θ_trspk, fmin, θmin, fk,
                                                                                   θk, xk_trspk, xk_, sk, Δ_trspk,
                                                                                   σ_trspk, Fθ, radii, TRFparams, k)


                else:
                    # find_best_point(Fθ_iter, radii_iter)
                    super_print(Fθ_iter, radii_iter, ite=ite+1)
                    print("Failure in TRSPk Solver. Restart Mechanism Starts...")
                    fk, θk, xk_, Δ_, σ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                     surrogate_model, rigorous_placeholder,
                                                                     Infeasibility,
                                                                     radius_validator, blackbox_idx, graybox_idx,
                                                                     TRFparams)
            else:
                print(f'TRSPk is not compatible. Perturbating the current point for next iteration.')
                xk_[blackbox_idx] = np.random.uniform(xk_[blackbox_idx] - σ_, xk_[blackbox_idx] + σ_)

        else:
            print("TRSPk is not compatible.")
            Fθ.append([fk, θk])
            radii.append([Δ_, σ_])

            if restoration_count >= max_restoration * 2:
                print(
                    f"Restoration excessively called {restoration_count}. Feasible region is small, hence loop terminated.")
                Fθ_iter.append([fk, θk])
                radii_iter.append([Δ_, σ_])
                # find_best_point(Fθ_iter, radii_iter)
                super_print(Fθ_iter, radii_iter, ite=ite+1)
                stop_time = time.time()  # Stop timing the iteration
                iter_time.append(stop_time - start_time)
                break
            # elif restoration_count >= max_restoration:
            #     print("Too many restoration warning: Enhancing trust region and infeasibility threshold.")
            #     TRFparams['εθ'] *= 1.3  # Increase infeasibility threshold
            #     Δ_ *= 2  # Expand trust region
            #     Δ_ *= 2  # Expand trust region
            else:
                # Restoration Procedure if not compatible
                print("Restoration procedure is called.")
                restoration_count += 1
                (restored_point, Δ_, σ_, f_restored,
                 θ_restored, is_feasible) = restoration(xk_, Δ_, surrogate_model, rigorous_placeholder, θk,
                                                        fk, blackbox_idx, TRFparams, Fθ, radii)
                if is_feasible:  # Added to the filter set
                    xk_ = restored_point
                    # σ_, Δ_ = radius_validator.MinFeasRadiiFinder(xk_, Δ_restored, blackbox_idx)
                    fk = f_restored
                    θk = θ_restored
                    print(f"Restoration successful. Restored xk_: {xk_}  for next iteration.")
                    k += 1
                else:
                    print("Failure in Restoration.")
                    fk, θk, xk_, Δ_, σ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                     surrogate_model, rigorous_placeholder,
                                                                     Infeasibility,
                                                                     radius_validator, blackbox_idx, graybox_idx,
                                                                     TRFparams)

        Fθ_iter.append([fk, θk])
        radii_iter.append([Δ_, σ_])

        is_stagnant, stagnation = check_stagnation(Fθ_iter, fk, θk, TRFparams, stagnation)
        if is_stagnant:
            # print("Stagnation detected.")
            fk, θk, xk_, Δ_, σ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                             surrogate_model, rigorous_placeholder, Infeasibility,
                                                             radius_validator, blackbox_idx, graybox_idx, TRFparams)

        # print(f"fk: {fk}, θk: {θk}, Δ_: {Δ_}")
        if fk > 1e5 or θk > 1e5:
            print(f"Terminating the loop.")
            # find_best_point(Fθ_iter, radii_iter)
            super_print(Fθ_iter, radii_iter, ite=ite+1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        if np.all(σ_ <= 1e-5):
            print(f"Convergence achieved: σ has approached zero. Terminating loop.")
            # find_best_point(Fθ_iter, radii_iter)
            super_print(Fθ_iter, radii_iter, ite=ite+1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        if ite > 5 and abs(fk - Fθ_iter[-1][0]) < TRFparams['ϵf'] and abs(θk - Fθ_iter[-1][1]) < TRFparams['ϵθ']:
            print("Convergence achieved. Terminating loop.")
            super_print(Fθ_iter, radii_iter, ite=ite+1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        # print(f"Chi_k: {Chi_k}")
        # if ite >= int(0.15 * iterations):
        #     if crit_success and (Chi_k <= TRFparams['ϵχ']).all():
        #         print("Criticality conditions met. Terminating loop.")
        #         break

        if ite == iterations:
            # find_best_point(Fθ_iter, radii_iter)
            super_print(Fθ_iter, radii_iter, ite=ite+1)

        stop_time = time.time()  # Stop timing the iteration
        iter_time.append(stop_time - start_time)
        print(f"{ite + 1}. Iteration Time: {stop_time - start_time:2f} sec.")

    return Fθ, radii, Fθ_iter, radii_iter, Chi_iter, iter_time





# Run


TRFparams = {
    'κf': 0.5,   # Feasibility threshold scaling factor (unchanged)
    'κg': 0.5,   # Trust region scaling factor (unchanged)
    'γc': 0.4,   # Shrink factor (unchanged)
    'γe': 1.2,   # Expand factor (unchanged)
    'γf': 0.5,   # f-criterion reduction factor (unchanged)
    'γθ': 0.2,   # θ-criterion reduction factor (unchanged)
    'κθ': 1.2,   # Feasibility threshold scaling factor (unchanged)
    'κΔ': 0.5,   # Trust region scaling factor (unchanged)
    'κμ': 0.1,   # Criticality threshold scaling factor (unchanged)
    'μ': 0.3,    # Trade-off parameter (decreased to favor objective function)
    'γs': 0.9,   # Step size scaling factor (unchanged)
    'ξ': 1e-1,   # Criticality boundary scale parameter (unchanged)
    'ω': 0.2,    # Criticality boundary scale parameter (unchanged)
    'η1': 0.2,   # Quality ratio shrink factor (unchanged)
    'η2': 0.3,   # Quality ratio expand factor (unchanged)
    'Niter': 200,  # Maximum number of iterations (unchanged)
    'Δ': 2,      # Trust region size (unchanged)
    'ϵc': 3,     # Termination tolerance for compatibility (unchanged)
    'ϵχ': 1e-1,  # Termination tolerance for criticality (unchanged)
    'ϵθ': 1e-3,    # Termination tolerance for infeasibility (increased to reduce convergence thresholds)
    'ϵf': 1e-3,  # Termination tolerance for optimality (tightened for better optimality)
    'ϵσ': 1e-1,  # Termination tolerance for sampling region size (unchanged)
    'ϵΔ': 1e-1,  # Termination tolerance (unchanged)
    'Ψ': 0.5,    # Sampling region reset parameter (unchanged)
}

dataframe = pd.read_csv(
    r'.\MEOH_doe.csv')
num_input = 9
X = dataframe.iloc[:, :num_input]
y = dataframe.iloc[:, num_input:]

blackbox_idx = [0, 1, 2, 3]

linear_Smodel = LinearModel()
linear_Smodel.train(X, y, blackbox_idx=blackbox_idx)
print("\nLinearModel initiated.")

# Initialize the AspenPlusModel
meoh_path = r'.\AspenPlusSimulation'
MEOH = AspenPlusModel(meoh_path)
MEOH.initialize_aspen_plus()

xk_initial = np.array([80,50,90,200,100,12.063,36.053,0.964,18.490], dtype=float)
graybox_idx = [i for i in range(len(xk_initial)) if i not in blackbox_idx]

surrogate = linear_Smodel
rigorous=MEOH.run_aspen_plus
placeholder = placeholder


FθT, radiiT, Fθ_iter, radii_iter, Chi_iter, iter_time = modified_TRF_model(
    xk_initial=xk_initial,
    # surrogate_model=poly2ridge_Smodel,
    surrogate_model=surrogate,
    # surrogate_model=kriging_Smodel,
    rigorous_model=rigorous,
    # rigorous_model=mlp_Rmodel.predict,
    rigorous_placeholder=placeholder,
    blackbox_idx=blackbox_idx,
    iterations=100,
    TRFparams=TRFparams
)

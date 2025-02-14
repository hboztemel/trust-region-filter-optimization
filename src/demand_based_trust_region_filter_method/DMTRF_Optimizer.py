import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import time


np.set_printoptions(linewidth=120, suppress=True, formatter={'float_kind': '{:.3f}'.format})





def DMTRF_model(xk_initial, surrogate_model, rigorous_model, rigorous_placeholder, blackbox_idx,
                iterations=50, TRFparams=None):
    global rw_global, dw_global, Fθ, radii, Fθ_iter, radii_iter, trspk_iter, restart_count, max_restart

    xk_ = xk_initial.copy()
    graybox_idx = [i for i in range(len(xk_)) if i not in blackbox_idx]
    Fθ = []  # Initial filter set
    radii = []  # Radii setup
    Fθ_iter = []
    radii_iter = []
    trspk_iter = []
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

    best_point = {"xk": None, "fk": np.inf, "θk": np.inf, "Δ_": None}
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
        Δ_ = radius_validator.MinFeasRadiiFinder(xk_, Δ_, blackbox_idx)
        Δ_ = np.clip(Δ_, 0, 1e3)
        if len(Δ_) != len(blackbox_idx):
            Δ_ = Δ_[blackbox_idx]

        # Check termination condition
        if k > 0 and sk is not None:
            sk_blackbox = sk[blackbox_idx]
            if θk <= TRFparams['ϵθ'] and (np.abs(sk_blackbox) <= TRFparams['ϵs']).all():
                print("Convergence achieved.")
                super_print(Fθ_iter, radii_iter, ite=ite + 1)
                Fθ_iter.append([fk, θk])
                radii_iter.append([Δ_])
                stop_time = time.time()  # Stop timing the iteration
                iter_time.append(stop_time - start_time)
                break

        print(f"Current point: {xk_}")
        print(f"Trust Region Radius (Δ): {Δ_}")

        rw = surrogate_model.predict(xk_)
        print(f'Surrogate Model Output: {rw}')

        aspen_plus, sim_stat1 = rigorous_model(xk_)
        # aspen_plus = rigorous_model(xk_)
        # sim_stat1 = 1
        dw = aspen_plus[blackbox_idx]
        yk = aspen_plus[graybox_idx]
        print(f'Aspen Plus Simulation Output: {aspen_plus}')

        drw = np.zeros(len(xk_))
        drw[blackbox_idx] = rw
        drw[graybox_idx] = yk

        fk = obj_function(xk_, drw)
        θk = Infeasibility(rw, dw) if sim_stat1 else np.Inf

        if θk <= TRFparams["ϵθ"]:
            memory.append(xk_)

        if ite == 0:
            Fθ.append([fk, θk])
            radii.append([Δ_])
            Fθ_iter.append([fk, θk])
            radii_iter.append([Δ_])

        θmin = min(Fθ, key=lambda x: x[1])[1] if ite > 0 else 1e10
        fmin = min(Fθ, key=lambda x: x[0])[0] if ite > 0 else 1e10
        Δmin = Min.trfradius(radii)

        print(f"Objective Function (Profit, f): ${-fk:,.2f} at Step {ite + 1}.")
        print(f"Infeasibility (θ): {θk:.2f} at Step {ite + 1}.")

        # kth Trust Region Sub-Problem (TRSPk)
        print(f"Solving Trust Region Sub-Problem ({ite + 1})...")
        f_trspk, xk_trspk, sk, Δ_trspk, success = adaptive_TRSP_solver(xk_, Δ_,
                                                                       surrogate_model, rigorous_placeholder,
                                                                       blackbox_idx, obj_function)
        if success:
            print("TRSPk found a solution.")
            trspk_iter.append([sk])
            rw_trspk = surrogate_model.predict(xk_trspk)
            aspen_plus, sim_stat2 = rigorous_model(xk_trspk)
            # aspen_plus = rigorous_model(xk_trspk)
            # sim_stat2 = 1
            dw_trspk = aspen_plus[blackbox_idx]
            yk_trspk = aspen_plus[graybox_idx]
            θ_trspk = Infeasibility(rw_trspk, dw_trspk) if sim_stat2 else np.Inf
            Δ_trspk = radius_validator.MinFeasRadiiFinder(xk_trspk, Δ_trspk, blackbox_idx)

            print("Filter Conditions:")
            step_acceptance = False
            step_type = False

            fk, θk, xk_, Δ_, Fθ, radii, k = evaluate_filter_conditions(f_trspk, θ_trspk, fmin, θmin, fk,
                                                                       θk, xk_trspk, xk_, sk, Δ_trspk,
                                                                       Fθ, radii, TRFparams, k)

        else:
            # find_best_point(Fθ_iter, radii_iter)
            super_print(Fθ_iter, radii_iter, ite=ite + 1)
            print("Failure in TRSPk Solver. Restart Mechanism Starts...")
            fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                         surrogate_model, rigorous_placeholder,
                                                         Infeasibility,
                                                         radius_validator, blackbox_idx, graybox_idx,
                                                         TRFparams)

        Fθ_iter.append([fk, θk])
        radii_iter.append([Δ_])

        if ite == iterations:
            # find_best_point(Fθ_iter, radii_iter)
            super_print(Fθ_iter, radii_iter, ite=ite + 1)

        stop_time = time.time()  # Stop timing the iteration
        iter_time.append(stop_time - start_time)
        print(f"{ite + 1}. Iteration Time: {stop_time - start_time:2f} sec.")

    return Fθ, radii, Fθ_iter, radii_iter, trspk_iter, iter_time




# RUN
dataframe = pd.read_csv(
     r'.\MEOH_DOE.csv')
num_input = 9
X = dataframe.iloc[:, :num_input]
y = dataframe.iloc[:, num_input:]

xk_initial = np.array([38.834, 43.273, 19.026, 12.406, 9.901, 16.843, 26.561, 43.748, 1.583], dtype=float)

# Initialize the AspenPlusModel
meoh_path = r'.\RealSimulation'
MEOH = AspenPlusModel(meoh_path)
MEOH.initialize_aspen_plus()

TRFparams = {
    'κf': 0.5,   # Feasibility threshold scaling factor (unchanged)
    'κg': 0.5,   # Trust region scaling factor (unchanged)
    'γc': 0.6,   # Shrink factor (unchanged)
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




xk_initial = np.array([70,50,52,50,100,12.063,36.053,0.964,18.490], dtype=float)
blackbox_idx = [2, 3]  # CO and H2 (Syngas) Mass Flow Rates
graybox_idx = [i for i in range(len(xk_initial)) if i not in blackbox_idx]

# Pass the callable function directly to reduced_TRF_model

FθT, radiiT, Fθ_iter, radii_iter, Chi_iter, iter_time = modified_TRF_model(
    xk_initial=xk_initial,
    # surrogate_model=poly2ridge_Smodel,
    surrogate_model=poly2_Smodel,
    # surrogate_model=kriging_Smodel,
    rigorous_model=MEOH.run_aspen_plus,
    # rigorous_model=mlp_Rmodel.predict,
    rigorous_placeholder=rf_Rmodel,
    blackbox_idx=blackbox_idx,
    iterations=100,
    TRFparams=TRFparams
)


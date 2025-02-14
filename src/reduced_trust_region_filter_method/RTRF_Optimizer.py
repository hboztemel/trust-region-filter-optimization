import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import time


np.set_printoptions(linewidth=120, suppress=True, formatter={'float_kind': '{:.3f}'.format})



def reduced_TRF_model(xk_initial, surrogate_model, rigorous_model, rigorous_placeholder, blackbox_idx,
                      iterations=50, TRFparams=None):
    global rw_global, dw_global, Fθ, restart_count, max_restart

    xk_ = xk_initial.copy()
    graybox_idx = [i for i in range(len(xk_)) if i not in blackbox_idx]
    Fθ = []  # Initial filter set
    radii = []  # Radii setup
    Fθ_iter = []
    radii_iter = []
    Chi_iter = []
    memory = []
    iter_time = []
    arrays = []
    k = 0
    restoration_count = 0
    max_restoration = 5
    restart_count = 0
    max_restart = 8

    rw_global, dw_global = None, None

    radius_validator = KfullyLinearization(surrogate_model, rigorous_model, TRFparams['κf'], TRFparams['κg'])

    Δ = TRFparams['Δ'] if TRFparams and 'Δ' in TRFparams else 1.0
    Δ_ = xk_[blackbox_idx] * Δ

    Δ_ = radius_validator.MinFeasRadiiFinder(xk_, Δ_, blackbox_idx)

    best_point = {"xk": None, "fk": np.inf, "θk": np.inf, "Δ_": None}
    stagnation = 0

    for ite in range(iterations):
        start_time = time.time()
        print(f"\nIteration {ite + 1}")
        if np.any(np.isinf(xk_)) or np.any(np.isnan(xk_)):
            print("Invalid xk_ values detected. Terminating loop.")
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        xk_ = np.clip(xk_, 0, 1e3)
        Δ_ = np.clip(Δ_, 0, 1e3)
        print(f"xk_: {xk_}, Δ_: {Δ_}")

        rw = surrogate_model.predict(xk_)

        aspen_plus, sim_stat1 = rigorous_model(xk_)
        # if aspen_plus is None:
        #     raise RuntimeError("Aspen Plus simulation returned None.")
        dw = aspen_plus[blackbox_idx]
        yk = aspen_plus[graybox_idx]

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

        θmin = min(Fθ, key=lambda x: x[1])[1]
        fmin = min(Fθ, key=lambda x: x[0])[0]

        MeOH_Production = drw[6] / xk_[6]
        Carbon_Conversion = drw[6] / (xk_[4] - drw[4])
        H2_Conversion = (4 * drw[6] - 2 * drw[5]) / (xk_[2] - drw[2])
        print(f"Methanol Production Increase [%]: {MeOH_Production:.2f}% at Step {ite + 1}.")
        print(f"Carbon Conversion to Methanol [-]: {Carbon_Conversion:.2f}% at Step {ite + 1}.")
        print(f"Hydrogen Conversion to Methanol [-]: {H2_Conversion:.2f}% at Step {ite + 1}.")
        print(f"Infeasibility (θ): {θk:.2f} at Step {ite + 1}.")

        arrays.append([xk_, rw, dw, yk])

        print(f"Compatibility Check starts.")
        # Compatibility Check
        xk_minimizer, Δ_comp, β, compat_succ = compatibility_check(
            xk_, dw, surrogate_model, rigorous_placeholder, blackbox_idx, Δ_,
            TRFparams["κΔ"], TRFparams["κμ"], TRFparams["μ"])

        if compat_succ:
            print(f"Solver is successful.")
            # print(f"β (Compatibility result): {β}")

            if β < TRFparams["ϵc"]:
                print(f'TRSPk is compatible.')

                # Criticality Check
                print(f"Criticality Check starts.")
                xk_critical, Chi_k, crit_success = criticality_check(xk_, drw, obj_function, surrogate_model,
                                                                     blackbox_idx)
                Chi_iter.append(Chi_k)

                # print(f'xk_critical: {xk_critical}')
                # print(f'Chi_k: {Chi_k}')
                # print(f'TRFparams[ξ] * Δ_ / xk_[blackbox_idx]: {TRFparams["ξ"] * Δ_ / xk_[blackbox_idx]}')
                if crit_success:
                    print(f"Solver is successful.")
                    print(f"Chi_k: {Chi_k}")
                    print(f"Δ_ / xk_[blackbox_idx]: {Δ_ / xk_[blackbox_idx]}")

                    if (Chi_k < TRFparams['ξ'] * Δ_ / xk_[blackbox_idx]).all():
                        Δ_ *= TRFparams['ω']
                        print("The TRSPk is in critical region. Trust region shrinkage.")
                        is_critical = True
                        Fθ_iter.append([fk, θk])
                        radii_iter.append([Δ_])
                        xk_blackbox = xk_[blackbox_idx]
                        xk_[blackbox_idx] = np.random.uniform(low=xk_blackbox - Δ_, high=xk_blackbox + Δ_,
                                                              size=xk_blackbox.shape)
                        print("Input array is perturbated for next iteration.")
                        continue
                    else:
                        print("TRSPk is not in a critical region.")

                    # kth Trust Region Sub-Problem (TRSPk)
                    print(f"Solving Trust Region Sub-Problem ({ite + 1})...")
                    f_trspk, xk_trspk, sk, Δ_trspk, success = TRSP_solver(xk_, Δ_comp,
                                                                          surrogate_model,
                                                                          rigorous_placeholder,
                                                                          blackbox_idx, obj_function)
                    if success:
                        print("TRSPk found a solution.")
                        rw_trspk = surrogate_model.predict(xk_trspk)
                        print(f'Surrogate Model Output: {rw_trspk}')

                        aspen_plus_trspk, sim_stat2 = rigorous_model(xk_trspk)
                        dw_trspk = aspen_plus_trspk[blackbox_idx]
                        yk_trspk = aspen_plus_trspk[graybox_idx]
                        print(f'Aspen Plus Simulation Output: {aspen_plus_trspk}')

                        θ_trspk = Infeasibility(rw_trspk, dw_trspk) if sim_stat2 else np.Inf

                        print("Filter Conditions:")
                        step_acceptance = False
                        step_type = False

                        fk, θk, xk_, Δ_, Fθ, radii, k, step_acceptance = evaluate_filter_conditions(f_trspk, θ_trspk,
                                                                                                    fmin, θmin, fk,
                                                                                                    θk, xk_trspk, xk_,
                                                                                                    sk, Δ_trspk,
                                                                                                    Fθ, radii,
                                                                                                    TRFparams, k)

                        if step_acceptance:
                            rw = rw_trspk
                            dw = dw_trspk
                            yk = yk_trspk

                        if ite == iterations:
                            super_print(Fθ_iter, radii_iter, ite=ite + 1)
                    else:
                        super_print(Fθ_iter, radii_iter, ite=ite + 1)
                        print("Failure in TRSPk Solver.")
                        fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                     surrogate_model, rigorous_placeholder,
                                                                     Infeasibility,
                                                                     blackbox_idx, graybox_idx, TRFparams["ϵθ"])
                else:
                    fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                 surrogate_model, rigorous_placeholder, Infeasibility,
                                                                 blackbox_idx, graybox_idx, TRFparams["ϵθ"])
            else:
                print("TRSPk is not compatible.")
                Fθ.append([fk, θk])
                radii.append([Δ_])

                if restoration_count >= max_restoration * 2:
                    print(
                        f"Restoration excessively called {restoration_count}. Feasible region is small, hence loop terminated.")
                    Fθ_iter.append([fk, θk])
                    radii_iter.append([Δ_])
                    super_print(Fθ_iter, radii_iter, ite=ite + 1)
                    stop_time = time.time()  # Stop timing the iteration
                    iter_time.append(stop_time - start_time)
                    break
                elif restoration_count >= max_restoration:
                    print("Too many restoration warning: Enhancing trust region and infeasibility threshold.")
                    TRFparams['εθ'] *= 1.3  # Increase infeasibility threshold
                    Δ_ *= 2  # Expand trust region
                else:
                    # Restoration Procedure if not compatible
                    print("Restoration procedure is called.")
                    restoration_count += 1
                    (restored_point, Δ_restored, f_restored,
                     θ_restored, is_feasible) = restoration(xk_, Δ_, surrogate_model, rigorous_placeholder, θk,
                                                            fk, blackbox_idx, TRFparams, Fθ, radii)
                    if is_feasible:  # Added to the filter set
                        xk_ = restored_point
                        fk = f_restored
                        θk = θ_restored
                        print(f"Restoration successful. Restored xk_: {xk_} for next iteration.")
                        k += 1
                    else:
                        print("Failure in Restoration.")
                        fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                                     surrogate_model, rigorous_placeholder,
                                                                     Infeasibility,
                                                                     blackbox_idx, graybox_idx, TRFparams["ϵθ"])
        else:
            print("Compatibility solver is failed.")
            fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                         surrogate_model, rigorous_placeholder, Infeasibility,
                                                         blackbox_idx, graybox_idx, TRFparams["ϵθ"])

        Fθ_iter.append([fk, θk])
        radii_iter.append([Δ_])

        is_stagnant, stagnation = check_stagnation(Fθ_iter, fk, θk, TRFparams, stagnation)
        if is_stagnant:
            # print("Stagnation detected.")
            fk, θk, xk_, Δ_ = adaptive_restart_mechanism(xk_, drw, Δ_, memory, fin_diff, obj_function,
                                                         surrogate_model, rigorous_placeholder, Infeasibility,
                                                         blackbox_idx, graybox_idx, TRFparams["ϵθ"])

        # print(f"fk: {fk}, θk: {θk}, Δ_: {Δ_}")
        if fk > 1e5 or θk > 1e5:
            print(f"Terminating the loop.")
            super_print(Fθ_iter, radii_iter, ite=ite + 1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        if np.any(Δ_ <= 1e-2):
            print(f"Convergence achieved: Δ has approached zero. Terminating loop.")
            super_print(Fθ_iter, radii_iter, ite=ite + 1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        if ite > 20 and abs(fk - Fθ_iter[-1][0]) < TRFparams['ϵf'] and abs(θk - Fθ_iter[-1][1]) < TRFparams['ϵθ']:
            print("Convergence achieved. Terminating loop.")
            super_print(Fθ_iter, radii_iter, ite=ite + 1)
            stop_time = time.time()  # Stop timing the iteration
            iter_time.append(stop_time - start_time)
            break

        # print(f"Chi_k: {Chi_k}")
        # if ite >= int(0.15 * iterations):
        #     if crit_success and (Chi_k <= TRFparams['ϵχ']).all():
        #         print("Criticality conditions met. Terminating loop.")
        #         break

        if ite == iterations:
            super_print(Fθ_iter, radii_iter, ite=ite + 1)

        stop_time = time.time()  # Stop timing the iteration
        iter_time.append(stop_time - start_time)
        print(f"{ite + 1}. Iteration Time: {stop_time - start_time:2f} sec.")

    return Fθ, radii, Fθ_iter, radii_iter, iter_time, arrays





#RUN
dataframe = pd.read_csv(
     r'..\MEOH_DOE.csv')

num_input = 9
X = dataframe.iloc[:, :num_input]
y = dataframe.iloc[:, num_input:]


# Initialize the AspenPlusModel
meoh_path = r'.\aspen_case'
MEOH = AspenPlusModel(meoh_path)
MEOH.initialize_aspen_plus()




TRFparams = {
    'κf': 0.5,   # Feasibility threshold scaling factor (unchanged)
    'κg': 0.5,   # Trust region scaling factor (unchanged)
    'γc': 0.8,   # Shrink factor (more aggressive)
    'γe': 1.5,   # Expand factor (faster expansion)
    'γf': 0.5,   # f-criterion reduction factor (unchanged)
    'γθ': 0.2,   # θ-criterion reduction factor (unchanged)
    'κθ': 1.2,   # Feasibility threshold scaling factor (unchanged)
    'κΔ': 0.7,   # Trust region scaling factor (more flexibility)
    'κμ': 0.2,   # Criticality threshold scaling factor (adjusted for compatibility focus)
    'μ': 0.2,    # Trade-off parameter (emphasize compatibility)
    'γs': 0.9,   # Step size scaling factor (unchanged)
    'ξ': 0.5,   # Criticality boundary scale parameter (unchanged)
    'ω': 0.2,    # Criticality boundary scale parameter (unchanged)
    'η1': 0.2,   # Quality ratio shrink factor (unchanged)
    'η2': 0.3,   # Quality ratio expand factor (unchanged)
    'Niter': 200,  # Maximum number of iterations (unchanged)
    'Δ': 1.5,    # Trust region size (reduced for stability)
    'ϵc': 4,     # Compatibility tolerance (adjusted for better balance)
    'ϵχ': 5e-2,  # Tightened criticality termination
    'ϵθ': 5e-3,  # Tightened infeasibility threshold
    'ϵf': 1e-6,  # Tightened optimality tolerance
    'ϵσ': 5e-2,  # Reduced sampling region termination tolerance
    'ϵΔ': 5e-2,  # Reduced termination tolerance
    'Ψ': 0.6,    # Sampling region reset parameter (unchanged)
}





blackbox_idx = [2, 3, 4]

poly2_Smodel = ReducedModels()
poly2_Smodel.train(X, y, model_type='poly_2', blackbox_idx=blackbox_idx)
cv_results = poly2_Smodel.cross_validate(X, y, model_type='poly_2', blackbox_idx=blackbox_idx)
print(cv_results)




xk_initial = np.array([52.7778,79.2687,200,100,100,12.063,36.053,0.964,1], dtype=float)
# xk_initial = np.array([80,79.2687,150,150,100,12.063,36.053,0.964,18.490], dtype=float)
graybox_idx = [i for i in range(len(xk_initial)) if i not in blackbox_idx]

# Define perturbation limits
pert_min = np.array([60, 70, 50, 50, 50, 0.00001, 0.00001, 0.00001, 0.00001])
pert_max = np.array([280, 100, 300, 200, 200, 1, 1, 1, 1])
perturbed_xk = xk_initial + np.random.uniform(-1, 1, size=xk_initial.shape) * (pert_max - pert_min) * 0.1  # 10% perturbation
perturbed_xk = np.clip(perturbed_xk, pert_min, pert_max)  # Ensure within bounds

surrogate = poly2_Smodel
rigorous=MEOH.run_aspen_plus
placeholder = placeholder

# Pass the callable function directly to reduced_TRF_model
Fθ, radii, Fθ_iter, radii_iter, iter_time, arrays = reduced_TRF_model(
    xk_initial=perturbed_xk,
    surrogate_model=surrogate,
    rigorous_model=rigorous,
    rigorous_placeholder=placeholder,
    blackbox_idx=blackbox_idx,
    iterations=50,
    TRFparams=TRFparams
)

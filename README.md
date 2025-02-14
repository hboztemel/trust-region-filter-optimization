
# Trust Region Filter (TRF) Optimization Framework

This repository contains the implementation of the Trust Region Filter (TRF) optimization framework for 1) Reduced Trust Region Filter and 1) Modified Trust Region Filter, focusing on adaptive mechanisms, Aspen Plus simulation integrations, and reduced models. The repository is structured for modularity and scalability, with clear separation of functionality into relevant folders [1][2].

---

## **Folder Structure**

### 1. `aspen_plus_simulation/`
This folder contains resources and backups related to the Aspen Plus simulations used in the TRF optimization process.

- **`MEOH_synthesis_plant.bkp`**: Backup file of the methanol synthesis plant simulation.

---

### 2. `automations/`
Contains scripts to automate key processes, such as generating samples, calculating coefficients, and integrating with Aspen Plus.

- **`FORTRAN_calculator.py`**: Automates the creation of FORTRAN-based models for Aspen Plus integration.
- **`poly2_beta_coeff_calculator.py`**: Computes the beta coefficients for second-order polynomial regression.
- **`sampling_generator.py`**: Generates sampling points for use in optimization and sensitivity analysis.

---

### 3. `src/`
The main source folder containing all core algorithms and models.

#### a) `algorithms/`
Contains implementations of the trust region filter algorithm and its variants.

##### i) `reduced_trust_region_filter/`
Implements the **Reduced Trust Region Filter (RTRF) Method** with supportting functionalities. Eason and Biegler (2016) introduced the reduced method in the literature [1].

##### ii) `modified_trust_region_filter/`
Implements the **Modified Trust Region Filter (MTRF) Method** with various adaptive mechanisms and supporting functionalities. Eason and Biegler (2018) introduced the modified method in the literature [2].

##### iii) `demandbased_trust_region_filter/`
Implements the **Demand-Based Trust Region Filter (DMTRF)** with simplified framework and supporting functionalities. Yoshio and Biegler (2021) introduced the demand-based method in an industrial scale cholorobenzene process [3].

---

#### b) `applications/`
Contains Jupyter notebooks for applying the TRF framework with Aspen Plus simulations.

- **`RTRF_Optimizer.py`**: Demonstrates the reduced TRF implementation with Aspen Plus integration.
- **`MTRF_Optimizer.py`**: Demonstrates the modified TRF implementation with Aspen Plus integration.
- **`DMTRF_Optimizer.py`**: Demonstrates the demand-based TRF implementation with Aspen Plus integration.

---

#### c) Surrogate-Based Optimization: `SurrogateModels.py/`
Contains surrogate models and regression implementations used in the TRF framework.

- **`LR`**: Linear regression.
- **`2PR`**: Second-order polynomial regression.
- **`2PRRIDGE`**: Ridge regression using second-order polynomial features.
- - **`KRIGING`**: Kriging model for surrogate-based optimization.

---

## **How to Use**

1. **Setup**: Clone the repository and install the required dependencies from `requirements.txt` **(TBD!)**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Aspen Plus Integration**: Use files from `automations/` and `aspen_plus_simulation/` to set up Aspen Plus simulation models.

3. **Run Algorithms**:
   - For **Modified TRF**, use files from `src/algorithms/modified_trust_region_filter/`.
   - For **Reduced TRF**, use files from `src/algorithms/reduced_trust_region_filter/`.

4. **Apply Surrogate Models**: Implement surrogate models from the `reduced_models/` folder as needed.

5. **Analyze Results**: Use notebooks in the `applications/` folder to analyze optimization results.

---

## References

[1] Eason, J. P., & Biegler, L. T. (2016). A trust region filter method for glass box/black box optimization. *AIChE Journal, 62*(9), 3124–3136. (https://doi.org/10.1002/aic.15325)

[2] Eason, J. P., & Biegler, L. T. (2018). Advanced trust region optimization strategies for glass box/black box models. *AIChE Journal, 64*(11), 3934–3943. (https://doi.org/10.1002/aic.16364)

[3] Yoshio, N., & Biegler, L. T. (2021). Demand-based optimization of a chlorobenzene process with high-fidelity and surrogate reactor models under trust region strategies}. *AIChE Journal, 67*(1), e17054. (https://doi.org/10.1002/aic.17054)

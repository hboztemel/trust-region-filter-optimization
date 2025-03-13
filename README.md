# Trust Region Filter (TRF) Optimization Framework

This repository is part of a Master's degree thesis project in the Chemical Engineering Department of Politecnico di Milano. 

**Project:** Comparative analysis of trust region filter methods with different mathematical models in surrogate-based optimization
**Author:** Hikmet Batuhan Oztemel
**Supervisor:** Prof. Flavio Manenti

## Project Overview

Process optimization is a crucial discipline for increasing production efficiency and reducing operational costs. This project focuses on applying Trust Region Filter (TRF) methods in surrogate-based optimization (SBO) for chemical process optimization. The study evaluates different TRF algorithm variations, including Reduced TRF, Modified TRF, and Demand-Based TRF, in an Aspen Plus simulation of a methanol synthesis plant. These methodologies help improve computational efficiency and optimization performance while integrating complex surrogate models.

---

## **Repository Structure**

### 1. `Thesis_Polimi/`
Contains the executive summary report of the thesis.
- **`2025_03_Oztemel_ExecutiveSummary.pdf`**: Summary report of the thesis project.

---

### 2. `src/`
The main source folder containing Aspen Plus simulations, TRF algorithm implementations, and supporting files.

#### a) `demand_based_trust_region_filter_method/`
Contains the implementation of the **Demand-Based Trust Region Filter (DMTRF)**.

- **`DMTRF_Functions.py`**: Core functions used in the DMTRF algorithm.
- **`DMTRF_Optimizer.py`**: Optimization script for running DMTRF.
- **`ProcessFunctions.py`**: General process-related functions.
- **`SurrogateModels.py`**: Surrogate model implementations for DMTRF.

---

#### b) `modified_trust_region_filter_method/`
Contains the implementation of the **Modified Trust Region Filter (MTRF)**.

- **`MTRF_Functions.py`**: Core functions used in the MTRF algorithm.
- **`MTRF_Optimizer.py`**: Optimization script for running MTRF.
- **`ProcessFunctions.py`**: General process-related functions.
- **`SurrogateModels.py`**: Surrogate model implementations for MTRF.

---

#### c) `reduced_trust_region_filter_method/`
Contains the implementation of the **Reduced Trust Region Filter (RTRF)**.

- **`RTRF_Functions.py`**: Core functions used in the RTRF algorithm.
- **`RTRF_Optimizer.py`**: Optimization script for running RTRF.
- **`ProcessFunctions.py`**: General process-related functions.
- **`SurrogateModels.py`**: Surrogate model implementations for RTRF.

---

#### d) `aspen_plus_simulation/`
Contains resources related to the Aspen Plus simulation used in the TRF optimization process. However, simulation files are not included in this repository due to licensing and copyright restrictions, as they are proprietary to AspenTech and used under a license granted to Politecnico di Milano.

- **`AspenPlusModel.py`**: Python script for interfacing with Aspen Plus.
- **`MEOH_synthesis_plant.bkp`**: Backup file of the methanol synthesis plant simulation.

---

## **How to Use**

1. **Aspen Plus Integration**: Use files from `aspen_plus_simulation/` to set up Aspen Plus simulation models.

2. **Run Algorithms**:
   - For **Demand-Based TRF**, use files from `src/demand_based_trust_region_filter_method/`.
   - For **Modified TRF**, use files from `src/modified_trust_region_filter_method/`.
   - For **Reduced TRF**, use files from `src/reduced_trust_region_filter_method/`.

4. **Apply Surrogate Models**: Implement surrogate models from `SurrogateModels.py` in each algorithm folder as needed.

5. **Analyze Results**: Run optimization scripts (`*_Optimizer.py`) to analyze results for each method.

---

## References

[1] Bozzano, G., & Manenti F. (2016). Efficient methanol synthesis: Perspectives, technologies and optimization strategies. *Prog. Energy Combust. Sci., 56*, 71–105.

[2] Eason, J. P., & Biegler, L. T. (2016). A trust region filter method for glass box/black box optimization. *AIChE Journal, 62*(9), 3124–3136.

[3] Eason, J. P., & Biegler, L. T. (2018). Advanced trust region optimization strategies for glass box/black box models. *AIChE Journal, 64*(11), 3934–3943.

[4] Yoshio, N., & Biegler, L. T. (2021). Demand-based optimization of a chlorobenzene process with high-fidelity and surrogate reactor models under trust region strategies. *AIChE Journal, 67*(1), e17054.


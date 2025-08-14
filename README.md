![HUMID TOC](https://jinjaehyeok.wordpress.com/wp-content/uploads/2025/08/humid_toc-1.jpg)

# HUMID: Hydronium Ultra-coarse-grained Model with Improved Dynamics

*A reactive bottom-up coarse-grained model for simulating hydrated proton transport with near-atomistic accuracy and a computational speed-up of three orders of magnitude.*

---

## 📜 Overview
Proton transport is a fundamental process in chemistry, biology, and materials science. However, simulating this phenomenon at large scales is computationally expensive due to its reactive, quantum-mechanical nature. This project introduces **HUMID** (Hydronium Ultra-coarse-grained Model with Improved Dynamics), a novel bottom-up coarse-grained (CG) model that bridges the gap between quantum mechanics and mesoscopic simulations.

HUMID uses an internal state-based approach, analogous to the Multiscale Empirical Valence Bond (MS-EVB) method, to capture the reactive proton hopping (*Grotthuss mechanism*). By representing molecules as CG sites with distinct *hydronium-like* and *water-like* internal states, the model can simulate chemical identity changes on-the-fly. This approach allows for the accurate reproduction of structural and dynamical properties, including accelerated proton diffusion, at a fraction of the computational cost of traditional atomistic simulations.

This repository contains the source codes, simulation inputs, analysis scripts, and results associated with the development and validation of the HUMID model.  

For the full MD trajectory files, please refer to the Zenodo repository  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16863477.svg)](https://doi.org/10.5281/zenodo.16863477)

---

## ✨ Key Features
- **Reactive Coarse-Graining**: The first bottom-up CG model to successfully simulate reactive proton transport by incorporating internal states.
- **High Fidelity**: Faithfully reproduces key structural correlations (RDFs, angular distributions) and complex dynamical properties (anomalous diffusion, Grotthuss shuttling rates).
- **Computational Efficiency**: Achieves up to a three-order-of-magnitude speed-up compared to reactive atomistic models like MS-EVB.
- **Transferability**: Designed to be extensible to more complex systems, including multi-proton solutions and nanoconfined environments like carbon nanotubes.
- **Systematic Design**: Provides a rigorous statistical mechanical framework and design principles for building other reactive CG models.

---

## 💻 Usage Guide

### Step 1: Compilation ⚙️
The HUMID model requires a custom version of **LAMMPS** that includes specialized pair styles for handling internal states. Both the custom LAMMPS for HUMID and the MS-EVB version (RAPTOR) are provided.

**Navigate to the HUMID LAMMPS Source**  
The code for the HUMID model's internal-state pair style is in: LAMMPS Codes/LAMMPS Custom Pair Style for HUMID/src/


**Core files:**
- `pair_table_ucg_hydronium.cpp/.h` – Implements the tabulated potentials and state-switching logic based on local coordination number.
- `pair_msucg_neigh.cpp/.h` – Neighbor list logic for internal state calculations.

**Compile LAMMPS:**
cd "LAMMPS Codes/LAMMPS Custom Pair Style for HUMID/src/"
./make-midway2.sh

This creates `lmp_midway1`, the executable for running coarse-grained simulations.  
(For reference atomistic simulations, compile the code in `LAMMPS Codes/LAMMPS MS-EVB RAPTOR/` similarly.)

---

### Step 2: Running Simulations ▶️

#### A. Single Proton (Primary Study)
**Location:**  Simulation Input/HUMID/single-aqueous/

**Key Files:**
- `in.mscg` – Main LAMMPS input script.
- `data.mscg` – Initial coordinates/topology for 256 particles.
- `fm.table` – Tabulated potentials for all state-wise interactions.
- `state_setting.txt` – Defines hydronium-like character using a sigmoidal function.

**Run Command:** mpirun -np 8 ./lmp_midway1 < path/to/Simulation\ Input/HUMID/single-aqueous/in.mscg

---

#### B. Transferable Systems
- **Multi-Proton Systems:**  
  `Simulation Input/HUMID/multiproton/` – Subdirectories (1, 2, 4, 8 protons) with adjusted `state_setting.txt` files.

- **Carbon Nanotube (CNT) System:**  
  `Simulation Input/HUMID/carbon-nanotube/` – Adds water/hydronium–carbon interactions.

---

### Step 3: Analysis 📊

**Location:** Codes/Analysis Codes/  
Organized by analysis type, with most scripts as compiled C++ executables.

#### Structural Correlations (RDFs & Angles)
- **Locations:** `RDF Analysis/`, `Angle Analysis/`
- **Executables:** `ucg_rdf.x`, `hydronium.x`

#### Dynamical Properties
- **MSD Analysis:** `MSD Analysis/`
  - Requires **Hydronium Mapping** (`Hydronium Mapping/`) via `new.x`.

- **Correlation Functions:** `Correlation Analysis/`
  - `forward hop/print_type2.x` – Grotthuss shuttling rate.
  - `population/continuous.x` – Hydronium population correlation function.

#### Deriving Dynamic Friction
- **Location:** `CG Langevin Equation/`
- **Executables:**  
  - `cfv.x` – Force-velocity correlation.  
  - `cvv.x` – Velocity autocorrelation.  

Final friction coefficient `η` ensures CG dynamics match atomistic timescales.

---

## 🗂️ Repository Structure: Refer to directory file for the detailed tree
```text
├── Codes/
│   ├── Analysis Codes/             # Scripts for analyzing simulation output
│   ├── Coarse-graining/            # Scripts for the CG mapping process
│   └── Parametrization/            # Scripts and data for parametrizing the HUMID model
├── Figure Data/                    # Raw data used to generate figures in the publication
├── Figures/                        # Final figures from the publication
├── LAMMPS Codes/
│   ├── LAMMPS Custom Pair Style for HUMID/ # Source code for the custom LAMMPS build
│   └── LAMMPS MS-EVB RAPTOR/       # Source code for the atomistic MS-EVB simulations
├── README.md                       # This file
├── Results/                        # Processed results from simulations
├── Simulation Input/               # LAMMPS input scripts for running simulations
└── Videos/                         # Movies visualizing simulation trajectories
```

---

## 📈 Results

The key findings of this work are summarized in the `Results/` directory, which contains processed data for:

* **Structure** – RDFs and angular distributions confirming that HUMID captures the distinct local structure of the hydrated proton.
* **Dynamics** – MSD plots showing the correct accelerated diffusion of hydronium and correlation functions quantifying the Grotthuss shuttling mechanism.
* **Interactions** – Optimized state-wise potential energy curves derived from the bottom-up procedure.

---

## 🎓 Citation

If you use this model or any part of the code in your research, please cite our paper:

Jin, J., Li, Z., & Voth, G. A. (2025). *Systematic Bottom-Up Coarse-Graining of Hydrated Excess Proton Transport Across Scales*. (Preprint).

```latex
@article{jin2025humid,
  title={Systematic Bottom-Up Coarse-Graining of Hydrated Excess Proton Transport Across Scales},
  author={Jin, Jaehyeok and Li, Zhefu and Voth, Gregory A.},
  journal={arXiv},
  year={2025},
  publisher={Will be Updated Soon}
}


# Graph-Guided Sparse Learning via Boolean Relaxation

##  Overview
This repository contains the code of implementation of experiments in paper **Graph-Guided Sparse Learning via Boolean Relaxation**, a method designed to enhance feature selection by incorporating **Boolean relaxation** and **graph-guided sparsity constraints**. 

## Repository Structure

```bash
Graph-Guided-Sparse-Learning/  (PROJECT ROOT)
├── random_ensemble.py         # Core class for your experiments
├── solver.py                  # Main solver class, dispatches to specific solvers
├── example.ipynb              # Demo notebook provides tutorial on how to reproduce the experimental results from the paper
│
├── solvers/                   # Python implementations of various algorithms
│   ├── __init__.py
│   ├── signal_family.py       # Python wrapper for Signal Family method
│   ├── gfl_pqn.py             # Python wrapper for MATLAB PQN code
│   ├── gfl_proximal.py        # Python wrapper for MATLAB proximal code
│   └── ... (other Python solvers)
│
├── src/                       # Contains C/MATLAB code and compiled modules
│   ├── sparse_module.so       # Compiled C extension (not included in repo)
│   ├── setup.py               # Build sparse_module.so for Linux environments
│   │
│   ├── algo_wrapper/          # Python/C code for signal family methods
│   │   └── c/                 # C source files
│   │
│   ├── PQN/                   # MATLAB library for Projected Quasi-Newton
│   │   └── gfl_pqn.m          # The MATLAB function being called
│   │   └── ... (many other .m, .mexmaca64 files)
│   │
│   └── code_fgfl_aaai14/      # MATLAB library (from AAAI14 paper)
│       └── GFL/               # Contains mexEFLSA, paragcut etc.
│       └── gfl_proximal.m     # The MATLAB function for proximal GFL
│       └── ... (many other .m, .mex files)
│
├── utils/                     # Utility Python modules
│   ├── __init__.py
│   ├── graph.py
│   ├── omse.py
│   └── ...
│
└── configs/                   # Configuration files (e.g., conda environments)
    ├── environment.yml
    └── ...
```

### Set Up Virtual Environment
Using Conda:
```bash
conda env create -f environment.yml
conda activate gfl
```

Some user may have the following error, 
```
PackagesNotFoundError: The following packages are not available from current channels:
  - gurobi
```
To resolve this, add the Gurobi channel explicitly before creating the env and ensure you have a valid academic or commercial license for Gurobi:

```bash
conda config --add channels gurobi
```

Run the `example.ipynb` notebook to see how to reproduce the experimental results from the paper. The notebook provides a step-by-step guide to run the experiments in the paper.


---

## Methods
We include implementations of the following methods in our experiments:
| Method                  | Description |
|-------------------------|-------------|
| **Lasso**               | Standard ℓ₁-regularized regression |
| **Adaptive Grace**      | [Variable selection and regression analysis for graph-structured covariates with an application to genomics](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-3/Variable-selection-and-regression-analysis-for-graph-structured-covariates-with/10.1214/10-AOAS332.full) |
| **Fast GFL (AAAI-2014)**| [Efficient Generalized Fused Lasso and Its Applications](https://dl.acm.org/doi/10.1145/2847421) |
| **Boolean Lasso**       | [Sparse learning via Boolean relaxations](https://link.springer.com/article/10.1007/s10107-015-0894-1) |
| **Boolean GFL (Ours)**  | Our proposed Graph-Guided Boolean Relaxation method |
| **Signal Family**      | [Graph-induced Constraint Method](https://proceedings.mlr.press/v162/zhou22i.html) |


The original code for **Adaptive Grace** is **not publicly available**, so we implemented them in **Python**.

For **Fast GFL**, we obtained the original implementation from this [link](https://www.tandfonline.com/doi/suppl/10.1080/10618600.2015.1114491?scroll=top).

Our method (**Boolean GFL**) and **Boolean Lasso** were implemented in **MATLAB** because we rely on the [Projected Quasi-Newton (PQN) method](https://www.cs.ubc.ca/~schmidtm/Software/PQN.html) to solve the optimization problem.

For **Signal Family**, we adapted and modified code from the original authors, which is available [here](https://github.com/baojian/dmo-fw). Note that you need to compile the C code in the `algo_wrapper/c` directory. See the README in that folder for OS-specific compilation instructions.
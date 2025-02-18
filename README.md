# Graph-Guided Sparse Learning via Boolean Relaxation

##  Overview
This repository contains the implementation of **Graph-Guided Sparse Learning via Boolean Relaxation**, a method designed to enhance feature selection by incorporating **Boolean relaxation** and **graph-guided sparsity constraints**. Our approach enables more interpretable and structured sparsity modeling for high-dimensional datasets.

##  Features
- **Structured Sparse Learning**: Utilizes prior graph knowledge to improve feature selection.
- **Boolean Relaxation**: Novel convex relaxation technique to enhance sparsity modeling.
- **Multiple Optimization Methods**: Includes Lasso, Adaptive Grace, Fast GFL, Boolean Lasso, and our proposed Boolean GFL.
- **Reproducible Experiments**: Scripts to generate datasets, run optimizations, and evaluate performance.

---

## 📂 Repository Structure

```
graph-guided-sparse-learning
│── Experiment.ipynb            # Jupyter notebook for running experiments
│── environment.yml              # Conda environment setup
│── README.md                    # Project documentation
│
│── PQN                       # Projected Quasi-Newton optimization 
│
│── code_fgfl_aaai14           # Fast GFL method (AAAI-2014) implementation
│   ├── GFL/                      # Graph-based Lasso utilities
│   ├── data_gfl/                 # Data storage for GFL
│   ├── result_gfl/               # Results from GFL experiments
│   ├── fast_gfl.m                # Fast GFL MATLAB implementation
│   ├── gfl_proximal.m            # Proximal-based GFL implementation
│   └── comparison_convergence.m  # Convergence analysis
|
├── Related Works.md          # Summary of related work
```


### Set Up Virtual Environment
Using Conda:
```bash
conda env create -f environment.yml
conda activate gfl
```

Run the `Experiment.ipynb` notebook to create datasets for training and evaluation.

Run the `inference.ipynb` notebook to run the inference on your data. We also include an example with small dataset in this notebook to display how to run the inference on a dataset.

---

## 🔬 Optimization Methods Implemented

| Method                  | Description |
|-------------------------|-------------|
| **Lasso**               | Standard ℓ₁-regularized regression |
| **Adaptive Grace**      | [Variable selection and regression analysis for graph-structured covariates with an application to genomics](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-3/Variable-selection-and-regression-analysis-for-graph-structured-covariates-with/10.1214/10-AOAS332.full) |
| **Fast GFL (AAAI-2014)**| [Efficient Generalized Fused Lasso and Its Applications](https://dl.acm.org/doi/10.1145/2847421) |
| **Boolean Lasso**       | [Sparse learning via Boolean relaxations](https://link.springer.com/article/10.1007/s10107-015-0894-1) |
| **Boolean GFL (Ours)**  | Our proposed Graph-Guided Boolean Relaxation method |


The **original code** for some methods (e.g., **Adaptive Grace**) is **not publicly available**, so we implemented them in **Python**.

For **Fast GFL**, we obtained the original implementation from this [link](https://www.tandfonline.com/doi/suppl/10.1080/10618600.2015.1114491?scroll=top).

Our method (**Boolean GFL**) and **Boolean Lasso** were implemented in **MATLAB** because we rely on the [Projected Quasi-Newton (PQN) method](https://www.cs.ubc.ca/~schmidtm/Software/PQN.html) to solve the optimization problem.

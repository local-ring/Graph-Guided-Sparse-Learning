import numpy as np
from gurobipy import Model, GRB, quicksum

def ProjOperator_Xs(vec, n, k):
    """
    Projects the upper triangular part of a matrix represented by 'vec'
    onto the constraint set:
    - sum(y_ii) <= k
    - 0 < y_ij < 1 for all elements
    
    Parameters:
    - vec: (n * (n + 1)) / 2 numpy array, upper triangular part of the matrix.
    - n: Integer, size of the matrix (n x n).
    - k: Integer, maximum sum of diagonal elements.
    
    Returns:
    - vec_proj: Projected upper triangular part as a flattened vector.
    """
    # Reconstruct the full matrix from the upper triangular vector
    Y = np.zeros((n, n))
    indices = np.triu_indices(n)
    # print("Y.shape", Y.shape, "indices.shape", indices[0].shape, "vec.shape", vec.shape)
    Y[indices] = vec.flatten()
    Y = Y + Y.T - np.diag(np.diag(Y))  # Fill lower triangular to make symmetric

    # Gurobi optimization
    model = Model("Projection")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    # Define variables for upper triangular part
    y = model.addVars(n, n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")

    # Objective: Minimize ||Y - y||_F^2
    model.setObjective(
        quicksum((Y[i, j] - y[i, j]) ** 2 for i in range(n) for j in range(i, n)),
        GRB.MINIMIZE
    )

    # Constraint 1: sum(y_ii) <= k
    model.addConstr(quicksum(y[i, i] for i in range(n)) <= k, name="DiagonalSum")
    model.addConstr(quicksum(y[i, i] for i in range(n)) >= k-1, name="DiagonalSum")

    # Optimize the model
    model.optimize()

    # Extract the projected matrix
    Y_proj = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            Y_proj[i, j] = y[i, j].x
            Y_proj[j, i] = Y_proj[i, j]  # Symmetric

    # Extract the upper triangular part of the projected matrix
    vec_proj = Y_proj[indices]

    return vec_proj

from gurobipy import Model, Env
import numpy as np


def construct_difference_matrix(m, n):
    """
    Construct the difference matrix D for variables indexed by (i, j) where
    there are m rows and n columns.

    Args:
        m: Number of rows (i index). here it is the d, number of features
        n: Number of columns (j index). here it is the h, number of clusters

    Returns:
        D: A sparse matrix of shape ((m * (n-1)) x (m * n)) representing the
           pairwise differences (x_{i,j} - x_{i,j-1}).
    """
    from scipy.sparse import lil_matrix

    # Number of rows in D is m * (n-1) (differences per row)
    # Number of columns in D is m * n (variables total)
    num_rows = m * (n - 1)
    num_cols = m * n

    # Initialize the sparse matrix
    D = lil_matrix((num_rows, num_cols))

    # Fill the matrix with +1 and -1 for each difference x_{i,j} - x_{i,j-1}
    for i in range(m):
        for j in range(1, n):
            row_idx = i * (n - 1) + (j - 1)  # Row in D
            col_plus = i * n + j            # x_{i,j} position (+1)
            col_minus = i * n + (j - 1)     # x_{i,j-1} position (-1)

            D[row_idx, col_plus] = 1
            D[row_idx, col_minus] = -1

    return D.tocsr()  # Convert to CSR format for efficient use in solvers

from gurobipy import Model, GRB, LinExpr
import numpy as np

from gurobipy import Model, GRB, LinExpr
import numpy as np

# def check_feasibility_with_gurobi(solution, k, d, h):
#     """
#     Check if a solution is feasible under the given constraints using Gurobi.
    
#     Parameters:
#         solution: numpy array, the solution vector to check (d*h elements).
#         k: int, sparsity level (number of non-zero entries allowed).
#         d: int, number of features.
#         h: int, number of clusters.

#     Returns:
#         bool: True if the solution is feasible, False otherwise.
#     """
#     solution = solution.flatten()

#     # Create Gurobi model
#     model = Model()
#     model.setParam('OutputFlag', 0)  # Suppress output

#     # Define decision variables (not actually optimized, just for feasibility check)
#     x = model.addMVar(shape=d * h, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

#     # Sparsity constraint: sum of all variables <= k
#     model.addConstr(x.sum() <= k, "SparsityConstraint")

#     # Feature assignment constraints: sum across clusters for each feature <= 1
#     for i in range(d):
#         model.addConstr(x[i * h:(i + 1) * h].sum() <= 1, f"FeatureAssignment_{i}")

#     # Cluster coverage constraints: sum across features for each cluster >= 1
#     for j in range(h):
#         model.addConstr(x[j::h].sum() >= 1, f"ClusterCoverage_{j}")

#     # Fix the variables to the provided solution
#     for i in range(d * h):
#         model.addConstr(x[i] == solution[i], f"FixVariable_{i}")

#     # Dummy objective (we're only checking feasibility)
#     model.setObjective(0, GRB.MINIMIZE)

#     # Optimize the model
#     model.optimize()

#     # Check if the solution is feasible
#     return GRB.OPTIMAL



def check_feasibility_with_gurobi(m, k, d, h):
    """
    This function projects the input vector m onto the simplex while satisfying constraints.
    Parameters:
    - m: input vector
    - k: sparsity level (number of non-zero entries allowed)
    - d: number of features
    - h: number of clusters
    """
    A = np.ones((1, d * h))  # Sparsity constraint matrix
    b = np.array([k])        # Sparsity constraint vector

    # Feature assignment constraints
    B = np.zeros((d, d * h))
    for i in range(d):
        B[i, i * h:(i + 1) * h] = 1
    c = np.ones((d, 1))

    # Cluster coverage constraints
    Cluster = np.zeros((h, d * h))
    for j in range(h):
        Cluster[j, j::h] = 1
    Cluster_b = np.ones((h, 1))

    Aa = np.ones((1, d * h))  # Sparsity constraint matrix
    bb = np.array([k-1])        # Sparsity constraint vector

    # Combine all constraints
    C = np.vstack([A, B])
    Cb = np.vstack([b, c])


    # add inter-cluster variable inequality constraints
    # D = construct_difference_matrix(d, h)
    

    with Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # Suppress output
        env.start()
        with Model(env=env) as model:
            # Add variables
            x = model.addMVar(d * h, lb=0.0, ub=1.0)

            # Objective function (quadratic)
            Q = np.eye(d * h)
            f = -2 * m.flatten()
            model.setObjective(x @ Q @ x + f @ x, sense=1)  # Minimize

            # Add constraints
            model.addConstr(C @ x <= Cb.flatten(), name="GeneralConstraints")
            model.addConstr(Aa @ x >= bb.flatten(), name="Sparsity")
            model.addConstr(Cluster @ x >= Cluster_b.flatten(), name="ClusterCoverage")
            for i in range(d * h):
                model.addConstr(x[i] == m[i], f"FixVariable_{i}")
            # model.addConstr(Aa @ x == bb.flatten(), name="Sparsity")
            # model.addConstr(x@ D.T @ D @ x >= k, name="InterCluster")
            

            # Optimize
            model.optimize()

            # Debugging infeasibility
            if model.Status == 3:  # Infeasible
                print("Model is infeasible. Writing infeasibility report...")
                model.computeIIS()
                model.write("infeasibility_report.ilp")
                return None

            # # Return solution
            # solution = x.X
            # # shift the solution to the right by one index
            # solution = [0] + solution

            return GRB.OPTIMAL

def ProjOperator_Gurobi(m, k, d, h):
    """
    This function projects the input vector m onto the simplex while satisfying constraints.
    Parameters:
    - m: input vector
    - k: sparsity level (number of non-zero entries allowed)
    - d: number of features
    - h: number of clusters
    """
    A = np.ones((1, d * h))  # Sparsity constraint matrix
    b = np.array([k])        # Sparsity constraint vector

    # Feature assignment constraints
    B = np.zeros((d, d * h))
    for i in range(d):
        B[i, i * h:(i + 1) * h] = 1
    c = np.ones((d, 1))

    # Cluster coverage constraints
    Cluster = np.zeros((h, d * h))
    for j in range(h):
        Cluster[j, j::h] = 1
    Cluster_b = np.ones((h, 1))

    Aa = np.ones((1, d * h))  # Sparsity constraint matrix
    bb = np.array([k-1])        # Sparsity constraint vector

    # Combine all constraints
    C = np.vstack([A, B])
    Cb = np.vstack([b, c])


    # add inter-cluster variable inequality constraints
    # D = construct_difference_matrix(d, h)
    

    with Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # Suppress output
        env.start()
        with Model(env=env) as model:
            # Add variables
            x = model.addMVar(d * h, lb=0.0, ub=1.0)

            # Objective function (quadratic)
            Q = np.eye(d * h)
            f = -2 * m.flatten()
            model.setObjective(x @ Q @ x + f @ x, sense=1)  # Minimize

            # Add constraints
            model.addConstr(C @ x <= Cb.flatten(), name="GeneralConstraints")
            # model.addConstr(Aa @ x >= bb.flatten(), name="Sparsity")
            # model.addConstr(Cluster @ x >= Cluster_b.flatten(), name="ClusterCoverage")
            # model.addConstr(Aa @ x == bb.flatten(), name="Sparsity")
            # model.addConstr(x@ D.T @ D @ x >= k, name="InterCluster")
            

            # Optimize
            model.optimize()

            # Debugging infeasibility
            if model.Status == 3:  # Infeasible
                print("Model is infeasible. Writing infeasibility report...")
                model.computeIIS()
                model.write("infeasibility_report.ilp")
                return None

            # # Return solution
            # solution = x.X
            # # shift the solution to the right by one index
            # solution = [0] + solution

            return x.x
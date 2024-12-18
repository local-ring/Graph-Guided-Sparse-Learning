from gurobipy import Model, Env
import numpy as np
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
    bb = np.array([k])        # Sparsity constraint vector

    # Combine all constraints
    C = np.vstack([A, B])
    Cb = np.vstack([b, c])

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
            model.addConstr(Cluster @ x >= Cluster_b.flatten(), name="ClusterCoverage")
            model.addConstr(Aa @ x == bb.flatten(), name="Sparsity")
            

            # Optimize
            model.optimize()

            # Debugging infeasibility
            if model.Status == 3:  # Infeasible
                print("Model is infeasible. Writing infeasibility report...")
                model.computeIIS()
                model.write("infeasibility_report.ilp")
                return None

            # Return solution
            return x.x
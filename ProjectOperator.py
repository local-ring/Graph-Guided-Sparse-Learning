from gurobipy import Model
import numpy as np
def ProjOperator_Gurobi(m, k, d, h):
    """
    This function we return a projection operator that projects the input vector m onto the simplex
    Parameters:
    - m: input vector (maybe in it matrix form -- no)
    - k: sparsity level
    - d: number of features
    - h: number of pre-defined clusters, which can be read from the input vector m (omit for now, since we pass the vector form, so we cannot know d and h directly)
    """
    # print("m", m.shape)
    # Create constraint matrix A for the sparsity level
    A = np.ones((1, d*h))
    b = np.array([k])

    # Create constraint matrix B for each feature belongs to at most one cluster
    # Initialize B as an empty matrix with shape (0, d*h)
    B = np.empty((0, d*h))

    # Initialize c as an empty array
    c = np.empty((0, 1))

    for i in range(d):
        # Create a new row of zeros with shape (1, d*h)
        B_row = np.zeros(d*h)
        # Set a specific element in the row to 1
        B_row[i*h: (i+1)*h] = 1

        # Stack the new row onto B
        B = np.vstack([B, B_row]) 

        # Append 1 to vector c
        c = np.vstack([c, [[1]]])

    # Concatenate A and B to create the constraint matrix
    C = np.vstack([A, B])
    Cb = np.vstack([b, c])

    # Create a Gurobi model
    model = Model()
    # Set Gurobi parameters
    model.setParam('OutputFlag', 0)
    model.setParam('IterationLimit', 500)

    # Add variables
    x = model.addMVar(d*h, lb=0.0, ub=1.0)

    # Set objective function
    Q = np.eye(d*h)
    f = -2*m.flatten()
    x = model.addMVar(d*h, ub=1.0, lb=0.0)
    # print("x", x.shape)
    # print("f", f.shape)
    model.setObjective(x@Q@x + x@f)
    model.addConstr(C @ x <= Cb)

    # Optimize model
    model.optimize()

    # Get the results
    x = x.x # no need the new-axis for now...
    # print("result_x", x.shape)

    return x

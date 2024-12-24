import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import inv
from utils.vec_mat import vector_to_symmetric, symmetric_to_vector, compute_upper_triangle_product

def L0Obj_Xs(X, m, y, L, rho, mu, d, h, n):
    """
    Compute the objective function value and gradient for the L0 regularized least squares problem.
    Parameters:
    - X: n x d matrix of features
    - y: n x 1 vector of labels
    - rho: regularization parameter for L2 penalty
    - mu: regularization parameter for L0-graph penalty
    - u: d x 1 vector of weights
    Returns:
    - f: objective function value
    - g: gradient
    """

    # convert the vector to the matrix Xs
    Xs = vector_to_symmetric(m, d)
    X_ii = np.diag(Xs)
    SpDiag = spdiags(X_ii, 0, d, d)
    
    B_inv = (1 / rho) * X @ SpDiag @ X.T + np.eye(n)
    B = np.linalg.solve(B_inv, np.eye(n))

    precision_penalty = y.T @ B @ y
    graph_penalty = 0.5 * mu * np.trace(L @ Xs) # generate the graph penalty term
    f = precision_penalty + graph_penalty

    # compute the gradient matrix with A_ii = precision gradient and A_ij = 0 for i != j
    A_grad = -(1/rho) * ((X.conj().T @ B @ y)**2) # the gradient of the first term when y_ii
    A_grad = np.diag(A_grad) 
    A_grad = symmetric_to_vector(A_grad)

    # compute the second term of the gradient
    B_grad = 0.5 * mu * compute_upper_triangle_product(L, m, d)
    g = A_grad + B_grad
    g = g.flatten()

    return f, g


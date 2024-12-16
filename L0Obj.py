import numpy as np
from scipy.sparse import spdiags
from scipy.linalg import inv

def L0Obj(X, m, y, L, pho, mu, d, h):
    """
    Compute the objective function value and gradient for the L0 regularized least squares problem.
    Parameters:
    - X: n x d matrix of features
    - y: n x 1 vector of labels
    - pho: regularization parameter for L2 penalty
    - mu: regularization parameter for L0-graph penalty
    - u: d x 1 vector of weights
    Returns:
    - f: objective function value
    - g: gradient
    """
    n, dh = X.shape
    if d*h != dh:
        raise ValueError("The dimensions of X and d*h do not match.")
    
    SpDiag = spdiags(m.flatten(), 0, dh, dh)
    # print(SpDiag)
    # print(f"SpDiag: {SpDiag.toarray()[-1][-1]}")
    M = inv((1/pho) * X @ SpDiag @ X.T + np.eye(n))
    # generate the correspodning assignment matrix
    assignment_matrix = m.reshape(d,h)
    # print(f"assignemen_matrix: {assignemen_matrix}")
    # print(f"m: {m}")
    # generate the graph penalty term
    graph_penalty = mu * np.trace(assignment_matrix.T @ L @ assignment_matrix)
    f = y.T @ M @ y + graph_penalty # AL: why conjuate transpose? i remove the .conj()

    A_grad = -(1/pho) * ((X.conj().T @ M @ y)**2) # the gradient of the first term
    B_grad = 2 * mu * L @ assignment_matrix # the gradient of the second term

    g = A_grad + B_grad.flatten()

    return f, g
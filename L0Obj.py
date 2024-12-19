import numpy as np
from scipy.sparse import spdiags
from scipy.linalg import inv

def L0Obj(X, m, y, L, pho, mu, d, h, n, C=1):
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
    B = inv((1/pho) * X @ SpDiag @ X.T + n * np.eye(n))
    # generate the correspodning assignment matrix
    # f(x)=3x² - 2x³, f'(x)=6x - 6x²
    # m = 3 * m**2 - 2 * m**3
    # grad_m = 6 * m - 6 * m**2
    # grad_m = grad_m.reshape(d,h)
    assignment_matrix = m.reshape(d,h)
    # print(f"assignemen_matrix: {assignemen_matrix}")
    # print(f"m: {m}")
    # generate the graph penalty term
    graph_penalty = mu * np.trace(assignment_matrix.T @ L @ assignment_matrix)
    precision_penalty = y.T @ B @ y
    # I want to add sum_i sum_{j\neq j'} m_{ij}m_{ij'} to the objective function

    M = m.reshape(d, h)
    # # Step 1: Compute M M^T (row-wise sums of all pairwise products)
    # pairwise_sum = np.dot(M, M.T)

    # # Step 2: Compute sum of squares (exclude diagonal terms)
    # sum_squares = np.sum(M**2, axis=1)

    # Step 3: Subtract diagonal contributions to exclude j = j'
    # correction_term = np.sum(np.dot(M.T, M)) - np.sum(np.diag(sum_squares))
    MTM = np.dot(M.T, M)
    correction_term = np.sum(MTM) - np.sum(np.diag(MTM))

    # Step 1: Compute row-wise sums
    row_sums = np.sum(M, axis=1, keepdims=True)  # Shape (n_rows, 1)

    # Step 2: Compute the gradient
    gradient = 2 * (row_sums - M)
    
    correction_term = C * np.sum(m**2)
    f = precision_penalty + graph_penalty + correction_term # AL: why conjuate transpose? i remove the .conj()

    A_grad = -(1/pho) * ((X.conj().T @ B @ y)**2) # the gradient of the first term
    # B_grad = 2 * mu * L @ assignment_matrix # the gradient of the second term
    # B_grad = 2 * mu * (L @ assignment_matrix) * grad_m # the gradient of the second term
    B_grad = 2 * mu * (L @ assignment_matrix)  # the gradient of the second term

    # print(f"precision_penalty: {precision_penalty}")
    # print(f"graph_penalty: {graph_penalty}")
    # print(f"A_grad: {A_grad}")
    # print(f"B_grad: {B_grad}")

    g = A_grad + B_grad.flatten() + C * gradient.flatten()

    return f, g


def L0Obj_separate(X, m, y, L, pho, mu, d, h, n):
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
    M = inv((1/pho) * X @ SpDiag @ X.T + n * np.eye(n))
    # generate the correspodning assignment matrix
    m = 3 * m**2 - 2 * m**3
    grad_m = 6 * m - 6 * m**2
    grad_m = grad_m.reshape(d,h)
    assignment_matrix = m.reshape(d,h)
    # generate the graph penalty term
    graph_penalty = mu * np.trace(assignment_matrix.T @ L @ assignment_matrix)
    precision_penalty = y.T @ M @ y
    f = precision_penalty + graph_penalty # AL: why conjuate transpose? i remove the .conj()

    A_grad = -(1/pho) * ((X.conj().T @ M @ y)**2) # the gradient of the first term
    B_grad = 2 * mu * (L @ assignment_matrix) * grad_m # the gradient of the second term

    g = A_grad + B_grad.flatten()

    return f, g, graph_penalty, precision_penalty, A_grad, B_grad
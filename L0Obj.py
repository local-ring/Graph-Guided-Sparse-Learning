import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import inv

def L0Obj(X, m, y, L, pho, mu, d, h, n):
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
    
    SpDiag = spdiags(m.flatten(order='F'), 0, dh, dh)
    # B_inv = (1/pho) * X @ SpDiag @ X.T + np.sqrt(n) * np.eye(n)
    # if np.linalg.matrix_rank(B_inv) != n:
    #     print(f"rank: {np.linalg.matrix_rank(B_inv)}, shape: {B_inv.shape}")
    #     raise ValueError("The matrix is not full rank.")
    # B = inv(B_inv)
    regularization = 1e-8
    B_inv = (1 / pho) * X @ SpDiag @ X.T + np.eye(n) * (1 + regularization)
    # print("B_inv_max:", np.max(B_inv), "B_inv_min:", np.min(B_inv), )
    B = np.linalg.solve(B_inv, np.eye(n))
    # print("B_max:", np.max(B), "B_min:", np.min(B), )
    # y = n * y
    precision_penalty = 0.5 * y.T @ B @ y

    # epsilon = 0.1
    # r = 1 + epsilon
    # r = 0
    # eta = 2 * r * mu + 2 * d + 0.4 * (pho ** 2) 
    # eta = 2 * r * mu + 0.4 * (pho ** 2) 
    # eta = 10 * 0.4 + 1
    # L = csr_matrix(L + r * np.eye(d)) # this can keep L as np.ndarray instead of np.matrix which will mess up with the flatten() function


    M = m.reshape(d,h) # generate the assignment matrix
    # graph_penalty = 0.5 * mu * np.trace(M.T @ L @ M) # generate the graph penalty term
    # only take the first column of M to compute the graph penalty
    # m_0 = M[:,0].reshape(d,1) # only take the first column of M to compute the graph penalty
    # # m_0 is the sum of the columns of M
    # m_0 = np.sum(M, axis=1, keepdims=True)
    # graph_penalty = 0.5 * mu * m_0.T @ L @ m_0 # generate the graph penalty term
    # M = M.astype(np.float64)
    # MTM = np.dot(M.T, M)
    # correction_term = 0.5 * eta * (np.sum(MTM) - np.sum(np.diag(MTM)))


    # Step 1: Compute the row sums of M
    row_sums = np.sum(M, axis=1)  # Sum across columns for each row

    # Step 2: Compute the new column as 1 - row sums
    new_column = 1 - row_sums

    # Step 3: Append the new column to M to form N
    N = np.hstack((M, new_column[:, np.newaxis]))
    graph_penalty = 0.5 * mu * np.trace(N.T @ L @ N) # generate the graph penalty term


    f = precision_penalty + graph_penalty # AL: why conjuate transpose? i remove the .conj()
    # f = precision_penalty + graph_penalty + correction_term 

    A_grad = -(1/(2*pho)) * ((X.conj().T @ B @ y)**2) # the gradient of the first term

    B_grad =  mu * (L @ M)  # the gradient of the second term
    B_grad_row_sums = np.sum(B_grad, axis=1, keepdims=True)  # Sum across columns for each row

    B_grad_new = np.tile(B_grad_row_sums, (1, h)) # copy the gradient mu * (L @ M) h times
    B_grad_new = B_grad_new.flatten() # flatten the gradient for the first column
    # except for the first column, all other gradient is zero
    # B_grad_0 = np.zeros_like(B_grad) # initialize the gradient for the first column

    # B_grad_0[:,0] = B_grad[:,0] 
    # B_grad_0 = B_grad_0.flatten() # flatten the gradient for the first column
    # copy the gradient mu * (L @ M) h times
    # B_grad = mu * (L @ m_0)  # the gradient of the second term 
    # B_grad_0 = np.tile(B_grad, (1, h)) # copy the gradient mu * (L @ M) h times
    # B_grad_0 = B_grad_0.flatten() # flatten the gradient for the first column
    

    # C_grad = eta * (row_sums - M) # the gradient of the third term

    # B_grad = B_grad.flatten()

    # C_grad = C_grad.flatten()

    # g = A_grad + B_grad + C_grad
    # g = A_grad + B_grad_0
    # g = A_grad + B_grad
    g = A_grad + B_grad_new
    # g = A_grad
    # f = precision_penalty
    # print("value of g norm:", np.linalg.norm(g))
    # print("max of g:", np.max(g))
    # # scale the gradient
    # f = f / np.linalg.norm(g)
    # g = g / np.linalg.norm(g)

    # print("value of g norm after scaling:", np.linalg.norm(g))
    # print("max of g after scaling:", np.max(g))


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
    B_inv = (1/pho) * X @ SpDiag @ X.T + n * np.eye(n)
    if np.linalg.matrix_rank(B_inv) != n:
        print(f"rank: {np.linalg.matrix_rank(B_inv)}, shape: {B_inv.shape}")
        raise ValueError("The matrix is not full rank.")
    B = inv(B_inv)

    precision_penalty = 0.5 * y.T @ B @ y

    epsilon = 0.1
    r = 1 + epsilon
    # r = 0
    # eta = 2 * r * mu + 2 * d + 0.4 * (pho ** 2) 
    # eta = 100
    eta = 0.34 * 0.4 + 1

    # L = L + C * np.eye(d) # add identity matrix to L
    L = csr_matrix(L + r * np.eye(d)) # this can keep L as np.ndarray instead of np.matrix which will mess up with the flatten() function


    M = m.reshape(d,h) # generate the assignment matrix
    graph_penalty = 0.5 * mu * np.trace(M.T @ L @ M) # generate the graph penalty term

    MTM = np.dot(M.T, M)
    correction_term = 0.5 * eta * (np.sum(MTM) - np.sum(np.diag(MTM)))

    row_sums = np.sum(M, axis=1, keepdims=True)  
    f = precision_penalty + graph_penalty + correction_term # AL: why conjuate transpose? i remove the .conj()

    A_grad = -(1/(2*pho)) * ((X.conj().T @ B @ y)**2) # the gradient of the first term
    # print("L type:", type(L))
    # print("M type:", type(M))
    B_grad =  mu * (L @ M)  # the gradient of the second term

    C_grad = eta * (row_sums - M) # the gradient of the third term

    # print("A_grad shape:", A_grad.shape)
    # print("B_grad shape:", B_grad.shape)
    B_grad = B_grad.flatten()
    # print("B_grad shape:", B_grad.shape)
    # print("C_grad shape:", C_grad.shape)
    C_grad = C_grad.flatten()
    # print("C_grad shape:", C_grad.shape)

    g = A_grad + B_grad + C_grad

    return f, g, graph_penalty, precision_penalty, correction_term, A_grad, B_grad, C_grad
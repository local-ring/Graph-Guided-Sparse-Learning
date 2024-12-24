import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.linalg import inv

def vector_to_symmetric(vec, n):
    """
    Convert a vector of independent variables to a symmetric matrix.
    
    Parameters:
    - vec: A flat vector containing the upper triangular entries of the matrix.
    - n: The dimension of the square symmetric matrix.
    
    Returns:
    - A symmetric matrix of shape (n, n).
    """
    sym_matrix = np.zeros((n, n))
    upper_indices = np.triu_indices(n)
    sym_matrix[upper_indices] = vec
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))
    return sym_matrix

def symmetric_to_vector(matrix):
    """
    Extract the upper triangular entries of a symmetric matrix as a vector.
    
    Parameters:
    - matrix: A symmetric matrix.
    
    Returns:
    - A flat vector of the independent entries.
    """
    n = matrix.shape[0]
    upper_indices = np.triu_indices(n)
    return matrix[upper_indices]

def compute_upper_triangle_product(L, vec, n):
    """
    Compute the product L_ij * y_ij for the upper triangular part of the symmetric matrix.
    
    Parameters:
    - L: Symmetric matrix of shape (n, n).
    - vec: Flattened upper triangular vector of the matrix to project.
    - n: Dimension of the square matrix.

    Returns:
    - Flattened product of L_ij * y_ij for the upper triangular part.
    """
    # vec = vec.flatten()
    # Reconstruct the symmetric matrix from the vector
    sym_matrix = vector_to_symmetric(vec, n)
    
    # Extract the upper triangular indices
    upper_indices = np.triu_indices(n)
    
    # Compute L_ij * y_ij for upper triangular elements
    L = L.toarray()
    # print("L[upper_indices].shape:", L[upper_indices].reshape(-1,1).shape, "sym_matrix[upper_indices].shape:", sym_matrix[upper_indices].shape)

    product_upper = L[upper_indices] * sym_matrix[upper_indices]
    
    # Return the product as a flat vector
    return product_upper

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
    # n, dh = X.shape
    # if d*h != dh:
    #     raise ValueError("The dimensions of X and d*h do not match.")
    
    Xs = vector_to_symmetric(m, d)
    X_ii = np.diag(Xs)
    SpDiag = spdiags(X_ii, 0, d, d)
    
    B_inv = (1 / rho) * X @ SpDiag @ X.T + np.eye(n)
    # print("B_inv_max:", np.max(B_inv), "B_inv_min:", np.min(B_inv), )
    B = np.linalg.solve(B_inv, np.eye(n))
    # print("B_max:", np.max(B), "B_min:", np.min(B), )
    # y = n * y
    precision_penalty = y.T @ B @ y

    # epsilon = 0.1
    # r = 1 + epsilon
    # # r = 0
    # # eta = 2 * r * mu + 2 * d + 0.4 * (rho ** 2) 
    # eta = 2 * r * mu + 0.4 * (rho ** 2) 
    # L = csr_matrix(L + r * np.eye(d)) # this can keep L as np.ndarray instead of np.matrix which will mess up with the flatten() function


    # M = m.reshape(d,h) # generate the assignment matrix
    graph_penalty = 0.5 * mu * np.trace(L @ Xs) # generate the graph penalty term


    f = precision_penalty + graph_penalty 

    A_grad = -(1/rho) * ((X.conj().T @ B @ y)**2) # the gradient of the first term when y_ii
    # i want to fill in the matrix with the gradient of the first term when y_ij for i != j it is zero otherwise its the above
    A_grad = np.diag(A_grad) 
    # take the upper triangular part of the matrix
    A_grad = symmetric_to_vector(A_grad)

    B_grad = 0.5 * mu * compute_upper_triangle_product(L, m, d)


    # print("A_grad.shape:", A_grad.shape, "B_grad.shape:", B_grad.shape)
    # g = A_grad + B_grad + C_grad
    g = A_grad + B_grad
    f = precision_penalty + graph_penalty
    # print("value of g norm:", np.linalg.norm(g))
    # print("max of g:", np.max(g))
    # # scale the gradient
    # f = f / np.linalg.norm(g)
    # g = g / np.linalg.norm(g)

    # print("value of g norm after scaling:", np.linalg.norm(g))
    # print("max of g after scaling:", np.max(g))

    # print the shape of the f, g
    # print("f.shape:", f.shape, "g.shape:", g.shape)
    # f = f.flatten()
    g = g.flatten()
    # print("f.shape:", f.shape, "g.shape:", g.shape)


    return f, g


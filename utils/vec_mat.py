import numpy as np

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

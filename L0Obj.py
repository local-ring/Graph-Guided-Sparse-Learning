import numpy as np
from scipy.sparse import spdiags
from scipy.linalg import inv

def L0Obj(u, X, y, pho):
    # u (feature, 1)
    # X (instance, feature)
    # y (instance, 1)
    # pho (1, 1)
    #print(pho)
    n, d = X.shape
    SpDiag = spdiags(u.flatten(), 0, d, d)
    # print(SpDiag)
    # print(f"SpDiag: {SpDiag.toarray()[-1][-1]}")
    M = inv((1/pho) * X @ SpDiag @ X.conj().T + np.eye(n))
    f = y.conj().T @ M @ y

    g = -(1/pho) * ((X.conj().T @ M @ y)**2)

    return f, g
import numpy as np
import random

# random.seed(1)

def ProjCSimplex(u, k):
    """
    Projects vector u onto the simplex defined by the sum of elements equal to k.
    """
    # u = u.flatten()
    # print(u.shape)
    n, m = u.shape
    # print(f"n: {n}, {m}")

    H = 0.5*np.eye(n)
    f = -0.5 * u
    A = np.ones((1, n))
    b = k

    lb = np.zeros((n, 1))
    ub = np.ones((n, 1))
    
    if np.sum(u) < k and np.all(np.logical_and(u >= 0, u <= 1)):
        up = u
    else:
        d = len(u)
        e = np.ones((d, 1))
        z0 = np.zeros((d, 1))
        l0 = float(0 + np.random.rand() * np.min(u))
        error = 100
        idt = 0
        while error > 1e-8:
            tmp = u - l0
            # tmp[tmp > 1] = 1
            # tmp[tmp < 0] = 0
            tmp = np.clip(tmp, 0, 1)
            error = k - np.sum(tmp)
            n = np.count_nonzero(tmp)
            if n == 0:
                l0 = float(0+np.random.rand() * np.min(u))
                error = 100
                continue
            idt += 1
            l1 = l0 - (error / n)
            tmp = u - l1
            # tmp[tmp > 1] = 1
            # tmp[tmp < 0] = 0
            tmp = np.clip(tmp, 0, 1)
            error = np.abs(k - np.sum(tmp))
            l0 = l1
        
        if l0 < 0:
            l0 = 0
        tmp = u - l0
        # tmp[tmp > 1] = 1
        # tmp[tmp < 0] = 0
        tmp = np.clip(tmp, 0, 1)
        up = tmp

    # print(f"up: {up[-1][-1]}")
    
    return up
import numpy as np
from scipy.sparse import spdiags
from scipy.linalg import inv
from gurobipy import Model, GRB, QuadExpr


import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time
import random
from scipy.linalg import block_diag
from sklearn.metrics import precision_recall_curve, auc
import scipy.stats
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from collections import defaultdict

import sys
# sys.path.append('/Users/yijwang-admin/Documents/Research/GFL/Code/PQN_Python_main')
from minConf.minConf_PQN import minConF_PQN
import random

def Lasso(u, X, y, rho):
    n, d = X.shape
    D_u = spdiags(u.flatten(), 0, d, d)
    M = inv((1/rho) * X @ D_u @ X.T + np.eye(n))
    f = y.T @ M @ y
    g = -(1/rho) * ((X.T @ M @ y)**2)

    return f, g

def GroupLasso(u, X, y, rho):
    n, d = X.shape
    D_u = spdiags(u.flatten(), 0, d, d)
    M = inv((1/rho) * X @ D_u @ X.T + np.eye(n))
    f = y.T @ M @ y
    g = -(1/(2*rho)) * ((X.T @ M @ y)**2)

    return f.flatten(), g.flatten()

def GeneralizedFusedLasso(u, X, y, rho, L, mu):
    n, d = X.shape
    D_u = spdiags(u.flatten(), 0, d, d)
    M = inv((1/rho) * X @ D_u @ X.T + np.eye(n))
    f = y.T @ M @ y + mu * u.T @ L @ u
    g = -(1/(2* rho)) * ((X.T @ M @ y)**2) + 2 * mu * L @ u

    return f.flatten(), g.flatten()


def ProjLassoGurobi(u, k, d):
    A = np.ones((1, d))
    b = np.array([k])
    Q = np.eye(d)
    f = -2 * u.flatten()

    model = Model("GeneralizedFusedLasso")
    x = model.addMVar(d, lb=0.0, ub=1.0)

    model.setObjective(x@Q@x + x@f)
    model.addConstr(A @ x == b)

    model.setParam('OutputFlag', 0)
    model.setParam('IterationLimit', 500)

    model.optimize()
    print("Lasso x.x: ", x.x)

    return x.x

def ProjGroupLassoGurobi(u, k, groups, h, d):
    """
    Solves the Group Lasso projection problem using Gurobi.

    Parameters:
    - u: Input vector of size (d, 1)
    - k: Sparsity level (L1 constraint)
    - groups: List of groups, where each group is a list of indices
    - h: Sum constraint for group norms

    Returns:
    - up: Projected vector (u')
    - zp: Auxiliary variables for group norms
    """
    # d = u.shape  # Number of variables
    g = len(groups)  # Number of groups

    # Quadratic objective matrices and linear term
    H1 = np.eye(d)  # Identity for quadratic term on `u`
    f1 = -2 * u.flatten()  # Linear term for `u`
    H2 = np.zeros((g, g))  # No quadratic term for auxiliary variables
    f2 = np.zeros(g)  # No linear term for auxiliary variables

    # Combine H1, H2 and f1, f2 into block matrices
    H = np.block([[H1, np.zeros((d, g))], [np.zeros((g, d)), H2]])
    f = np.concatenate([f1, f2])

    # Constraint matrices for `u` (A1) and `z` (A2)
    A1 = np.vstack([np.ones((1, d)), -np.ones((1, d))])  # L1 constraint on `u`
    A2 = np.zeros((2, g))  # Initialize A2
    group_constraints_count = 0

    # Build group constraints
    for i, group in enumerate(groups):
        for idx in group:
            A1_row = np.zeros((1, d))
            A1_row[0, idx] = 1
            A1 = np.vstack([A1, A1_row])  # Adding constraints for group indices

            A2_row = np.zeros((1, g))
            A2_row[0, i] = -1
            A2 = np.vstack([A2, A2_row])  # Adding constraints for group norms
            group_constraints_count += 1

    # Add overall group norm constraint
    A1 = np.vstack([A1, np.zeros((1, d))])
    A2 = np.vstack([A2, np.ones((1, g))])
    A = np.hstack([A1, A2])  # Combine A1 and A2

    # RHS vector for constraints
    b = np.zeros(group_constraints_count + 3)
    b[0] = k  # Sparsity constraint (sum of u <= k)
    b[1] = -k + 1  # Lower bound for sparsity
    b[-1] = h  # Sum constraint for group norms

    # Lower and upper bounds for variables
    lb = np.zeros(d + g)
    ub = np.ones(d + g)

    # Setup Gurobi model
    model = Model("ProjGroupLasso")
    x = model.addMVar(d + g, lb=lb, ub=ub)  # Variables for `u` and `z`

    # Set quadratic objective
    model.setObjective(x @ H @ x + x @ f)

    # Add constraints
    model.addConstr(A @ x <= b)

    # Configure Gurobi parameters
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('IterationLimit', 500)
    # Solve the model
    model.optimize()

    # Extract the results
    up = x.x[:d]  # Projected vector `u'`
    zp = x.x[d:]  # Auxiliary variables `z'`

    print("GroupLasso x.x: ", up[:, np.newaxis])

    return up[:, np.newaxis]


def ProjGeneralizedFusedLassoGurobi(u, k, d):
    d = u.shape
    A = np.ones((1, d))
    b = np.array([k])
    Q = np.eye(d)
    f = -2 * u.flatten()

    model = Model("GeneralizedFusedLasso")
    x = model.addMVar(d, lb=0.0, ub=1.0)

    model.setObjective(x@Q@x + x@f)
    model.addConstr(A @ x == b)

    model.setParam('OutputFlag', 0)
    model.setParam('IterationLimit', 500)

    model.optimize()
    print("GeneralizedFusedLasso x.x: ", x.x)

    return x.x



class RandomEnsemble:
    def __init__(self, n, d, k, h_total, h_selected, h_rest, gamma, 
                 p=0.95, q=0.01, 
                 options=None, num_replications=100):
        assert h_total == h_selected + h_rest, "h_total should be equal to h_selected + h_rest"
        self.n = n
        self.d = d
        self.k = k
        self.h_total = h_total
        self.h_selected = h_selected
        self.h_rest = h_rest
        self.gamma = gamma
        self.p = p
        self.q = q
        self.options = options if options else {'maxIter': 500, 'verbose': 0}
        self.num_replications = num_replications

    def _random_partition(self):
        # partition the first k nodes into h_selected groups and the rest into h_rest groups
        assert self.h_selected <= self.k, "h_selected should be less than k"
        assert self.h_rest <= self.d - self.k, "h_rest should be less d-k"
        break_points = np.sort(random.sample(range(1, self.k), self.h_selected-1))
        break_points_rest = np.sort(random.sample(range(self.k+1, self.d), self.h_rest-1))

        return break_points, break_points_rest
    
    def _generate_clusters(self):
        break_points, break_points_rest = self._random_partition()
        clusters = []
        clusters.append(np.arange(break_points[0])) # first selected cluster
        for i in range(1, self.h_selected-1):
            clusters.append(np.arange(break_points[i-1], break_points[i]))
        clusters.append(np.arange(break_points[-1], self.k)) # last selected cluster

        clusters.append(np.arange(self.k, break_points_rest[0])) # first rest cluster
        for i in range(1, self.h_rest-1):
            clusters.append(np.arange(break_points_rest[i-1], break_points_rest[i]))
        clusters.append(np.arange(break_points_rest[-1], self.d))

        return clusters

    def _generate_graph(self):
        # here we generate the adjacency matrix and laplacian of the graph
        clusters = self._generate_clusters()

        A = sp.lil_matrix((self.d, self.d), dtype=int)
        
        # generate thr inner cluster connections
        for cluster in clusters:
            cluster_size = len(cluster)
            block = (np.random.rand(cluster_size, cluster_size) < self.p).astype(int)
            np.fill_diagonal(block, 0) # no self-loop
            block = np.triu(block) + np.triu(block, 1).T # make it symmetric
            for i, node_i in enumerate(cluster):
                for j, node_j in enumerate(cluster):
                    A[node_i, node_j] = block[i, j]

        # generate the connections between clusters
        for i in range(self.h_total):
            for j in range(i+1, self.h_total):
                cluster_i = clusters[i]
                cluster_j = clusters[j]
                block = (np.random.rand(len(cluster_i), len(cluster_j)) < self.q).astype(int)
                for m, node_i in enumerate(cluster_i):
                    for n, node_j in enumerate(cluster_j):
                        A[node_i, node_j] = block[m, n]
                        A[node_j, node_i] = block[m, n]

        # TODO: check if the graph is connected and make it connected if not 
        # (optional, maybe not necessary)
        D = sp.diags(np.ravel(A.sum(axis=1)))
        L = D - A
        return L, clusters
    
    def _visualize_graph(self, A):
        # if we want to visualize the graph, we need to change it to array rather than sparse matrix
        A_arr = A.toarray()
        plt.figure(figsize=(8, 8))
        plt.title('Adjacency matrix')
        plt.spy(A_arr)
        plt.axis('off')
        plt.show()

    def _generate_w(self, clusters):
        w = np.zeros(self.d)
        for i in range(self.h_selected):
            cluster_sign = np.random.choice([-1, 1])
            cluster_weight = cluster_sign * (1 / np.sqrt(self.k)) # TODO: make more choice other than 1/sqrt(k)
            for node in clusters[i]:
                w[node] = cluster_weight
        return w

    def _generate_X(self):
        X = np.random.normal(0, 1, (self.n, self.d))
        return X
    
    def _generate_y(self, X, w):
        noise = np.random.normal(0, self.gamma, self.n)
        signal = X @ w
        y = signal + noise
        SNR = self._compute_snr(signal, noise)
        if 0: # TODO: add a debug/verbose flag
            print(f'SNR: {SNR}')

        return y

    def _generate_data(self):
        L, clusters = self._generate_graph()
        if 0: # TODO: add a debug/verbose flag 
            self._visualize_graph(A)
        w = self._generate_w(clusters)
        X = self._generate_X()
        y = self._generate_y(X, w)
        return L, w, X, y, clusters


    def _compute_snr(self, signal, noise):  
        # in our case, the snr is 10*log10(1/gamma^2)
        signal = np.asarray(signal)
        noise = np.asarray(noise)

        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
        
        return snr_db
    
    def _define_obj(self, model, X, y, rho=1, L=None, mu=None):
        """
        Support models: 'Lasso', GroupLasso', 'GeneralizedFusedLasso'
        I plan to support Proximal method by calling matlab session
        this part and the one below is to support PQN, the Proxiaml method is on its own, it does not need this
        """
        if model == "Lasso":
            return lambda u: Lasso(u, X, y, rho)
        elif model == "GroupLasso":
            return lambda u: GroupLasso(u, X, y, rho)
        elif model == "GeneralizedFusedLasso":
            return lambda u: GeneralizedFusedLasso(u, X, y, rho, L, mu)
        else:
            raise ValueError("Model not supported")
        
    def _define_projection(self, model, groups=None):
        """
        Support models: 'Lasso', GroupLasso', 'GeneralizedFusedLasso'
        """
        if model == "Lasso":
            return lambda u: ProjLassoGurobi(u, self.k, self.d)
        elif model == "GroupLasso":
            return lambda u: ProjGroupLassoGurobi(u, self.k, groups, self.h_selected, self.d)
        elif model == "GeneralizedFusedLasso":
            return lambda u: ProjGeneralizedFusedLassoGurobi(u, self.k, self.d)
        else:
            raise ValueError("Model not supported")
        

    def solver(self, model, X, y, clusters=None, L=None):
        if model == "Proximal":
            return self.solve_proximal(X, y)
        elif model == "Lasso":
            rho = np.sqrt(self.n) # TODO: check if this is the correct value
            funObj = self._define_obj(model, X, y, rho)
            funProj = self._define_projection(model)
        elif model == "GroupLasso":
            rho = np.sqrt(self.n) # TODO: check if this is the correct value
            funObj = self._define_obj(model, X, y, rho)
            funProj = self._define_projection(model, clusters)
        elif model == "GeneralizedFusedLasso":
            rho = np.sqrt(self.n) # TODO: check if this is the correct value
            mu = 1
            funObj = self._define_obj(model, X, y, rho, L, mu)
            funProj = self._define_projection(model)
        else:
            raise ValueError("Model not supported")
        
        uSimplex = np.ones((self.d, 1)) / self.d
        uout, obj, _ = minConF_PQN(funObj, uSimplex, funProj, self.options)
        uout = funProj(uout) # TODO: check if this is necessary

        return uout
        
    def solver_proximal(self):
        # need to return the result,
        pass

    def cross_validation(self, model):
        pass

    def recovery_accuracy(self, u):
        # evaluate the support recovery accuracy
        # we take top k for u
        selected_features_true = np.arange(self.k)
        selected_features_pred = np.argsort(u)[-self.k:]
        correct_pred = np.intersect1d(selected_features_true, selected_features_pred)
        accuracy = len(correct_pred) / self.k

        return accuracy
    
    def main(self):
        models = ["Lasso", "GroupLasso", "GeneralizedFusedLasso"]
        model_accuracy = defaultdict(list)
        for _ in range(self.num_replications):
            L, w, X, y, clusters = self._generate_data()
            for model in models[:1]:
                u = self.solver(model, X, y, clusters, L)
                accuracy = self.recovery_accuracy(u)
                model_accuracy[model].append(accuracy)

        return model_accuracy
    
# TODO: solve the orginal problem and calculate the MSE?
    
        
a = RandomEnsemble(n=100, d=200, k=20, h_total=10, h_selected=2, h_rest=8, gamma=0.5)
res = a.main()
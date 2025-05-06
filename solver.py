# === Standard Library ===
import os
import sys
import time
import random
import pickle
import itertools
from collections import defaultdict

# === Scientific Libraries ===
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy import sparse
from scipy.linalg import toeplitz, block_diag
from scipy.sparse import csgraph
from scipy.stats import multivariate_normal

# === Machine Learning ===
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, ElasticNetCV
from sklearn.metrics import precision_recall_curve, auc

# === Plotting ===
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# === Graph Tools ===
import networkx as nx

# === Local Module ===
from signal_family_solver import sparse_learning_solver

# Optional MATLAB Engine (commented out)
# import matlab.engine


class Solver:    
    def __init__(self, models, c=1, options=None,                  
                 datafile=f'./code_fgfl_aaai14/data_gfl/',
                 resultfile='./code_fgfl_aaai14/result_gfl/',):   
        self.res = defaultdict(list)
        self.models = models
        self.n = None
        self.d = None
        self.k = None
        self.options = options if options else {'maxIter': 500, 'verbose': 0, 'SPGiters': 100}
        self.datafile = os.path.join(os.path.abspath(datafile), f'real_data')
        self.resultfile = os.path.join(os.path.abspath(resultfile), f'real_data') # matlab does not like relative path
        self.best_rho1 = 0.5
        self.best_rho2 = 0.5
        self.best_mu = 1.0
        self.best_rho = None
        self.datafile_pqn = os.path.join(os.path.abspath('./PQN/data/'), f'real_data')
        self.resultfile_pqn = os.path.join(os.path.abspath('./PQN/result/'), f'real_data')
        self._init(self.datafile, self.resultfile)
        self._init(self.datafile_pqn, self.resultfile_pqn)
        self.c = c



    def _init(self, datafile, resultfile):
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        if not os.path.exists(resultfile):
            os.makedirs(resultfile)

    def _solver_lasso_sklearn(self, X, y):
        """
        Use sklearn's Lasso implementation to solve the Lasso problem.
        """
        alpha = 0.1
        lasso_model = Lasso(alpha=alpha, max_iter=10000)  # Lasso model with high max_iter
        lasso_model.fit(X, y)  # Fit the model
        u = lasso_model.coef_  # Get the coefficients
        # print(f"Lasso coefficients: {u}")
        return u 

    def _solver_proximal(self, X, y, A, i):
        datafile_name = os.path.join(self.datafile, f'data_{i}.mat')
        resultfile_name = os.path.join(self.resultfile, f'result_{i}.mat')
        self._save_mat(X, y, A, datafile_name)
        self._call_proximal(datafile_name, resultfile_name, self.best_rho1, self.best_rho2)
        u, funcVal = self._read_result(resultfile_name)
        return u.flatten() # the original return a vector with shape (d,1), will not work with recovery_accuracy

    def _solver_gfl(self, X, y, L, i, rho=None, mu=0.01, k=None):
        datafile_pqn = os.path.join(self.datafile_pqn, f'data_{i}.mat')
        resultfile_pqn = os.path.join(self.resultfile_pqn, f'result_{i}.mat')
        self._save_mat_pqn(X, y, L, datafile_pqn)
        # if rho is None or mu is None: # we don't need to store the data when we do Lasso
            # self._save_mat_pqn(X, y, L, datafile_pqn)
        if rho is None or mu is None:
            if i == 0:
                rho_values = [np.sqrt(self.n), 6.8 * np.sqrt(self.n)]
                mu_values = [0.01, 0.1, 1.0]
                self.best_rho, self.best_mu = self._cross_validation_gfl(X, y, L, rho_values, mu_values, k=k)
                print(f"Best rho: {self.best_rho}, Best mu: {self.best_mu}")
            rho = self.best_rho
            mu = self.best_mu

        self._call_gfl(datafile_pqn, resultfile_pqn, rho, mu, k)
        u, _ = self._read_result(resultfile_pqn)
        return u.flatten()
    
    def _save_mat_pqn(self, X, y, L, filename=None):
        # save the data to .mat file so that the matlab code of proxiaml can use it
        # print("X.shape", X.shape)   
        # print("y.shape", y.shape)
        # print("A.shape", A.shape)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        data = {
            "X": X,
            "y": y,
            "L": L.toarray() if sp.issparse(L) else L,  # we store the adjacency matrix as dense matrix
        }
        sio.savemat(filename, data)

    def _read_result(self, resultfile):
        result = sio.loadmat(resultfile)
        beta, funcVal = result['beta'], result['funcVal']
        return beta, funcVal
        

    def _call_gfl(self, datafile, resultfile, rho, mu, k=None):
        eng = matlab.engine.start_matlab()
        try:
            eng.cd(os.path.abspath('./PQN/'))
            eng.addpath(os.path.abspath('./PQN/'))
            eng.addpath(eng.genpath(os.path.abspath('./PQN/')))
            eng.addpath(eng.genpath(os.path.abspath('./PQN/minConF/')))
            if k:
                eng.gfl_pqn(datafile, resultfile, rho, mu, float(k), nargout=0)
            else:
                eng.gfl_pqn(datafile, resultfile, rho, mu, float(self.k), nargout=0)
        finally:
            eng.quit()
    
    def _save_result(self, u, filename):
        sio.savemat(filename, {'beta': u})

    def _call_proximal(self, datafile, resultfile, rho1, rho2):
        eng = matlab.engine.start_matlab()
        try:
            eng.cd(os.path.abspath('./code_fgfl_aaai14/'))
            eng.addpath(os.path.abspath('./code_fgfl_aaai14/GFL/'))
            eng.addpath(eng.genpath(os.path.abspath('./code_fgfl_aaai14/')))
            eng.gfl_proximal(datafile, resultfile, rho1, rho2, nargout=0)
        finally:
            eng.quit()

    def _solver_aGrace(self, X, y, W, lambda1=1.0, lambda2=1.0, max_iter=1000, tol=1e-4):
        # Standardize X and center y
        n, p = X.shape
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1  # avoid division by zero
        X = (X - X_mean) / X_std
        y_mean = y.mean()
        y = y - y_mean

        # Compute initial estimate beta_tilde
        if p < n:
            lr = LinearRegression(fit_intercept=False)
            lr.fit(X, y)
            beta_tilde = lr.coef_
        else:
            enet = ElasticNetCV(l1_ratio=0.5, fit_intercept=False, cv=5, max_iter=10000)
            enet.fit(X, y)
            beta_tilde = enet.coef_

        # Construct modified Laplacian matrix Lstar
        d = W.sum(axis=1).A1 if hasattr(W, 'A1') else W.sum(axis=1)  # handle sparse matrices
        Lstar = np.zeros((p, p))
        rows, cols = W.nonzero()
        for i in range(len(rows)):
            u, v = rows[i], cols[i]
            if u >= v:
                continue  # process each edge once
            if d[u] == 0 or d[v] == 0:
                Lstar[u, v] = Lstar[v, u] = 0
            else:
                sign_u = np.sign(beta_tilde[u]) if beta_tilde[u] != 0 else 0
                sign_v = np.sign(beta_tilde[v]) if beta_tilde[v] != 0 else 0
                weight = W[u, v] if isinstance(W, np.ndarray) else W.data[i]
                Lstar_uv = -sign_u * sign_v * weight / np.sqrt(d[u] * d[v])
                Lstar[u, v] = Lstar_uv
                Lstar[v, u] = Lstar_uv
        np.fill_diagonal(Lstar, 1 * (d > 0))  # set diagonal to 1 if degree > 0

        # Precompute adjacency list
        adjacency_list = [[] for _ in range(p)]
        for u, v in zip(rows, cols):
            if u != v:
                adjacency_list[u].append(v)

        # Initialize beta and residual
        beta = np.zeros(p)
        residual = y.copy()
        prev_beta = np.inf * np.ones(p)
        iter = 0

        # Coordinate descent
        while iter < max_iter and np.linalg.norm(beta - prev_beta) > tol:
            prev_beta = beta.copy()
            for u in range(p):
                xu = X[:, u]
                current_beta_u = beta[u]

                # Compute xuTr and neighbor_sum
                xuTr = xu @ residual
                xuTr_plus = xuTr + n * current_beta_u  # since xu.T @ xu = n

                neighbor_sum = 0
                for v in adjacency_list[u]:
                    neighbor_sum += Lstar[u, v] * beta[v]

                # Update beta_u
                z = (xuTr_plus - lambda2 * neighbor_sum) / (n + lambda2)
                threshold = lambda1 / (2 * (n + lambda2))
                beta_u_new = np.sign(z) * max(abs(z) - threshold, 0)

                # Update residual and beta
                delta = beta_u_new - current_beta_u
                residual -= xu * delta
                beta[u] = beta_u_new

            iter += 1

        return beta

    def _save_mat(self, X, y, A, filename=None):
        # save the data to .mat file so that the matlab code of proxiaml can use it
        # print("X.shape", X.shape)   
        # print("y.shape", y.shape)
        # print("A.shape", A.shape)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        data = {
            "X": X,
            "y": y,
            "AdjMat": A.toarray() if sp.issparse(A) else A,  # we store the adjacency matrix as dense matrix
        }
        sio.savemat(filename, data)

    def _signal_family_solver(self, X, y, i, c=1, g=1, edges=None, costs=None):

        step = 1
        # num_cpus = 40
        # num_trials = 10
        max_epochs = 50
        tol_algo = 1e-20
        s = self.k # sparsity level
        # (trial_i, x_mat, y, edges, costs, s, g, max_epochs, tol_algo, step, c)
        results = sparse_learning_solver((i, X, y, edges, costs, s, g, max_epochs, tol_algo, step, c))
        return results


    def _convert_to_edges(self, A):
        if not sparse.issparse(A):
            A = sp.csr_matrix(A)
        A_coo = A.tocoo()
        edges = np.vstack((A_coo.row, A_coo.col)).T
        costs = A_coo.data.astype(np.float64)
        return edges, costs

    def solver(self, model, X, y, c=1, clusters=None, L=None, A=None, i=None, rho=1, mu=1, x_star=None):
        if model == "Proximal":
            return self._solver_proximal(X, y, A, i)
        elif model == "Lasso":
            rho = np.sqrt(self.n) 
            return self._solver_gfl(X, y, L, i, rho, mu=0.0)
        elif model == "GFL_Matlab":
            return self._solver_gfl(X, y, L, i, rho=np.sqrt(self.n), mu=0.01)
        elif model == "Lasso_Sklearn":
            return self._solver_lasso_sklearn(X, y)
        elif model == "Adaptive_Grace":
            if not isinstance(A, np.ndarray):
                Aa = A.toarray()
                return self._solver_aGrace(X, y, Aa)
            else:
                return self._solver_aGrace(X, y, A)
        elif model == "signal_family":
            edges, costs = self._convert_to_edges(A)
            return self._signal_family_solver(X, y, i, c=1, edges=edges, costs=costs)
        else:
            raise ValueError("Model not supported")

    def inference(self, X, y, L, A, k):
        self.n, self.d = X.shape
        self.k = k
        for i, model in enumerate(self.models):
            print(f"Running model {model}")
            if model == "signal_family":
                u = self.solver(model, X, y, c=self.c, L=L, A=A, i=1)
                # unpack the result
                trial_i, results = u
                for method, (x_hat, *_) in results.items():
                    self.res[method] = x_hat
            else:
                u = self.solver(model, X, y, L=L, A=A, i=i)
                self.res[model] = u
        return self.res
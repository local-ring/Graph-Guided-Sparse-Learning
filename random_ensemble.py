# ──────────────────────────────
# 1.  Standard‑library modules
# ──────────────────────────────
import itertools
import multiprocessing
import os
import pickle
import random
import sys
import time
from collections import defaultdict

# ──────────────────────────────
# 2.  Scientific stack
# ──────────────────────────────
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy import linalg, stats            # toeplitz, block_diag, multivariate_normal, csgraph, …
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LinearRegression, Ridge
)
from sklearn.metrics import precision_recall_curve, auc

# ──────────────────────────────
# 3.  Graph & visualisation
# ──────────────────────────────
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as snsdin
from tqdm import tqdm

# ──────────────────────────────
# 4.  External engines (heavy / optional)
# ──────────────────────────────
import matlab.engine       # start when needed, not at import time



class RandomEnsemble:
    def __init__(self, n, d, k, h_total, h_selected, h_rest, gamma, 
                 p=0.95, q=0.01, 
                 options=None, num_replications=20,
                 datafile=f'./code_fgfl_aaai14/data_gfl/',
                 resultfile='./code_fgfl_aaai14/result_gfl/',
                 models=None):
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
        self.options = options if options else {'maxIter': 500, 'verbose': 0, 'SPGiters': 100}
        self.num_replications = num_replications
        self.datafile = os.path.join(os.path.abspath(datafile), f'{n}_{d}_{k}_{h_total}_{h_selected}_{h_rest}_{gamma}_{p}_{q}')
        self.resultfile = os.path.join(os.path.abspath(resultfile), f'{n}_{d}_{k}_{h_total}_{h_selected}_{h_rest}_{gamma}_{p}_{q}') # matlab does not like relative path
        self.best_rho1 = 0.5
        self.best_rho2 = 0.5
        self.best_mu = 1.0
        # self.best_rho = np.sqrt(self.n) * 6.8
        self.best_rho = np.sqrt(self.n)
        self.datafile_pqn = os.path.join(os.path.abspath('./PQN/data/'), f'{n}_{d}_{k}_{h_total}_{h_selected}_{h_rest}_{gamma}_{p}_{q}')
        self.resultfile_pqn = os.path.join(os.path.abspath('./PQN/result/'), f'{n}_{d}_{k}_{h_total}_{h_selected}_{h_rest}_{gamma}_{p}_{q}')
        self._init(self.datafile, self.resultfile)
        self._init(self.datafile_pqn, self.resultfile_pqn)
        self.models = models

    
    def _init(self, datafile, resultfile):
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        if not os.path.exists(resultfile):
            os.makedirs(resultfile)

    def _random_partition(self):
        # partition the first k nodes into h_selected groups and the rest into h_rest groups
        assert self.h_selected <= self.k, "h_selected should be less than k"
        assert self.h_rest <= self.d - self.k, "h_rest should be less d-k"
        if self.k % self.h_selected != 0 or (self.d - self.k) % self.h_rest != 0:
            break_points = np.sort(random.sample(range(1, self.k), self.h_selected-1))
            break_points_rest = np.sort(random.sample(range(self.k+1, self.d), self.h_rest-1))
        else: # evenly divide the nodes
            break_points = np.arange(self.k // self.h_selected, self.k, self.k // self.h_selected)
            break_points_rest = np.arange(self.k + (self.d - self.k) // self.h_rest, self.d, (self.d - self.k) // self.h_rest)
            
        return break_points, break_points_rest
    
    def _generate_clusters(self):
        if self.h_selected == 1 and self.h_rest == 1:
            clusters = [np.arange(self.k)]
            clusters.append(np.arange(self.k, self.d))
        else:
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

        A = sp.lil_matrix((self.d, self.d))
        
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
        return L, clusters, A
    
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
        # w = w[:, np.newaxis]
        return w

    def _generate_X(self):
        # X = np.random.normal(0, 1, (self.n, self.d))
        mean = np.zeros(self.d)
        cov = np.eye(self.d)
        X = np.random.multivariate_normal(mean, cov, self.n)
        return X
    
    def _generate_y(self, X, w):
        signal = X @ w
        noise = np.random.normal(0, self.gamma, signal.shape)
        y = signal + noise
        SNR = self._compute_snr(signal, noise)
        if 0: # TODO: add a debug/verbose flag
            print(f'SNR: {SNR}')

        return y

    def _generate_data(self):
        L, clusters, A = self._generate_graph()
        if 0: # TODO: add a debug/verbose flag 
            self._visualize_graph(A)
        w = self._generate_w(clusters)
        X = self._generate_X()
        y = self._generate_y(X, w)
        return L, w, X, y, clusters, A


    def _compute_snr(self, signal, noise):  
        # in our case, the snr is 10*log10(1/gamma^2)
        signal = np.asarray(signal)
        noise = np.asarray(noise)

        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
        
        return snr_db
        
    def _max_degree(self, L):
        # find the maximum degree of the graph according to the laplacian matrix
        return np.max(np.diag(L.toarray()))
    
    def solver(self, model, X, y, clusters=None, L=None, A=None, i=None, rho=1, mu=1):
        if model == "Proximal":
            return self._solver_proximal(X, y, A, i)
        elif model == "Lasso":
            rho = np.sqrt(self.n) # TODO: check if this is the correct value
            return self._solver_gfl(X, y, L, i, rho, mu=0.0)
        elif model == "GFL_Matlab":
            return self._solver_gfl(X, y, L, i, rho=np.sqrt(self.n), mu=0.01)
        elif model == "GFL_normalized":
            L = self._normalize_laplacian(L)
            return self._solver_gfl(X, y, L, i, rho=1.0, mu=1.0)
        elif model == "Lasso_Sklearn":
            return self._solver_lasso_sklearn(X, y)
        elif model == "Adaptive_Grace":
            return self._solver_aGrace(X, y, A.toarray())
        else:
            raise ValueError("Model not supported")
        
    def _normalize_laplacian(self, L):
        # normalize the laplacian matrix
        D = np.array(L.diagonal())
        D_inv_sqrt = sp.diags(1 / np.sqrt(D))
        L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
        return L_normalized
    
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
    
    def _solver_gfl(self, X, y, L, i, rho=None, mu=0.01, k=None):
        datafile_pqn = os.path.join(self.datafile_pqn, f'data_{i}.mat')
        resultfile_pqn = os.path.join(self.resultfile_pqn, f'result_{i}.mat')
        self._save_mat_pqn(X, y, L, datafile_pqn)
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
        if y.ndim == 1:
            y = y[:, np.newaxis]
        data = {
            "X": X,
            "y": y,
            "L": L.toarray() if sp.issparse(L) else L,  # we store the adjacency matrix as dense matrix
        }
        sio.savemat(filename, data)

    
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

    def _cross_validation(self, X, y, A, rho1_values, rho2_values, k_folds=5):
        results = []
        n = self.n
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)
        datafile = os.path.join(self.datafile, 'data.mat') 
        resultfile = os.path.join(self.resultfile, 'result.mat')

        for rho1, rho2 in itertools.product(rho1_values, rho2_values):
            mse_list = []
            for fold in folds:
                train_indices = np.setdiff1d(indices, fold)
                test_indices = fold
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]

                self._save_mat(X_train, y_train, A, datafile)
                self._call_proximal(datafile, resultfile, rho1, rho2)
                u, funcVal = self._read_result(resultfile)      
                mse = np.mean((X_test @ u - y_test) ** 2)
                mse_list.append(mse)

            avg = np.mean(mse_list)
            results.append((rho1, rho2, avg))

        best_rho1, best_rho2, _ = min(results, key=lambda x: x[2])
        return best_rho1, best_rho2
    
    def _cross_validation_gfl(self, X, y, L, rho_values, mu_values, k_folds=5, k=None):
        results = []
        n = self.n
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)
        datafile = os.path.join(self.datafile_pqn, 'data.mat')
        resultfile = os.path.join(self.resultfile_pqn, 'result.mat')

        for rho, mu in itertools.product(rho_values, mu_values):
            acc_list = []
            for fold in folds:
                train_indices = np.setdiff1d(indices, fold)
                test_indices = fold
                X_train, y_train = X[train_indices], y[train_indices]
                # X_test, y_test = X[test_indices], y[test_indices]
                self._save_mat_pqn(X_train, y_train, L, datafile)
                self._call_gfl(datafile, resultfile, rho, mu, k)
                u, _ = self._read_result(resultfile)
                u = u.flatten()
                acc_score = self.recovery_accuracy(u)
                acc_list.append(acc_score)
            
            avg = np.mean(acc_list)
            results.append((rho, mu, avg))
            print(f"rho: {rho}, mu: {mu}, acc: {avg}")
    
        best_rho, best_mu, _ = max(results, key=lambda x: x[2])
        # print(f"Best rho: {best_rho}, Best mu: {best_mu}")
        return best_rho, best_mu

    def _read_result(self, resultfile):
        result = sio.loadmat(resultfile)
        beta, funcVal = result['beta'], result['funcVal']
        return beta, funcVal
        
    def _solver_proximal(self, X, y, A, i):
        rho1_values = [0.1, 0.5, 1.0, 5.0]
        rho2_values = [0.1, 0.5, 1.0, 5.0]
        if i == 0: # we only do cross validation once and use the best rho1 and rho2 for the rest of the replications
            self.best_rho1, self.best_rho2 = self._cross_validation(X, y, A, rho1_values, rho2_values)
            print(f"Best rho1: {self.best_rho1}, Best rho2: {self.best_rho2}")

        datafile_name = os.path.join(self.datafile, f'data_{i}.mat')
        resultfile_name = os.path.join(self.resultfile, f'result_{i}.mat')
        self._save_mat(X, y, A, datafile_name)
        self._call_proximal(datafile_name, resultfile_name, self.best_rho1, self.best_rho2)
        u, funcVal = self._read_result(resultfile_name)
        return u.flatten() # the original return a vector with shape (d,1), will not work with recovery_accuracy
    
    def _adaptive_laplacian(self, L, beta_tilde):
        """Modify the Laplacian matrix to incorporate sign adjustments."""
        sign_beta = np.sign(beta_tilde)
        L_star = L * (sign_beta[:, None] @ sign_beta[None, :])
        return L_star
    
    def _soft_thresholding(self, z, gamma):
        return np.sign(z) * max(abs(z) - gamma, 0)
    

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

    def recovery_accuracy(self, u):
        # evaluate the support recovery accuracy
        # we take top k for u
        selected_features_true = np.arange(self.k)
        selected_features_pred = np.argsort(np.abs(u))[-self.k:] # take absolute value for proximal method
        correct_pred = np.intersect1d(selected_features_true, selected_features_pred)
        accuracy = len(correct_pred) / self.k
        return accuracy
    
    def _save_mat(self, X, y, A, filename=None):
        # save the data to .mat file so that the matlab code of proxiaml can use it
        if y.ndim == 1:
            y = y[:, np.newaxis]
        data = {
            "X": X,
            "y": y,
            "AdjMat": A.toarray() if sp.issparse(A) else A,  # we store the adjacency matrix as dense matrix
        }
        sio.savemat(filename, data)

    def _report(self, model_accuracy):
        for model, accuracy in model_accuracy.items():
            avg_accuracy = np.mean(accuracy)
            std_accuracy = np.std(accuracy)
            print(f"Model: {model}, Avg. Accuracy: {avg_accuracy}, Std. Accuracy: {std_accuracy}")

    def runtime(self):
        """
        Measures runtime for different methods and generates a formatted table.
        """
        if self.models is None:
            models = ["GFL_Matlab", "Lasso", "Proximal", "Lasso_Sklearn", "Adaptive_Grace"]
        else:
            models = self.models
        
        runtime_results = {model: [] for model in models}  # Store runtimes

        for i in range(self.num_replications):
            L, w, X, y, clusters, A = self._generate_data()
            
            for model in models:
                print(f"Running {model} on replication {i+1}/{self.num_replications}...")

                start_time = time.time()  # Start timer
                u = self.solver(model, X, y, clusters, L, A, i)  # Solve
                end_time = time.time()  # End timer

                runtime_results[model].append(end_time - start_time)  # Store runtime
            
            print(f"Replication {i+1} completed.")

        # Compute mean and standard deviation for each method
        runtime_summary = {model: (np.mean(times), np.std(times)) for model, times in runtime_results.items()}
        return runtime_summary

    def out_of_sample(self, ):
        k_values = np.arange(30, 110, 10)
        mse_results = defaultdict(list)
        for i in range(self.num_replications):
            print(f"Running replication {i+1}/{self.num_replications}...")
            # Generate synthetic dataset
            L, w, X, y, clusters, A = self._generate_data()
            for k in k_values:
                # Solve the original problem with given k
                u = self._solver_gfl(X, y, L, i, rho=np.sqrt(self.n), mu=0.01,k=k)  

                # Select top k features
                selected_features = np.argsort(np.abs(u))[-k:]

                # Solve the subproblem with selected features
                X_sub = X[:, selected_features]
                linear_model = Lasso(alpha=0.5)
                linear_model.fit(X_sub, y)
                y_pred = linear_model.predict(X_sub)

                # Compute MSE
                mse = np.mean((y - y_pred) ** 2)
                mse_results[k].append(mse)

                print(f"Replication {i+1} with value {k} completed.")

        return mse_results


    # ------------------------------------------------------------------
    #  Patched main() with inner tqdm bars
    # ------------------------------------------------------------------
    def main(self, show_progress: bool = True):
        """
        Run `num_replications × len(models)` experiments.
        Returns
        -------
        dict : {model: [accuracy_rep1, … , accuracy_repR]}
        """
        models = self.models or [
            "GFL_Matlab", "Lasso", "Proximal",
            "Lasso_Sklearn", "Adaptive_Grace"
        ]

        model_accuracy = defaultdict(list)

        # unique console position for each worker
        proc       = multiprocessing.current_process()
        worker_idx = (proc._identity[0] - 1) if proc._identity else 0
        rep_pos    = worker_idx * 2 + 1      # first bar line
        model_pos  = worker_idx * 2 + 2      # nested bar line

        rep_iter = range(self.num_replications)
        if show_progress:
            rep_iter = tqdm(rep_iter, desc=f"[n={self.n}] reps",
                            position=rep_pos, leave=False)

        for i in rep_iter:
            L, w, X, y, clusters, A = self._generate_data()

            model_iter = models
            if show_progress:
                model_iter = tqdm(model_iter,
                                  desc=f"rep {i+1}/{self.num_replications}",
                                  position=model_pos, leave=False)

            for model in model_iter:
                u        = self.solver(model, X, y, clusters, L, A, i)
                acc      = self.recovery_accuracy(u)
                model_accuracy[model].append(acc)

        return model_accuracy
    

class RandomEnsembleWeight(RandomEnsemble):
    def _generate_w(self, clusters):
        w = np.zeros(self.d)
        cluster_weights = []

        # Generate non-zero random weights for clusters
        for _ in range(self.h_selected):
            cluster_sign = np.random.choice([-1, 1])
            cluster_weight = cluster_sign * (1 / np.sqrt(self.k))
            cluster_weights.append(cluster_weight)

        # Generate zero weights for the rest of the clusters
        for _ in range(self.h_rest):
            cluster_weights.append(0)

        for i in range(self.h_total):
            for node in clusters[i]:
                w[node] = cluster_weights[i] + np.random.normal(0, 0.1* np.abs(cluster_weights[i]))  # we allow weights in the same cluster to be slightly different

        return w
    
class RandomEnsembleCorrelation(RandomEnsemble):
    def _generate_X(self):
        mean = np.zeros(self.d)
        cov = np.eye(self.d)
        # for the correlated case, we find some correlation between the selected features and non-selected features
        correlated_ratio = 0.3
        selected_features = np.arange(self.k)
        non_selected_features = np.arange(self.k, self.d)

        correlated_pairs = []
        selected_features_correlated = random.sample(list(selected_features), int(correlated_ratio * self.k))
        non_selected_features_correlated = random.sample(list(non_selected_features), int(correlated_ratio * self.k))
        for i in selected_features_correlated:
            for j in non_selected_features_correlated:
                correlated_pairs.append((i, j))
        
        for i, j in correlated_pairs:
            cov[i, j] = 0.9
            cov[j, i] = 0.9

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)  # Set negative eigenvalues to zero
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        X = np.random.multivariate_normal(mean, cov, self.n)
        return X
    
class RandomEnsembleCorrelationWeight(RandomEnsemble):
    def _generate_w(self, clusters):
        w = np.zeros(self.d)
        cluster_weights = []

        # Generate non-zero random weights for clusters
        for _ in range(self.h_selected):
            cluster_weight = (1 / np.sqrt(self.k))
            cluster_weights.append(cluster_weight)

        # Generate zero weights for the rest of the clusters
        for _ in range(self.h_rest):
            cluster_weights.append(0)

        for i in range(self.h_total):
            for node in clusters[i]:
                node_sign = np.random.choice([-1, 1])
                w[node] = node_sign * cluster_weights[i] + np.random.normal(0, 0.01)  # we allow weights in the same cluster to be slightly different

        return w
    
    def _generate_X(self):
        mean = np.zeros(self.d)
        cov = np.eye(self.d)
        # for the correlated case, we find some correlation between the selected features and non-selected features
        correlated_ratio = 0.3
        selected_features = np.arange(self.k)
        non_selected_features = np.arange(self.k, self.d)

        correlated_pairs = []
        selected_features_correlated = random.sample(list(selected_features), int(correlated_ratio * self.k))
        non_selected_features_correlated = random.sample(list(non_selected_features), int(correlated_ratio * self.k))
        for i in selected_features_correlated:
            for j in non_selected_features_correlated:
                correlated_pairs.append((i, j))
        
        for i, j in correlated_pairs:
            cov[i, j] = 0.9
            cov[j, i] = 0.9

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)  # Set negative eigenvalues to zero
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        X = np.random.multivariate_normal(mean, cov, self.n)
        return X

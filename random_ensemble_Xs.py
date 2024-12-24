import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time

from L0Obj_Xs import L0Obj_Xs
from data_generator import generate_synthetic_data_with_graph, read_synthetic_data, save_synthetic_data
from ProjectOperator_Xs import ProjOperator_Xs
from minConf.minConf_PQN_mod import minConF_PQN
import random
from utils.vec_mat import vector_to_symmetric


tStart = time.process_time()

# Parameters
n = 1000  # Number of samples
d = 100   # Number of features
h_total = 5    # number of cluster in the graph
h = 3 # number of cluster that are selected i.e. related to the dependent variable
nVars = d * (d+1) // 2
inter_cluster = 0.95 # probability of inter-cluster edges in graph
outer_cluster = 0.0 # probability of outer-cluster edges in graph
gamma = 1.5  # Noise standard deviation
mu = 1
SNR = 1

fixed_seed = 0
random_rounding = 0
connected = False
correlated = True
random_graph = True
visualize = True

if fixed_seed:
    file_path = "data/synthetic_data.pkl"
    print("Fixed seed is enabled. Reading synthetic data from file.")
    # Read synthetic data from the saved files
    X, w_true, y, adj_matrix, L, clusters_true, k = read_synthetic_data(file_path, visualize=visualize)
else:
    print("Fixed seed is disabled. Generating synthetic data.")
    # Generate synthetic data
    X, w_true, y, adj_matrix, L, clusters_true, k = generate_synthetic_data_with_graph(
        n, d, h_total, h,
        inter_cluster=inter_cluster,
        outer_cluster=outer_cluster,
        gamma=gamma,
        visualize=visualize,
        connected=connected,
        random=random_graph
    )
    
    # Save synthetic data to files for future use
    file_path = "data/synthetic_data.pkl"
    save_synthetic_data(file_path, X, w_true, y, adj_matrix, L, clusters_true, k)
    print("Synthetic data has been saved.")

clusters_size = [len(cluster) for cluster in clusters_true]
clusters_size.sort()

# find the largest/smallest degree of the node
max_degree = max(L.diagonal())
min_degree = min(L.diagonal())
# compute rho that make sure the exactness of the solution 
rho = 5 * mu * k * (max_degree - min_degree)

print("Check!!!")
tEnd = time.process_time() - tStart
print("Execution time (generating the data):", tEnd)

# initial guess of parameters
Xs_initial = np.zeros((nVars, 1))


m_ground_truth = np.zeros((d, h))
for c, cluster in enumerate(clusters_true[:h]):
    for i in cluster:
        m_ground_truth[i, c] = 1

# Set up Objective Function L0Obj(X, m, y, L, rho, mu, d, h, n)::
funObj = lambda m: L0Obj_Xs(X, m, y, L, rho, mu, d, h, n)

# Set up Simplex Projection Function ProjOperator_Gurobi(m, k, d, h):
funProj = lambda m: ProjOperator_Xs(m, d, k)

tEnd = time.process_time() - tStart
print("Execution time(Before):", tEnd)
print("start!!!")

# Solve with PQN
options = {'maxIter': 100, 'verbose': 2}
tStart = time.process_time()
Xs_out, obj, _ = minConF_PQN(funObj, Xs_initial, funProj, options)
print(f"Xs_out: {Xs_out}")

# save the result to a file 
sio.savemat('Xs_out.mat', {'Xs_out': Xs_out})
tEnd = time.process_time() - tStart


# extract the predicted features Xs_ii
Xs = vector_to_symmetric(Xs_out, d)
Xs_ii = np.diag(Xs)
# get the first k largest elements
selected_features_predict = np.argsort(Xs_ii)[-k:]


# find the clusters of the selected features
clusters_predict = {}
cluster_number = 0
already_assigned = set() # keep track of the features that we find the its buddy
for i in selected_features_predict:
    # Xs_ij = 1 implies feature i and feature j are in the same cluster
    if i in already_assigned:
        continue
    already_assigned.add(i)
    clusters_predict[cluster_number] = [i]
    Xs_ij = Xs[i, :]
    # find the index of the features that are in the same cluster
    buddy = np.where(Xs_ij == 1)[0].tolist()
    if len(buddy) == 0:
        continue
    for b in buddy:
        print("b", b)
        if b in already_assigned:
            print("Error: feature", b, "is already assigned to a cluster")
        else:
            clusters_predict[cluster_number].append(b)
            already_assigned.add(b)

    cluster_number += 1



tEnd = time.process_time() - tStart

# find the intersection of the selected features and clusters
selected_features_true = np.arange(k)
C = np.intersect1d(selected_features_true, selected_features_predict)

# Find the intersection
AccPQN = len(C) / k

print("Execution time:", tEnd)
selected_features_predict.sort()
selected_features_true.sort()
print("Accuracy of PQN:", AccPQN)
# print("clusters_predict", clusters_predict_without_order)
print("clusters_true", clusters_true[:h])
print("clusters_predict", clusters_predict) 
print("selected_features_predict", selected_features_predict)
print("selected_features_true", selected_features_true)

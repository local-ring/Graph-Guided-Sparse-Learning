import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time

from L0Obj import L0Obj, L0Obj_separate
from data_generator import generate_synthetic_data_with_graph, read_synthetic_data_from_file, save_synthetic_data_to_file
from ProjectOperator import ProjOperator_Gurobi
from minConf.minConf_PQN import minConF_PQN
import random


tStart = time.process_time()
# Generate synthetic data
# Parameters
n = 1000  # Number of samples
d = 500   # Number of features
k = 20  # Number of non-zero features
h_total = 40    # number of cluster in the graph
h = 5 # number of cluster that are selected i.e. related to the dependent variable
nVars = d*h # Number of Boolean variables in m
inter_cluster = 1 # probability of inter-cluster edges in graph
outer_cluster = 0 # probability of outer-cluster edges in graph
gamma = 1.5  # Noise standard deviation
pho = np.sqrt(n)
mu = 0.1

SNR = 1

fixed_seed = 0
random_rounding = 0
connected = False
correlated = True
random_graph = True
visualize = True
# read a fixed synthetic data from a file if fixed_seed is True because we want to compare the results with the original results
if fixed_seed:
    file_path = "data/synthetic_data.npz"
    X, w_true, y, adj_matrix, L, clusters_true, selected_features_true = read_synthetic_data_from_file(file_path)
else:
    # Generate synthetic data
    X, w_true, y, adj_matrix, L, clusters_true, k = generate_synthetic_data_with_graph(n, d, 
                                                                                      h_total, h, 
                                                                                      inter_cluster=inter_cluster, 
                                                                                      outer_cluster=outer_cluster, 
                                                                                      gamma=gamma,
                                                                                      visualize=visualize, 
                                                                                      connected=connected, 
                                                                                      correlated=correlated, 
                                                                                      random=random_graph)
    # Save the synthetic data to a file
    file_path = "synthetic_data.npz"
    # save_synthetic_data_to_file(file_path, X, w_true, y, adj_matrix, L, clusters_true, selected_features_true)

    # print("selected_features_true", selected_features_true)
    # print("clusters_true", clusters_true)
# L = - adj_matrix
clusters_size = [len(cluster) for cluster in clusters_true]
clusters_size.sort()
C = (2 * clusters_size[-1] +  outer_cluster * (d - clusters_size[0]- clusters_size[1])) + 2 * d
# we need to modify the matrix X to define the objective function
X_hat = np.repeat(X, h, axis=1)
# print("w_true:", w_true) 

print("Check!!!")
tEnd = time.process_time() - tStart
print("Execution time (generating the data):", tEnd)

# Initial guess of parameters
# m_initial = np.ones((nVars, 1)) * (1 / nVars)
# m_initial = np.random.normal(0, 1, (nVars, 1))
# m_initial = np.random.normal(0, 1/nVars, (nVars, 1))
# m_initial = np.zeros((nVars, 1))
m_initial = np.ones((nVars, 1)) * (k / nVars)
# m_initial[0:k] = 1

# Set up Objective Function L0Obj(X, m, y, L, rho, mu, d, h, n)::
funObj = lambda m: L0Obj(X_hat, m, y, L, pho, mu, d, h, n, C)

# Set up Simplex Projection Function ProjOperator_Gurobi(m, k, d, h):
funProj = lambda m: ProjOperator_Gurobi(m, k, d, h)

funObj_separate = lambda m: L0Obj_separate(X_hat, m, y, L, pho, mu, d, h, n)

tEnd = time.process_time() - tStart
print("Execution time(Before):", tEnd)
print("start!!!")
# Solve with PQN
options = {'maxIter': 100, 'verbose': 3}
tStart = time.process_time()
mout, obj, _ = minConF_PQN(funObj, m_initial, funProj, options)
print(f"uout: {mout}")
print(f"m_sum: {np.sum(mout)}, m_featuress_sum: {np.sum(mout.reshape(d, h), axis=1)}, m_clusters_sum: {np.sum(mout.reshape(d, h), axis=0)}")
# save the result to a file 

m_ground_truth = np.zeros((d, h))
for c, cluster in enumerate(clusters_true[:h]):
    for i in cluster:
        m_ground_truth[i, c] = 1

f, g, graph_penalty, precision_penalty, A_grad, B_grad = funObj_separate(m_ground_truth.flatten())
print(f"obj: {f}, graph_penalty_gt: {graph_penalty}, precision_penalty_gt: {precision_penalty}")


f_init, g_init, graph_penalty_init, precision_penalty_init, A_grad_init, B_grad_init = funObj_separate(m_initial)
f, g, graph_penalty, precision_penalty, A_grad, B_grad = funObj_separate(mout)
print(f"obj: {f}, graph_penalty_init: {graph_penalty_init}, precision_penalty_init: {precision_penalty_init}")
print(f"obj: {f}, graph_penalty: {graph_penalty}, precision_penalty: {precision_penalty}")
sio.savemat('mout.mat', {'mout': mout})
tEnd = time.process_time() - tStart


if random_rounding:
    # round the result randomly according to its value for a few times and select the best one in terms of the objective function
    T = 1000
    min_obj = np.inf
    min_round = np.zeros((nVars, ))  # initialize the best result
    m_grouped = mout.reshape(d, h)
    """
    here we propose a different way to round this result. note that we have introduce h times more variables than the original problem. so the number are diluted, it is hard to get more than a lot of ones in the result if we round it according to its probability. because even the selected one are very small.

    so we need to, first, treat variables associated with the same feature as a whole, take the sum, and then round it according to the probability to determine whether this feature is selected or not. Then, for the selected features, we need to determine which cluster it belongs to according to the probability.
    """
    for _ in range(T):
        one_realization = np.zeros((d, h))
        feature_round = (np.random.rand(d)<np.sum(m_grouped, axis=1)).astype(int)

        for i in range(d):
            if feature_round[i] == 1:
                cluster = np.random.choice(h, p=m_grouped[i]/np.sum(m_grouped[i])) # TODO: another choice is to introduce a temperature parameter to control the randomness or an extra "discard" choice with the 1- sum(m_grouped[i]) probability
                one_realization[i, cluster] = 1

        m_round = one_realization.flatten()
        obj = funObj(m_round)[0]
        if obj < min_obj:
            min_obj = obj
            min_round = m_round

    selected_features_predict = []
    clusters_predict = {}
    # parse the result m_round to the selected features and clusters
    m = min_round.reshape(d, h)
    for i in range(d):
        if np.sum(m[i]) > 0:
            selected_features_predict.append(i)
            cluster = np.where(m[i] > 0)[0][0]
            if cluster in clusters_predict:
                clusters_predict[cluster].append(i)
            else:
                clusters_predict[cluster] = [i]

else:
    m_grouped = mout.reshape(d, h)
    feature_prob = np.sum(m_grouped, axis=1)
    feature_rank = np.argsort(np.abs(feature_prob))[::-1]
    selected_features_predict = feature_rank[:k]
    clusters_predict = {i: [] for i in range(h)}
    for i in selected_features_predict:
        cluster = np.argmax(m_grouped[i])
        clusters_predict[cluster].append(i)


tEnd = time.process_time() - tStart

# find the intersection of the selected features and clusters
selected_features_true = np.arange(k)
C = np.intersect1d(selected_features_true, selected_features_predict)

# Find the intersection
AccPQN = len(C) / k

# sort the dict for clear comparison
print("clusters_predict", clusters_predict)
# clusters_predict_without_order = [clusters_predict[i] for i in range(h)] if clusters_predict else []
# for cluster in clusters_predict_without_order:
#     cluster.sort()
# clusters_predict_without_order.sort(key=lambda x: x[0])
# for cluster in clusters_true:
#     cluster.sort()
# clusters_true.sort(key=lambda x: x[0])

print("Execution time:", tEnd)
selected_features_predict.sort()
selected_features_true.sort()
print("Accuracy of PQN:", AccPQN)
# print("clusters_predict", clusters_predict_without_order)
print("clusters_true", clusters_true[:h])
print("clusters_predict", clusters_predict) 
print("selected_features_predict", selected_features_predict)
print("selected_features_true", selected_features_true)

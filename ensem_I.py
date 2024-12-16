import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time

from L0Obj import L0Obj
from data_generator import generate_synthetic_data_with_graph, read_synthetic_data_from_file, save_synthetic_data_to_file
from ProjectOperator import ProjOperator_Gurobi
from minConf.minConf_PQN import minConF_PQN
import random



def experiment(n, d, k, h, theta, gamma, pho, mu, fixed_seed, random_rounding, connected, correlated, random_graph, visualize, maxIter=50):
    nVars = d*h # Number of Boolean variables in m
    tStart = time.process_time()
    # Generate synthetic data
    # Parameters

    # read a fixed synthetic data from a file if fixed_seed is True because we want to compare the results with the original results
    if fixed_seed:
        file_path = "synthetic_data.npz"
        X, w_true, y, adj_matrix, L, clusters_true, selected_features_true = read_synthetic_data_from_file(file_path)
    else:
        # Generate synthetic data
        X, w_true, y, adj_matrix, L, clusters_true, selected_features_true = generate_synthetic_data_with_graph(n, d, k, h, theta, gamma, visualize=visualize, connected=connected, correlated=correlated, random=random_graph)
        clusters_true = [np.array(cluster) for cluster in clusters_true]  # Ensure clusters are arrays
        selected_features_true = np.array(selected_features_true)  # Ensure selected_features is an array
        # print("selected_features_true", selected_features_true)
        # print("clusters_true", clusters_true)
        # print("Type of clusters:", type(clusters_true))
        # print("Contents of clusters:", clusters_true)
        # print("Type of each cluster:", [type(cluster) for cluster in clusters_true])
        # print("Lengths of clusters:", [len(cluster) for cluster in clusters_true])
        # Save the synthetic data to a file
        file_path = "synthetic_data.npz"
        # save_synthetic_data_to_file(file_path, X, w_true, y, adj_matrix, L, clusters_true, selected_features_true)

        # print("selected_features_true", selected_features_true)
        # print("clusters_true", clusters_true)

    # we need to modify the matrix X to define the objective function
    X_hat = np.repeat(X, h, axis=1)
    # print("w_true:", w_true) 

    print("Check!!!")
    tEnd = time.process_time() - tStart
    print("Execution time (generating the data):", tEnd)

    # Initial guess of parameters
    # m_initial = np.ones((nVars, 1)) * (1 / nVars)
    m_initial = np.random.normal(0, 1, (nVars, 1))


    # Set up Objective Function L0Obj(X, m, y, L, rho, mu, d, h)::
    funObj = lambda m: L0Obj(X_hat, m, y, L, pho, mu, d, h)

    # Set up Simplex Projection Function ProjOperator_Gurobi(m, k, d, h):
    funProj = lambda m: ProjOperator_Gurobi(m, k, d, h)

    tEnd = time.process_time() - tStart
    print("Execution time(Before):", tEnd)
    print("start!!!")
    # Solve with PQN
    options = {'maxIter': maxIter, 'verbose': 1}
    tStart = time.process_time()
    mout, obj, _ = minConF_PQN(funObj, m_initial, funProj, options)
    # print(f"mout: {mout}")
    # save the result to a file 
    sio.savemat('mout.mat', {'mout': mout})
    tEnd = time.process_time() - tStart


    if random_rounding:
        # round the result randomly according to its value for a few times and select the best one in terms of the objective function
        T = 2000
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
        clusters_predict = {}
        for i in selected_features_predict:
            cluster = np.argmax(m_grouped[i])
            if cluster in clusters_predict:
                clusters_predict[cluster].append(i)
            else:
                clusters_predict[cluster] = [i]

    tEnd = time.process_time() - tStart

    # find the intersection of the selected features and clusters
    C = np.intersect1d(selected_features_true, selected_features_predict)

    # Find the intersection
    AccPQN = len(C) / k

    # sort the dict for clear comparison
    print("clusters_predict", clusters_predict)
    clusters_predict_without_order = [clusters_predict[i] for i in range(h)]
    for cluster in clusters_predict_without_order:
        cluster.sort()
    clusters_predict_without_order.sort(key=lambda x: x[0])
    for cluster in clusters_true:
        cluster.sort()
    clusters_true.sort(key=lambda x: x[0])

    print("Execution time:", tEnd)
    selected_features_predict.sort()
    selected_features_true.sort()
    print("Accuracy of PQN:", AccPQN)
    # print("clusters_true", clusters_true)
    # print("clusters_predict", clusters_predict_without_order)
    # print("selected_features_true", selected_features_true)
    # print("selected_features_predict", selected_features_predict)




    return mout, X, y, L, clusters_true, selected_features_true, clusters_predict, selected_features_predict
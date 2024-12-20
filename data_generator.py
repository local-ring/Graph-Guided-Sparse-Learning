import numpy as np
import random
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle


def add_noise_to_signal(signal, desired_snr_db):
    # Convert signal to numpy array
    signal = np.asarray(signal)
    # Calculate signal power
    signal_power = np.mean(signal ** 2)
    # Calculate the desired noise power based on the desired SNR (in dB)
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    noise_power = signal_power / desired_snr_linear
    # Generate noise with the calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal, noise
 
 
def compute_snr(signal, noise):
    # Convert signal and noise to numpy arrays for calculations
    signal = np.asarray(signal)
    noise = np.asarray(noise)
 
    # Calculate the power of the signal and the noise
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    # Calculate the SNR
    snr = signal_power / noise_power
    # Convert SNR to decibels (dB)
    snr_db = 10 * np.log10(snr)
    return snr_db


# Define colors for each cluster, and use gray for unselected nodes
def visualize_graph(G, clusters, selected_features):
    # Get distinct colors for clusters using matplotlib's Tableau colors
    unique_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = unique_colors[:len(clusters)]  # Select enough colors for each cluster
    unselected_color = "gray"  # Color for nodes not in any cluster

    # Create a color mapping for each node
    node_colors = []
    for node in range(G.number_of_nodes()):
        if node in selected_features:
            # Find which cluster the node belongs to
            for i, cluster in enumerate(clusters):
                if node in cluster:
                    node_colors.append(colors[i % len(colors)])
                    break
        else:
            # If the node is not selected, color it gray
            node_colors.append(unselected_color)

    # Draw the graph with cluster-based colors and gray for unselected nodes
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistent visualization
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
    plt.title("Generated Graph Structure with Cluster Colors and Unselected Nodes in Gray")
    plt.show()

def read_synthetic_data_from_file(file_path):
    print("reading the synthetic data from the file", file_path)
    with np.load(file_path, allow_pickle=True) as data:
        X = data["X"]
        w = data["w"]
        y = data["y"]
        clusters = data["clusters"]
        selected_features = data["selected_features"]
    adj_matrix = sp.load_npz(file_path.replace(".npz", "_adj_matrix.npz"))
    laplacian_matrix = sp.load_npz(file_path.replace(".npz", "_laplacian_matrix.npz"))
    # print(X.shape, w.shape, y.shape, adj_matrix.shape, laplacian_matrix.shape, clusters.shape, selected_features.shape)
    return X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features

def save_synthetic_data_to_file(file_path, X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features):
    np.savez(file_path, X=X, w=w, y=y, clusters=clusters, selected_features=selected_features, allow_pickle=True)
    sp.save_npz(file_path.replace(".npz", "_adj_matrix.npz"), adj_matrix) # we need to save the sparse matrix separately
    sp.save_npz(file_path.replace(".npz", "_laplacian_matrix.npz"), laplacian_matrix)

    print("synthetic data saved to the file", file_path)


def random_partition(d, h_total):
    """
    Randomly partition `d` into `h_total` parts, ensuring the sum equals `d`.
    
    Parameters:
    - d (int): Total number to partition.
    - h_total (int): Number of parts/clusters.

    Returns:
    - list: A list of breakpoints that divide the range [0, d] into `h_total` parts.
    """
    if d < h_total:
        raise ValueError("The number of clusters exceeds the total number.") 
    # Generate `h_total - 1` random breakpoints in the range [0, d]
    breakpoints = sorted(np.random.choice(range(1, d), h_total - 1, replace=False))
    
    # Add the boundaries (0 and d) to the breakpoints
    breakpoints = [0] + breakpoints + [d] # each interval is [breakpoints[i], breakpoints[i+1]) including the left boundary but excluding the right boundary

    for i in range(h_total):
        print(f"cluster {i} contains features from {breakpoints[i]} to {breakpoints[i + 1] - 1}, with size {breakpoints[i + 1] - breakpoints[i]}")
    
    return breakpoints

def generate_random_graph(d, h_total, h, inter_cluster_prob, outer_cluster_prob, connected=False, visualize=True):
    # by symmetry, wlog, the first d_1 features are in the first cluster, the next d_2 features are in the second cluster, and so on
    # we need to make sure that d_1 + d_2 + ... + d_h = d, so we can implement this by random partitioning of d into h_total parts

    # randomly partition d into h_total parts
    breakpoints = random_partition(d, h_total)
    clusters = [range(breakpoints[i], breakpoints[i + 1]) for i in range(h_total)]

    # again, without loss of generality, we can assume that the first h clusters are the selected clusters
    selected_clusters = clusters[:h]
    k = selected_clusters[-1][-1]   # the last feature in the last selected cluster
    print("number of selected/non-zero features:", k)

    # create the adjacency matrix
    adj_matrix = sp.lil_matrix((d, d), dtype=int)
    
    # generate the inter-cluster part of the adjacency matrix
    for cluster in clusters:
        size = len(cluster)
        block = (np.random.rand(size, size) < inter_cluster_prob).astype(int)
        np.fill_diagonal(block, 0) # no self-loops
        block = np.triu(block) + np.triu(block, 1).T # make the matrix symmetric
        for i, node_i in enumerate(cluster):
            for j, node_j in enumerate(cluster):
                adj_matrix[node_i, node_j] = block[i, j]

    # generate the outer-cluster part of the adjacency matrix
    for i in range(h_total):
        for j in range(i + 1, h_total):
            cluster_i = clusters[i]
            cluster_j = clusters[j]
            block = (np.random.rand(len(cluster_i), len(cluster_j)) < outer_cluster_prob).astype(int)
            for m, node_i in enumerate(cluster_i):
                for n, node_j in enumerate(cluster_j):
                    adj_matrix[node_i, node_j] = block[m, n]
                    adj_matrix[node_j, node_i] = block[m, n]

    # TODO: ensure the graph is connected

    if not visualize:
        plt.figure(figsize=(8, 8))
        plt.spy(adj_matrix, markersize=1)
        plt.title("Generated Adjacency Matrix")
        plt.axis("off")
        plt.show()

    G = nx.from_scipy_sparse_array(adj_matrix)  
    print("Number of connected components:", nx.number_connected_components(G))

    return adj_matrix, clusters, k

def generate_weight(d, selected_clusters, k):
    """
    we need to find a different way to assign the value of the weights because we want different weights for each cluster, what i am concerned is that
    if they only have two values, it's highly likely that the weights will be the same for most of the clusters, which will confuse the model
    """
    w = np.zeros(d)
    for i, cluster in enumerate(selected_clusters):
        sign = np.random.choice([-1, 1])
        # feature_weight = np.random.normal(1/np.sqrt(k), 1) * sign
        feature_weight = 1.0 * sign
        for feature in cluster:
            w[feature] = feature_weight
        print(f"cluster {i}: {cluster}, feature_weight: {feature_weight}")
    return w

def generate_graph(d, h_total, h, inter_cluster=0.9, outer_cluster=0.05, connected=False, visualize=True, random=True):
    correlated_pairs = []
    if random:
        adj_matrix, clusters, k = generate_random_graph(d, h_total, h, 
                                                        inter_cluster, 
                                                        outer_cluster, 
                                                        connected=connected, 
                                                        visualize=visualize)

        # # normalized laplacian matrix
        degree_matrix_sqrt_inv = sp.diags(np.ravel(1 / np.sqrt(adj_matrix.sum(axis=1))))
        laplacian_matrix = sp.eye(d) - degree_matrix_sqrt_inv @ adj_matrix @ degree_matrix_sqrt_inv

    else:
        inter_cluster_prob = 0.05
        proportion_correlated = 0.2

        all_features = np.arange(d)
        selected_features = np.arange(k)

        not_selected_features = np.setdiff1d(all_features, selected_features)

        if h != 1:
            raise ValueError("Only one cluster is supported for the fixed graph structure.")
        
        clusters = [selected_features]
        w = np.zeros(d)
        for i in range(k):
            w[i] = 1.0

        # selected features form a complete graph
        adj_matrix = sp.lil_matrix((d, d))
        for i in range(k):
            for j in range(i + 1, k):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # not selected features form a complete graph
        for i in range(k, d):
            for j in range(i + 1, d):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

        # connect the two clusters with an edge between a selected feature and a not selected feature
        for i in range(k):
            for j in range(k, d):
                if np.random.rand() < inter_cluster_prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                    print(f"inter-cluster edge: ({i}, {j})")

        # we only connect the first selected feature with the last not selected feature
        # adj_matrix[0, d-1] = 1
        # adj_matrix[d-1, 0] = 1

        # form the laplacian matrix
        G = nx.from_scipy_sparse_array(adj_matrix)
        adj_matrix = nx.to_scipy_sparse_array(G, format="csr")
        degree_matrix = sp.diags(np.ravel(adj_matrix.sum(axis=1)))
        laplacian_matrix = degree_matrix - adj_matrix

        # correlated pairs
        print(f"proportion_correlated: {proportion_correlated}")
        if proportion_correlated > 0:
            print(f"proportion_correlated: {proportion_correlated}")
            num_correlated_pairs = int(proportion_correlated * k)
            # for simplicity, we assume that the correlated pairs from the selected cluster are the first num_correlated_pairs pairs
            # and the correlated pairs from the not selected cluster are the first num_correlated_pairs pairs
            for i in range(num_correlated_pairs):
                correlated_pairs.append((i, i + k))
                print(f"correlated pair: ({i}, {i + k})")


    # adj_matrix = adj_matrix.tocsr() # convert to csr format for faster matrix-vector multiplication
    # degree_matrix = sp.diags(np.ravel(adj_matrix.sum(axis=1)))  
    # laplacian_matrix = degree_matrix - adj_matrix

            
    # Optional: Visualize the graph with cluster-based colors
    # if visualize:
    #     visualize_graph(G, clusters, selected_features=selected_features)

    return adj_matrix, laplacian_matrix, clusters, k, correlated_pairs

def generate_X(n, d, correlated_pairs=None):
    X = np.random.normal(0, 1, (n, d))
    if correlated_pairs:
        for (i, j) in correlated_pairs:
            X[:, j] = X[:, i] + np.random.normal(0, 0.1, n)
        return X
    return X

def generate_Y(X, w, gamma):
    n, d = X.shape
    epsilon = np.random.normal(0, gamma, n)
    signal = X @ w
    y = signal + epsilon
    
    # print the SNR
    snr = compute_snr(signal, epsilon)
    print("SNR:", snr)
    return y

def generate_synthetic_data_with_graph(n, d, h_total, h, inter_cluster, outer_cluster, gamma, visualize=False, connected=False, random=True, correlated=False):

    adj_matrix, laplacian_matrix, clusters, k, _ = generate_graph(d, h_total, h, 
                                                                  inter_cluster, 
                                                                  outer_cluster, 
                                                                  connected=connected, 
                                                                  visualize=visualize, 
                                                                  random=random)
    selected_features = clusters[:h]
    w = generate_weight(d, selected_features, k)
    
    X = generate_X(n, d)
    y = generate_Y(X, w, gamma)

    
    # # Optional: Visualize the graph
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color="skyblue", node_size=500, font_size=10)
    # plt.title("Generated Graph Structure")
    # plt.show()
    
    return X, w, y, adj_matrix, laplacian_matrix, clusters, k

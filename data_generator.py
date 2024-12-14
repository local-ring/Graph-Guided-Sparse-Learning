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
    print(X.shape, w.shape, y.shape, adj_matrix.shape, laplacian_matrix.shape, clusters.shape, selected_features.shape)
    return X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features

def save_synthetic_data_to_file(file_path, X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features):
    np.savez(file_path, X=X, w=w, y=y, clusters=clusters, selected_features=selected_features, allow_pickle=True)
    sp.save_npz(file_path.replace(".npz", "_adj_matrix.npz"), adj_matrix) # we need to save the sparse matrix separately
    sp.save_npz(file_path.replace(".npz", "_laplacian_matrix.npz"), laplacian_matrix)

    print("synthetic data saved to the file", file_path)

def generate_graph(d, k, h, theta, connected=False, visualize=True, random=True):
    if random:
        selected_features = np.random.choice(d, k, replace=False)
        
        # Step 3: Divide the selected features into h clusters
        cluster_size = k // h  # Assume k is divisible by h for simplicity TODO: imbalanced clusters
        clusters = [selected_features[i * cluster_size : (i + 1) * cluster_size] for i in range(h)]
        
        # Step 4: Construct the regression weight vector w
        w = np.zeros(d)
        for cluster in clusters:
            """
            we need to find a different way to assign the value of the weights because we want different weights for each cluster, what i am concerned is that
            if they only have two values, it's highly likely that the weights will be the same for most of the clusters, which will confuse the model
            """
            sign = np.random.choice([-1, 1])
            feature_value = np.random.normal(1/np.sqrt(k), 1) * sign
            for feature in cluster:
                w[feature] = feature_value
        
        # Step 5: Create a sparse adjacency matrix for the graph
        adj_matrix = sp.lil_matrix((d, d))  # Start with a sparse matrix in List of Lists format
        
        for i in range(d):
            for j in range(i + 1, d):
                # Check if i and j are in the same cluster
                same_cluster = any(i in cluster and j in cluster for cluster in clusters)
                prob = theta if same_cluster else (1 - theta)
                
                if np.random.rand() < prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Ensure symmetry
        
        # Ensure the graph is connected
        G = nx.from_scipy_sparse_array(adj_matrix)  
        if not nx.is_connected(G):
            print("Number of connected components:", nx.number_connected_components(G))
            if connected:
                # Connect the isolated components 
                components = list(nx.connected_components(G))
                for i in range(len(components)):
                    isolated_node = next(iter(components[i]))  # Pick a node from the isolated component

                    # Select a random cluster and a random node from that cluster
                    random_cluster = random.choice(clusters)  # Select a random cluster
                    random_node = random.choice(random_cluster)  # Pick a random node from the selected cluster
                    
                    # Add an edge to connect the isolated component to the selected node
                    G.add_edge(isolated_node, random_node)

                # check if the graph is connected after connecting components
                try:
                    assert nx.is_connected(G)
                except AssertionError:
                    print("The graph is not connected")
                    
        # Update the adjacency matrix after connecting components
        adj_matrix = nx.to_scipy_sparse_array(G, format="csr")
        
        # Step 6: Compute the Laplacian matrix as a sparse matrix
        degree_matrix = sp.diags(np.ravel(adj_matrix.sum(axis=1)))
        laplacian_matrix = degree_matrix - adj_matrix
    else:
        # Create a graph
        G = nx.Graph()
        
        # Define nodes (0-based indexing)
        nodes = range(d)
        G.add_nodes_from(nodes)
        
        # Add edges for the first cluster (0, 1, 2, 3)
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
        
        # Add edges for the second cluster (4, 5, 6)
        G.add_edges_from([(4, 5), (5, 6), (6, 4)])
        
        # Connect the two clusters with an edge between nodes 3 and 4
        G.add_edge(3, 4)
        
        # Create adjacency matrix
        adj_matrix = nx.to_scipy_sparse_array(G, format="csr")
        
        # Compute the Laplacian matrix
        degree_matrix = sp.diags(np.ravel(adj_matrix.sum(axis=1)))
        laplacian_matrix = degree_matrix - adj_matrix
        
        # Define clusters explicitly
        clusters = [[0, 1, 2, 3], [4, 5, 6]]
        
        # Define the regression weight vector
        w = np.zeros(d)
        selected_features = [0, 1, 2, 3]  # Selected features (adjusted for 0-based indexing)
        feature_value = 1.0  # Assign the same weight to all selected features

            
    # Optional: Visualize the graph with cluster-based colors
    if visualize:
        visualize_graph(G, clusters, selected_features=selected_features)

    print(selected_features)
    print(clusters)

    return w, adj_matrix, laplacian_matrix, clusters, selected_features

def generate_X(n, d, correlated=False):
    if correlated:
        # generate a matrix with correlated columns first one and the sixth one are correlated
        X = np.random.normal(0, 1, (n, d))
        # X[:, 5] = X[:, 0] + np.random.normal(0, 0.1, n)
        X[:, 5] = X[:, 0]
        return X
    return np.random.normal(0, 1, (n, d))

def generate_Y(X, w, gamma):
    n, d = X.shape
    epsilon = np.random.normal(0, gamma, n)
    signal = X @ w
    y = signal + epsilon
    
    # print the SNR
    snr = compute_snr(signal, epsilon)
    print("SNR:", snr)
    return y

def generate_synthetic_data_with_graph(n, d, k, h, theta, gamma, visualize=False, connected=False, random=True, correlated=False):
    # Step 1: Generate the design matrix X with i.i.d. N(0, 1) entries
    X = generate_X(n, d, correlated=correlated)
    
    w, adj_matrix, laplacian_matrix, clusters, selected_features = generate_graph(d, k, h, theta, connected, visualize, random)

    y = generate_Y(X, w, gamma)

    
    # # Optional: Visualize the graph
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color="skyblue", node_size=500, font_size=10)
    # plt.title("Generated Graph Structure")
    # plt.show()
    
    return X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features

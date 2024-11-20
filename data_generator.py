import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    np.savez(file_path, X=X, w=w, y=y, clusters=clusters, selected_features=selected_features)
    sp.save_npz(file_path.replace(".npz", "_adj_matrix.npz"), adj_matrix) # we need to save the sparse matrix separately
    sp.save_npz(file_path.replace(".npz", "_laplacian_matrix.npz"), laplacian_matrix)

    print("synthetic data saved to the file", file_path)

def generate_synthetic_data_with_graph(n, d, k, h, theta, gamma, visualize=False):
    # Step 1: Generate the design matrix X with i.i.d. N(0, 1) entries
    X = np.random.normal(0, 1, (n, d))
    
    # Step 2: Randomly select k features as non-zero/contributing features
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
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            # Connect an isolated component to the main component 
            # TODO: not only the main component, but also to the other components
            G.add_edge(next(iter(components[i])), next(iter(components[0])))
    
    # Update the adjacency matrix after connecting components
    adj_matrix = nx.to_scipy_sparse_array(G, format="csr")
    
    # Step 6: Compute the Laplacian matrix as a sparse matrix
    degree_matrix = sp.diags(np.ravel(adj_matrix.sum(axis=1)))
    laplacian_matrix = degree_matrix - adj_matrix
    
    # Step 7: Generate the response vector y = Xw + epsilon
    epsilon = np.random.normal(0, gamma, n)
    y = X @ w + epsilon

    # Optional: Visualize the graph with cluster-based colors
    if visualize:
        visualize_graph(G, clusters, selected_features=selected_features)
    
    # # Optional: Visualize the graph
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color="skyblue", node_size=500, font_size=10)
    # plt.title("Generated Graph Structure")
    # plt.show()
    
    return X, w, y, adj_matrix, laplacian_matrix, clusters, selected_features

# Dependencies
import numpy as np
import pandas as pd
import networkx as nx


def get_google_matrix(data, cluster, alpha):
    """
    Input:
        - data: pandas.DataFrame with columns names = ["hashtags", "tweet_id", "cluster_id"]
        - cluster: list of nodes
        - alpha: float between 0 and 1 - Dumping factor (which is 1 - Teleport probability)
    Output:
        - numpy.matrix - Google matrix G = alpha A + (1-alpha) C
    """
    # 1. Compute A
    # Create networkx graph object
    graph = nx.from_edgelist(data)
    # Extract adjacency matrix
    A = nx.to_numpy_matrix(graph)
    # Normalize A
    A /= A.sum(axis=0)

    # 2. Compute G
    nodes = graph.nodes(data=True)

    mask = # Lista booleana di nodi in cluster, ordinata come A

    G = alpha*A
    G[:, mask] += (1-alpha)/len(cluster)

    return G



def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k



def main():
    # Load data

    # Compute Google matrix

    # Compute eigenvector

    # Save results



if __name__ == "__main__":
    main()

# Dependencies
import json
import numpy as np
import pandas as pd
import networkx as nx


def get_adjacency_matrix(data):
    """
    Input:
        - data: pandas.DataFrame with columns names = ["hashtags", "tweet_id", "cluster_id"]
    Output:
        - numpy.matrix A - Adjacency matrix
        - dict           - Node number: node name
    """
    # Create networkx graph object
    graph = nx.from_edgelist(data)
    # Extract adjacency matrix
    A = nx.to_numpy_matrix(graph)
    # Get graph n_nodes
    nodes = graph.nodes(data=True)

    return A, nodes



def get_google_matrix(A, nodes, cluster, alpha):
    """
    Input:
        - A       : numpy.matrix of dimension [n_nodes, n_nodes]
        - nodes   : dictionary that associates each node number to its name
        - cluster : list of strings (node names)
        - alpha   : float between 0 and 1 -- Dumping factor (which is 1 - teleport probability)
    Output:
        - numpy.matrix - Google matrix G = alpha A + (1-alpha) C
    """
    # Normalize A (stochastic on columns)
    A /= A.sum(axis=0)

    mask = # Lista booleana di nodi in cluster, ordinata come A

    G = alpha*A
    G[:, mask] += (1-alpha)/len(cluster)

    return G



def power_iteration(G, max_iter: int, tolerance=10**(-3)):
    """
    Input:
        - G         : squared numpy.matrix -- Google matrix
        - max_iter  : int -- maximum number of iterations
        - tolerance : float -- maximum accepted error
    Output:
        - approximate eigenvector of G (unique if G is a Google matrix)
    """
    # Choose a random vector to decrease the chance that our vector is orthogonal to the eigenvector
    b_k = np.random.rand(G.shape[1])

    for _ in range(max_iter):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(G, b_k)

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

        # If the precision increment is uniformly lower than the tolerance, break
        if np.allclose(b_k, b_k1, atol=tolerance):
            break

    return b_k



def main():

    # Load communities data
    data_path = "data/communities/"
    data_communities = pd.read_csv(data_path+"hashtags_community_selected.csv", header=0)

    # Load tweet_id - hashtag map
    data_tweet_id = pd.read_json("data/db/hashtags.json", orient=str)

    # Join with tweet_id
    data_communities["hashtag"] = data_communities["hashtag"].astype(str)
    data_tweet_id["hashtag"] = data_tweet_id["entity_text"].astype(str)
    data = data_communities.join(data_tweet_id[["tweet_id", "hashtag"]], on="hashtag", how="outer")

    # Init metrics container
    clusters = data_communities.community.unique()
    id_tweets = data_tweet_id.tweet_id.unique()
    community_similarity = pd.DataFrame(index=id_tweets, columns = clusters, dtype=float)

    # Loop through communities

    # Compute adjacency matrix

    # Compute Google matrix

    # Compute eigenvector

    # Add eigenvector to metrics container

    # Save results



if __name__ == "__main__":
    main()

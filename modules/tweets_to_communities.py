# Dependencies
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx

# Local dependencies
from dataset.tweets import Tweets
from dataset.entities import Entities

# Constants
alpha = 0.9
max_iter = 100
out_dir_path = "data/communities/"


def get_adjacency_matrix(data):
    """
    Input:
        - data: pandas.DataFrame with columns names = ['index_id', 'index_tag']
    Output:
        - numpy.matrix A - Adjacency matrix
    """
    # Create networkx graph object
    graph = nx.from_pandas_edgelist(data, source='index_id', target='index_tag')
    # Check if the graph is connected
    cc = nx.number_connected_components(graph)
    if cc > 1:
        warnings.warn('The bipartite graph is not connected!')
    # Extract adjacency matrix
    A = nx.to_numpy_matrix(graph)

    return A



def get_google_matrix(A, e2i, cluster, alpha):
    """
    Input:
        - A       : numpy.matrix of dimension [n_nodes, n_nodes]
        - e2i     : dictionary that associates each node name to its number in the graph
        - cluster : list of strings (node names)
        - alpha   : float between 0 and 1 -- Dumping factor (which is 1 - teleport probability)
    Output:
        - numpy.matrix - Google matrix G = alpha A + (1-alpha) C
    """
    # Normalize A (stochastic on columns)
    A /= A.sum(axis=0)
    # Mask of indices in the cluster
    mask = [ e2i[e] for e in cluster ]
    # Compute google matrix
    G = alpha*A
    G[mask, :] += (1-alpha)/len(cluster)

    return G



def power_iteration(G, max_iter: int, tolerance=1e-3):
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
        b_k1 = G @ np.reshape(b_k, (-1,1))

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
    communities = pd.read_csv(data_path+"hashtags_community_selected.csv", header=0)

    # Load tweet_id - hashtag map
    hashtags = Entities()
    hashtags.from_json("data/db/hashtags.json")
    hashtags.df = hashtags.df[['tweet_id','entity_text']].rename(columns={'entity_text': 'hashtag'})
    hashtags.df.hashtag = hashtags.df.hashtag.apply(lambda x: x.lower())
    # Drop rows with search hashtags
    seed_list = ["#climatechange", "#climate", "#sdgs", "#sustainability", "#environment", "#globalwarming"]
    hashtags.df = hashtags.df[~hashtags.df.hashtag.isin(seed_list)]

    # Load tweet data
    tweets = Tweets()
    tweets.from_json("data/db/tweets.json")

    # Join with tweet_id
    for year in years:
        # Select ids of the year
        curr_ids = list(tweets.df.tweet_id[tweets.df.tweet_date.dt.year == year].values)
        # Select communities of the year
        curr_communities = communities[communities.year == year]
        # Select hashtags of interest
        curr_hashtags = hashtags.df[hashtags.df.tweet_id.isin(curr_ids)]
        # Create edges
        data = curr_communities.merge(curr_hashtags, on="hashtag", how="outer")
        # Drop not-in-cluster hashtags
        data = data.loc[~data.isna().any(axis=1)]

        # Map entities in index
        nodes = list(data.hashtag.unique())
        nodes.extend(data.tweet_id.unique())
        e2i = dict(zip(nodes, range(len(nodes))))
        # Map index in entities
        i2e = dict(zip(range(len(nodes)), nodes))

        # Add indices to data
        data['index_id'] = data.tweet_id.apply(lambda x: e2i[x])
        data['index_tag'] = data.hashtag.apply(lambda x: e2i[x])
        data = data[['index_id', 'index_tag']]
        # Init metrics container for year
        clusters = curr_communities.community.unique()
        community_similarity = pd.DataFrame(columns=clusters, dtype=float)

        # Compute adjacency matrix
        A = get_adjacency_matrix(data)

        print("Network {:d}".format(year))
        # Loop through communities
        for cluster in clusters:
            # Compute Google matrix
            G = get_google_matrix(A, e2i, curr_communities.hashtag[curr_communities.community == cluster], alpha)
            # Compute eigenvector
            v = power_iteration(G, 100)
            # Add eigenvector to metrics container
            community_similarity[cluster] = np.array(v).squeeze()

        # Save results
        community_similarity.to_csv(out_dir_path+"tweet_communities{}.csv".format(year))



if __name__ == "__main__":
    main()

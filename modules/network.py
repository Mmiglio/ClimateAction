import numpy as np
import pandas as pd
import networkx as nx


class Network(object):

    def __init__(self):
        self.graph = None

    def create_network(self, df, words_network=True):
        """
        Input:
            - df (pd.DataFrame): dataframe with entities structure
            - words_network (Bool): True if df contain words, False if
                                    df contain hashtags.
        """
        # create column containing nodes: if words_network is True then nodes will be
        # of the type (word, tag) otherwise (#hashtag)
        if words_network:
            df['node'] = df.apply(lambda row: (row['entity_text'], row['entity_tag']),axis=1)
        else:
            df['node'] = df['entity_text']

        # create edges and remove self loops
        edges = pd.merge(df, df, on='tweet_id')
        edges = edges[edges.entity_index_x != edges.entity_index_y]

        # Count how many times the same word matches have been found
        edges = edges.groupby(['node_x', 'node_y']).size()
        edges = edges.reset_index(name='weight')

        # create nx graph from edges dataframe
        graph = nx.from_pandas_edgelist(
            df=edges,
            source='node_x',
            target='node_y',
            edge_attr=['weight']
        )
        self.graph = graph

    def load_network(self, path):
        """
        Load a network from disk (.gexf)
        """
        self.graph = nx.read_gexf(path)

    def save_network(self, path):
        """
        Load a network from disk (.gexf)
        """
        nx.write_gexf(self.graph, path)

    def degree_statistics(self):
        """
        Compute quantities relative to the degree distribution
        """
        degree = self.get_degree()
        degree, count = np.unique(degree.values, return_counts=True)
        pdf = count / np.sum(count)  # Compute pdf
        cdf = list(1 - np.cumsum(pdf))[:-1] + [0]  # Compute cdf
        return degree, count, pdf, cdf

    def power_law(self, k_sat):
        """
        Fit the degree distribution usign a power law
        """
        # Get the unique values of degree and their counts
        degree = self.get_degree()#np.array(list(self.get_degree().values()))
        k, count = np.unique(degree, return_counts=True)
        # Define minumum and maximum k (degree)
        k_min = np.min(k)
        k_max = np.max(k)
        # Estimate parameters
        n = degree[k_sat:].shape[0]
        gamma = 1 + n / np.sum(np.log(degree[k_sat:] / k_sat))
        c = (gamma - 1) * k_sat ** (gamma - 1)
        # Compute cutoff
        cutoff = k_sat * n ** (1 / (gamma - 1))
        return k_min, k_max, gamma, c, cutoff

    def get_connected_components(self):
        """
        Find the connected components in graph
        """
        cc = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        connected_components = []
        for component in cc:
            connected_components.append({
                'component': component,
                'size': len(component)
            })
        return connected_components

    def project_giant_component(self, component):
        """
        Use the giant component as graph
        """
        self.graph = self.graph.subgraph(component)

    def get_degree(self):
        return pd.Series({node: degree for node, degree in nx.degree(self.graph, weight='weight')})

    def get_page_rank(self):
        return pd.Series({node: score for node, score in nx.pagerank_numpy(self.graph, weight='weight').items()})

# Test
if __name__ == '__main__':
    # load a pandas dataframe
    df = pd.read_json('data/db/test_hashtags.json')

    # create network of hashtags from df
    network = Network()
    network.create_network(df, words_network=False)

    # write to file
    network.save_network('data/db/test_network.gexf')

    # load from file
    network.load_network('data/db/test_network.gexf')

    # compute only degrees
    degree = network.get_degree()

    # get degree statistics
    degree, count, pdf, cdf = network.degree_statistics()

    # power law
    k_min, k_max, gamma, c, cutoff = network.power_law(k_sat=10)

    # compute page rank score
    page_rank = network.get_page_rank()

    # find conected components (NO DIAMETER)
    cc = network.get_connected_components()
    print("Size of the components: {}".format([c['size'] for c in cc]))

    # select the largest connected components (it will substitute the entire graph)
    network.project_giant_component(cc[0])

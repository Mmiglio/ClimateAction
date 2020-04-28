# Dependencies
import numpy as np
import pandas as pd
import networkx as nx


class Network:

    # Constructor
    def __init__(self, net=None):
        # Initialize NetworkX inner instance
        self.net = net

    # Generate inner networkx instance from Entities table
    @staticmethod
    def from_entities(entities, node_getter):
        # Create a copy of input entities
        entities = entities.copy()

        # Create nodes column containing nodes
        # Set nodes as entity text
        entities.df['node'] = entities.df.apply(
            node_getter,
            axis=1
        )

        # Create edges (new, detached DataFrame)
        edges = pd.merge(entities.df, entities.df, on='tweet_id')
        # Remove self loops
        edges = edges[edges.entity_index_x != edges.entity_index_y]

        # Count how many times the same word matches have been found
        edges = edges.groupby(['node_x', 'node_y']).size()
        edges = edges.reset_index(name='weight')

        # Create inner NetworkX object from edges DataFrame
        return Network(nx.from_pandas_edgelist(
            df=edges,
            source='node_x',
            target='node_y',
            edge_attr=['weight']
        ))

    # Load inner NetworkX object from .gexf file
    def from_gexf(self, in_path):
        self.net = nx.read_gexf(in_path)

    # Store inner NetworkX object in .gexf file
    def to_gexf(self, out_path):
        nx.write_gexf(self.net, out_path)

    # Compute degree and return it as Pandas Series
    def get_degree(self):
        return pd.Series({
            node: degree for node, degree in nx.degree(
                self.net,
                weight='weight'
            )
        })

    # Compute and retrieve degree statistics (degree, count, pdf, cdf)
    def get_degree_stats(self):
        # Retrieve degree
        degree = self.get_degree()
        # Get degree counts
        degree, count = np.unique(degree.values, return_counts=True)
        # Compute PDF
        pdf = count / np.sum(count)
        # Compute CDF
        cdf = list(1 - np.cumsum(pdf))[:-1] + [0]
        # Return degree, degree count, degree PDF and degree CDF
        return degree, count, pdf, cdf

    # Compute and retrieve power law parameter
    def power_law(self, k_sat):
        # Get the unique values of degree and their counts
        degree = self.get_degree()
        # Get degree and degree counts
        k, count = np.unique(degree, return_counts=True)
        # Define minumum and maximum k (degree)
        k_min, k_max = np.min(k), np.max(k)
        # Estimate parameters
        n = degree[k_sat:].shape[0]
        gamma = 1 + n / np.sum(np.log(degree[k_sat:] / k_sat))
        c = (gamma - 1) * k_sat ** (gamma - 1)
        # Compute cutoff
        cutoff = k_sat * n ** (1 / (gamma - 1))
        # Return power law parameters
        return k_min, k_max, gamma, c, cutoff

    # Find connected components
    def get_connected_components(self):
        # Compute connected components and sort them
        cc = sorted(nx.connected_components(self.net), key=len, reverse=True)
        # Parse connected components as array of dictionaries (component: len)
        for i in range(len(cc)):
            # Overwrite component with dictionary
            cc[i] = {
                'component': cc[i],
                'size': len(cc[i])
            }
        # return connected components
        return cc

    # Project a subgraph given a component
    def project_component(self, component):
        return Network(self.net.subgraph(component))

    # Compute page rank as Pandas Series
    def get_page_rank(self):
        return pd.Series({
            node: score for node, score in nx.pagerank_scipy(
                self.net,
                weight='weight'
            ).items()
        })

    # get pandas dataframe with degree and page rank for each node
    def get_metrics_df(self):
        # compute metrics
        degree = self.get_degree()
        page_rank = self.get_page_rank()
        # concat the two series
        df = pd.concat([degree, page_rank], axis=1).reset_index()
        # rename columsn
        df.columns = ['word', 'tag', 'degree', 'page_rank']
        return df


class WordsNet(Network):

    @staticmethod
    def from_entities(entities):
        return Network.from_entities(
            entities=entities,
            node_getter=lambda row: (row['entity_text'], row['entity_tag'])
        )


class HashNet(Network):

    @staticmethod
    def from_entities(entities):
        return Network.from_entities(
            entities=entities,
            node_getter=lambda row: row['entity_text']
        )


# # Test
# if __name__ == '__main__':
#     # load a pandas dataframe
#     df = pd.read_json('data/db/test_hashtags.json')
#
#     # create network of hashtags from df
#     network = Network()
#     network.create_network(df, words_network=False)
#
#     # write to file
#     network.save_network('data/db/test_network.gexf')
#
#     # load from file
#     network.load_network('data/db/test_network.gexf')
#
#     # compute only degrees
#     degree = network.get_degree()
#
#     # get degree statistics
#     degree, count, pdf, cdf = network.degree_statistics()
#
#     # power law
#     k_min, k_max, gamma, c, cutoff = network.power_law(k_sat=10)
#
#     # compute page rank score
#     page_rank = network.get_page_rank()
#
#     # find conected components (NO DIAMETER)
#     cc = network.get_connected_components()
#     print("Size of the components: {}".format([c['size'] for c in cc]))
#
#     # select the largest connected components (it will substitute the entire graph)
#     network.project_giant_component(cc[0])

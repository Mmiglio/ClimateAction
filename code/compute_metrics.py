import numpy as np
import pandas as pd
import networkx as nx

TOPIC = 'greta'


# Extract cardinality of connected components and diameter of the giant component for both nets
def components(networks):
    # Initialize components container
    connected_components = {}
    # Compute giant component for every network
    for i, y in enumerate(networks.keys()):
        # Compute connected component
        cc = sorted(nx.connected_components(networks[y]), key=len, reverse=True)
        # Compute diameter of the giant component
        d = nx.diameter(networks[y].subgraph(cc[0]))
        # Store the tuple (giant component, cardinality, diameter)
        connected_components[y] = []
        connected_components[y].append({
            'component': cc[0],
            'size': len(cc[0]),
            'diameter': d
        })
        # Store each component
        for component in cc[1:]:
            # Add component, without diameter
            connected_components[y].append({
                'component': component,
                'size': len(component)
            })
    # Save connected components to disk
    np.save('../data/connected_components_{}.npy'.format(TOPIC), connected_components)
    # Show connected components info for each year
    for y in networks.keys():
        # Retrieve connected component
        cc = connected_components[y]
        # Show giant component info
        print('Network {:d}'.format(y))
        print('Giant component has cardinality={:d} and diameter={:d}'.format(
                cc[0]['size'], cc[0]['diameter']))
        # Store each component
        for j, component in enumerate(cc):
              if j == 0: continue
              # Show other components
              print('Connected component nr {:d} has cardinality={:d}'.format(
                    j + 1, component['size']))
        print()
    return connected_components


# Define function to retrieve degree as Pandas Series object
def get_degree(networks, connected_components):
    degree = {}
    for y in networks.keys():
        # Define giant component subgraph
        giant_component = connected_components[y][0]['component']
        subgraph = nx.induced_subgraph(networks[y], giant_component)
        # Compute degree
        degree[y] = pd.Series({node: degree for node, degree in nx.degree(
                subgraph, weight='weight')})
    # Save betweenness as numpy array
    np.save('../data/degree_{}.npy'.format(TOPIC), degree)
    return degree


# Define function to retrieve betweenness
def get_betweenness(networks, connected_components):
    # dict { year : vector }
    betweenness = {}
    for y in networks.keys():
        # Define giant component subgraph
        giant_component = connected_components[y][0]['component']
        subgraph = nx.induced_subgraph(networks[y], giant_component)
        # Compute betweenness
        betweenness[y] = nx.betweenness_centrality(subgraph, weight='weight')
    # Save betweenness as numpy array
    np.save('../data/betweenness_{}.npy'.format(TOPIC), betweenness)
    return betweenness


def main():
    # info
    print('Currently analysed network:', TOPIC, end = '\n\n')
    # Define the years of interest per topic
    if TOPIC == 'metoo': years = [2017, 2018]
    if TOPIC == 'greta': years = [2018, 2019]
    # Define networkx graph objects starting from edgelists
    networks = dict()
    # Load edges
    edges = pd.read_csv('../data/edges_{}.csv'.format(TOPIC))
    edges = {y: edges[edges.year == y][['node_x', 'node_y', 'weight']] for y in years}

    # Create newtorks
    for y in edges.keys():
        networks[y] = nx.from_pandas_edgelist(edges[y], source='node_x',
                    target='node_y', edge_attr=True, create_using=nx.Graph)
    # Retrieve connected components
    connected_components = components(networks)
    # Retrieve degree centrality on the GC
    _ = get_degree(networks, connected_components)
    # Retrive betweenness centrality on the GC
    _ = get_betweenness(networks, connected_components)
    print("{} network's components & metrics saved!".format(TOPIC))


if __name__ == "__main__":
    main()

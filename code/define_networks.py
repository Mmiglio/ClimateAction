'''
This python script generates the edges dataframe from selected words and tweets
datasets (through the TOPIC constant variable). In addiction, the word-to-index
and index-to-word dictionaries are stored for the current topic.
'''

import numpy as np
import pandas as pd

# Constants
TOPIC = 'metoo' # change to create a different network


def load_data(years):

    words = pd.read_csv('../data/words_{}.csv'.format(TOPIC), dtype={
        'id': np.unicode_,
        'index': np.int,
        'text': np.unicode_,
        'pos': np.unicode_,
        'conf': np.float
    })

    if TOPIC == 'greta':
        tweets = {  years[0] : pd.read_json('../data/tweets_{}Greta.json'.format('pre')),
                    years[1] : pd.read_json('../data/tweets_{}Greta.json'.format('post')) }
        # return pre and post datasets
        return  words, tweets

    if TOPIC == 'metoo':
        # Load tweets dataset
        tweets_all = pd.read_csv('../data/tweets_metoo.csv', dtype={
            'id': np.unicode_,
            'created_at': np.unicode_
        })
        # Parse created_at attribute to Datetime format
        tweets_all.created_at = pd.to_datetime(tweets_all.created_at, format='%a %b %d %H:%M:%S %z %Y')
        tweets =  { years[0] : tweets_all[tweets_all.created_at.dt.year == years[0]],
                    years[1] : tweets_all[tweets_all.created_at.dt.year == years[1]] }
        # return pre and post datasets
        return  words, tweets

# Define function for getting word map to integer
def map_words(words):
    # Get unique lemmas
    unique_words = words.groupby(by=['text', 'pos']).size().reset_index(name='counts')
    unique_words.sort_values(by='counts', ascending=False, inplace=True)
    # Map each unique lemma to a number and vice versa
    w2i, i2w = dict(), dict()
    for index, word in unique_words.iterrows():
        w2i.setdefault((word.text, word.pos), index)
        i2w.setdefault(index, (word.text, word.pos))
    # Return dictionaries
    return w2i, i2w

# Define function for creating edges dataset
def get_edges(words):
    """
    Input:
    - words dataset containing words (nodes)
    Output
    - edges: word edges DataFrame (node_x, node_y, counts)
    """

    # Make join to obtain words in the same tweet
    edges = pd.merge(words, words, on='id')
    edges = edges[edges.index_x != edges.index_y]  # Remove self join

    # Count how many times the same word matches have been found
    edges = edges.groupby(['node_x', 'node_y']).size()
    edges = edges.reset_index(name='weight')

    # Return edges dataset
    return edges


def main():
    # Define the years of interest per topic
    if TOPIC == 'metoo': years = [2017, 2018]
    if TOPIC == 'greta': years = [2018, 2019]
    # Load words and tweets datasets (already splitted per year)
    words, tweets = load_data(years)
    # Retrieve dictionaries mapping lemma tuples to numeric value
    w2i, i2w = map_words(words)
    # Map lemmas to node numbers
    words['node'] = words.apply(lambda w: w2i[(w.text, w.pos)], axis=1)
    # Define edges for pre and post (as Pandas DataFrames)
    edges = { years[0] : get_edges(words[words.id.isin(tweets[years[0]].id.values)]) ,
            years[1] : get_edges(words[words.id.isin(tweets[years[1]].id.values)]) }

    # Save vocabularies to disk
    np.save('../data/w2i_{}.npy'.format(TOPIC), w2i)  # Save tuple to index vocabulary
    np.save('../data/i2w_{}.npy'.format(TOPIC), i2w)  # Save index to tuple vocabulary
    # Save edges to disk
    edges_ = [*years]
    # Loop through each edges table
    for i, y in enumerate(years):
        # Add year column
        edges_[i] = edges[y].copy()
        edges_[i]['year'] = y
    # Concatenate DataFrames
    edges_ = pd.concat(edges_, axis=0)
    print(edges_)
    # Save dataframe to disk
    edges_.to_csv('../data/edges_{}.csv'.format(TOPIC), index=False)
    print("{} network's edges saved!".format(TOPIC))

if __name__ == "__main__":
    main()

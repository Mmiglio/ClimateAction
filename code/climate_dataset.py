'''This python script generates the climate_tweets datasets (pre/post) selecting
at random 5k tweets containing #climatechange from each already created dataset
on Greta.
MUST BE RUNNED ONLY ONCE.'''

import numpy as np
import pandas as pd
import re

# Constants
years = [2018, 2019]


def load():
    tweets = {}
    tweets[years[0]] = pd.read_json('../data/tweets_preGreta.json',
                                    dtype = { 'id': np.unicode_})
    tweets[years[1]] = pd.read_json('../data/tweets_postGreta.json',
                                    dtype = { 'id': np.unicode_})
    return tweets


def define_mask(row, mask, words):
    #find hashtags
    hash_tweet = set(re.findall(r"#(\w+)", row.text.lower()))
    #check if is one of interest
    mask.append((hash_tweet & words) != set())


def generate_data(tweets):
    climate_tweets = {}
    for y in years:
        # initialize: the mask is changed in place
        climate_mask = []
        # define masks for tweets containing #climatechange
        tweets[y].apply(define_mask, mask = climate_mask, words={'climatechange'},
                    axis = 1)
        # random selection of 5k indices w/o repetition
        sample = np.random.choice(tweets[y][climate_mask].index, size = 5000,
                                    replace = False)
        # filter and beautify data
        climate_tweets[y] = tweets[y][[j in sample for j in tweets[y].index]].sort_values(
                            by='created_at').reset_index(drop=True)

    return climate_tweets


def main():
    # dict {year : dataframe}
    tweets = load()
    # dict {year : dataframe}
    climate_tweets = generate_data(tweets)
    # save results
    dataset = pd.concat([climate_tweets[y] for y in years], ignore_index=True)
    dataset.to_json('../data/dataset/tweets_climatechange.json')


if __name__ == "__main__":
    main()

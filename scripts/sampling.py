# Dependencies
import sys, os; sys.path.insert(1, os.path.join(sys.path[0], '..'))
from modules.dataset.tweets import Tweets
from datetime import datetime, date, timedelta
from random import randrange
import argparse
import pandas as pd
import re


# Constants
RANDOM_SEED = 42  # Random seed, grants replicability
BATCH_SIZE = 5000  # Number of entry to be returned per year


# Main
if __name__ == '__main__':

    # Load tweets pre greta
    tweets_pre_greta = Tweets()
    tweets_pre_greta.from_json_list(in_path='data/preGreta.jsonl')
    print(tweets_pre_greta.df.head())
    print('Shape', tweets_pre_greta.df.shape)
    print()

    # Load tweets post great
    tweets_post_greta = Tweets()
    tweets_post_greta.from_json_list(in_path='data/postGreta.jsonl')
    print(tweets_post_greta.df.head())
    print('Shape', tweets_post_greta.df.shape)
    print()

    # Merge the two datasets into one
    tweets = Tweets()
    tweets.df = pd.concat([tweets_pre_greta.df, tweets_post_greta.df], axis=0)
    # print(tweets.df.head())
    print(tweets.df.head())
    print('Shape', tweets.df.shape)
    print()

    # Define container for each year tweet
    tweets_year = {2018: None, 2019: None}
    # Make random sampling of tweets in either 2017 and 2018
    for curr_year in tweets_year.keys():
        # Create a new empty tweets dataset
        curr_tweets = Tweets()
        # Subset current year tweets
        curr_tweets.df = tweets.df[tweets.df.tweet_date.dt.year == curr_year]
        print(curr_tweets.df.head())
        print('Shape', curr_tweets.df.shape)
        print()
        # Make samples on dataset indexes
        curr_tweets.df = curr_tweets.df.sample(
            n=BATCH_SIZE,
            random_state=RANDOM_SEED,
            replace=False,
            axis=0
        )
        # Store sampled tweets
        tweets_year[curr_year] = curr_tweets

    # Create a new sampled tweets dataset
    tweets_sampled = Tweets()
    tweets_sampled.df = pd.concat([tw.df for tw in tweets_year.values()], axis=0)
    print(tweets_sampled.df.head())
    print('Shape', tweets_sampled.df.shape)
    print()

# Dependencies
from modules.dataset.dataset import Dataset
from modules.dataset.entities import Entities
from datetime import datetime
import pandas as pd
import numpy as np
import json
import re


class Tweets(Dataset):

    # Construtcor
    def __init__(self):
        # Call parent constructor
        super().__init__(columns={
            'tweet_id': np.unicode_,
            'tweet_date': np.unicode_,
            'tweet_text': np.unicode_
        })

    # TODO Search tweet through APIs and fill inner dataset
    def search(self):
        return NotImplemented()

    # Retrieve hashtags and words dataset from tweets
    def get_entities(self, subs={}):
        # Create new Pandas dataframe containing entities (either words and hashtags)
        entities = Entities()
        entities.from_tweets(self)  # Tag entities for the first time
        entities.df.sort_values(by=['tweet_id', 'entity_index'], ascending=True, inplace=True)
        # Create separate hashtags dataset from entities dataset
        hashtags = Entities()
        hashtags.df = entities.df.loc[entities.df.entity_text.apply(lambda x: bool(re.match(r'^#', x)))].copy()
        # Filter out stand alone hashtags (tagged #)
        entities.df = entities.df.loc[entities.df.entity_tag != '#']
        # Loop through each tweet
        for i, tweet in self.df.iterrows():
            # Get entities for current tweet
            tweet_entities = entities.df.loc[entities.df.tweet_id == tweet.tweet_id]
            # Reinitialize tweet text
            tweet_text = ''
            # Rebuild the sentence using words tagged
            for j, entity in tweet_entities.iterrows():
                # Substitute complex hashtags with splitted ones
                entity_text = subs.get(entity.entity_text.lower(), entity.entity_text)
                # Create new tweet text
                tweet_text = ' '.join([tweet_text, entity_text])
            # Overwrite current tweet text
            self.df.at[i, 'tweet_text'] = tweet_text
        # Launch tagger again
        words = Entities()
        words.from_tweets(self)
        words.df.sort_values(by=['tweet_id', 'entity_index'], inplace=True, ascending=True)
        # Return retrieved hashtags and words datasets
        return hashtags, words


# Test
if __name__ == '__main__':

    # Set scipt start time
    start_time, end_time = None, None
    start_time = datetime.now()

    # Instantiate a Tweets object
    tweets = Tweets()
    tweets.from_json('data/db/test_tweets.json')
    # Get only a small batch of the whole dataset (e.g. first 1000 rows)
    tweets.df = tweets.df[:1000]

    print(tweets.df.head(), '\n')

    # Initialize substitution dictionary
    subs = {}
    # Load hahstag substitutions
    with open('data/hashtag_subs.json', 'r') as file:
        subs = {**subs, **json.load(file)}
    # Load contact forms substitutions
    with open('data/contract_forms.json', 'r') as file:
        subs = {**subs, **json.load(file)}

    # Get hashtags and words datasets
    hashtags, words = tweets.get_entities(subs=subs)

    print(hashtags.df.head(), '\n')
    print(words.df.head(), '\n')

    # Store hashtags table
    hashtags.to_json('data/db/test_hashtags.json')
    # Store words table
    words.to_json('data/db/test_words.json')

    # Get end time
    end_time = datetime.now()
    # Check duration
    print('It took ', end_time - start_time, 'to execute')

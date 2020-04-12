# Dependencies
from modules.dataset.dataset import Dataset
from modules.dataset.entities import Entities
from TwitterAPI import TwitterAPI
from datetime import datetime
import numpy as np
import json_lines
import json
import re


# Constants
API_PRODUCT_30DAY = '30day'
API_PRODUCT_FULL = 'fullarchive'


class Tweets(Dataset):

    # Attributes
    api = None  # Twitter's APIs instance

    # Construtcor
    def __init__(self):
        # Call parent constructor
        super().__init__(columns={
            'tweet_id': np.unicode_,
            'tweet_date': np.datetime64,
            'tweet_text': np.unicode_
        })

    # Authentication: allows to query Twitter's web APIs
    def auth(self, consumer_key, consumer_secret, token_key, token_secret):
        # Return asuthenticated twitter APIs object
        self.api = TwitterAPI(
            consumer_key=consumer_key, consumer_secret=consumer_secret,
            access_token_key=token_key, access_token_secret=token_secret
        )

    # Authentication: load credentials from .json file
    def auth_from_json(self, in_path):
        # Initialize credentials
        credentials = {}
        # Load credentials file
        with open(in_path, 'r') as in_file:
            credentials = json.load(in_file)
        # Authenticate using loaded credentials
        self.auth(**credentials)

    # Search tweet through APIs and fill inner dataset
    def search_tweets(self, label, query=None, from_date=None, to_date=None, batch_size=100, params={}, product=API_PRODUCT_30DAY):
        # Initialize retrieved tweets list
        tweets = list()
        # Parse from and to dates
        from_date = from_date.strftime('%Y%m%d%H%M') if from_date is not None else None
        to_date = to_date.strftime('%Y%m%d%H%M') if from_date is not None else None
        # Execute request to Twitter's web API
        res = self.api.request('tweets/search/{0:}/:{1:}'.format(product, label), {
            # Free parameters, will be overwritten by specific ones
            **params,
            # Specific parameters
            **{
                'query': query,
                'fromDate': from_date,
                'toDate': to_date,
                'maxResults': batch_size
            }
        })
        # Parse tweets and fill retrieved tweets list
        for retrieved_tweet in res:
            # Parse tweet and append it to retrieved tweets list
            tweets.append(parse_tweet(retrieved_tweet))
        # Add parsed tweets to inner DataFrame
        self.df = self.df.append(tweets)

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

    # Retrieve hashtag counts
    def get_hashtag_counts(self):
        # Initialize hashtags dictionary (hashtag: counts)
        hashtag_counts = {}
        # Loop through each tweet text inside inner dataframe
        for tweet_text in self.df.tweet_text:
            # Find hashtags in current tweet tweet text
            tweet_hashtags = re.findall(r'#(\w+)', tweet_text.lower())
            # Loop through each hashtag
            for hashtag in tweet_hashtags:
                # Initialize keyword if not already in dictionary
                hashtag_counts.setdefault(hashtag, 0)
                # Update hashtag counts
                hashtag_counts[hashtag] = hashtag_counts[hashtag] + 1
        # Qui tornerei una serie con indice le chiavi, ma vabbè
        return hashtag_counts

    # Load inner dataset from disk (.json file)
    def from_json(self, in_path):
        # Load entries into inner DataFrame
        super().from_json(in_path, date_columns=['tweet_date'])

    # Load inner dataset from unparsed json list (.jsonl file)
    def from_json_list(self, in_path):
        # Initialize tweets container
        tweets = list()
        # Load input file
        with open(in_path, 'rb') as in_file:
            # Loop through each line in input .jsonl formatted file
            for retrieved_tweet in json_lines.reader(in_file, broken=True):
                # Format retrieved tweet according to inner DataFrame
                parsed_tweet = parse_tweet(retrieved_tweet)
                # Append parsed tweet to tweets list
                tweets.append(parsed_tweet)
        # Append list of retrieved tweets to inner Dataframe
        self.df = self.df.append(tweets)


# Parse retrieved tweets fo fill into internal DataFrame
def parse_tweet(retrieved_tweet, datetime_format='%a %b %d %H:%M:%S %z %Y'):
    # Initialize parsed tweet object
    parsed_tweet = dict()
    # Get tweet id
    parsed_tweet['tweet_id'] = str(retrieved_tweet.get('id_str'))
    # Get tweet date
    parsed_tweet['tweet_date'] = datetime.strptime(
        retrieved_tweet.get('created_at'),
        datetime_format
    )
    # Initialize parsed tweet text
    tweet_text = ''
    # Case tweet is a retweet
    if 'retweeted_status' in set(retrieved_tweet.keys()):
        # Get inner tweet
        retrieved_tweet = retrieved_tweet['retweeted_status']
    # Check if current tweet is an extended tweet
    if 'extended_tweet' in set(retrieved_tweet.keys()):
        tweet_text = retrieved_tweet['extended_tweet']['full_text']
    # Case current tweet is not an extended one
    else:
        tweet_text = retrieved_tweet['text']
    # Store tweet text
    parsed_tweet['tweet_text'] = tweet_text
    # Return tweet
    return parsed_tweet


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

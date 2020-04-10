# Dependencies
from resources.CMUTweetTagger import runtagger_parse
from nltk.stem import WordNetLemmatizer
from nltk.corpus.stopwords import words as stopwords
from modules.dataset.dataset import Dataset
import unidecode as ud
import numpy as np
import json
import re

# Constants
# Path to tagger executable
TAG_RUN = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar resources/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'
# Load set of stopwords
SET_STOPWORDS = set(stopwords('english'))
# Load set of pronouns
SET_PRONOUNS = set(['I', 'you', 'it', 'she', 'he', 'we', 'they', 'me', 'her',
                    'hers', 'him', 'us', 'them', 'my', 'your', 'yours', 'his',
                    'its', 'our', 'ours', 'their', 'myself', 'yourself',
                    'himself', 'herself', 'itself', 'ourselves', 'yourselves'])


class Entities(Dataset):

    # Constructor
    def __init__(self):
        # Call parent constructor
        super().__init__(columns={
            'tweet_id': np.unicode_,
            'entity_index': np.int,
            'entity_text': np.unicode_,
            'entity_tag': np.unicode_,
            'entity_conf': np.float
        })

    # Define function for filling table by running ARK twitter parser
    def from_tweets(self, tweets):
        # Tag tweet text
        entities = runtagger_parse(tweets.df.tweet_text.values,  run_tagger_cmd=TAG_RUN)
        # Define new dataset content
        df = []
        # Loop through each tagged tweet
        for i, tweet in tweets.df.iterrows():
            # Loop through each entity for current tweet
            for j, entity in enumerate(entities[i]):
                # print('\n'.join(['DEBUG', i, tweet, j, entity]))
                # Get attributes for j-th tagged entity of i-th tweet
                text, tag, conf = entity
                # Append new entry to dataset
                df.append({
                    'tweet_id': tweet.tweet_id,
                    'entity_index': j,
                    'entity_text': text,
                    'entity_tag': tag,
                    'entity_conf': conf
                })
        # Set new dataset content
        self.df = self.df.append(df, ignore_index=True)

    # Define function for cleaning entities text
    def clean_entities(self):
        # Apply clean entity function to each row
        self.df = self.df.apply(clean_entity, axis=1)
        # Define entries which are either stopwords or contain symbols
        are_stopwords = self.df.entity_text.apply(is_stopword)  # Stopwords
        have_symbols = self.df.entity_text.apply(has_symbols)  # Symbols
        # Subset dataset excluding invalid entries
        self.df = self.df.loc[are_stopwords & ~have_symbols]


# Lemmatizing a word, given text and pos tag
def lemmatize(text, tag):
    # Instantiate a word lemmatizer instance
    wnl = WordNetLemmatizer()
    # Pronouns don't need lemmatization
    if tag not in {'O', 'S'}:
        # Return plain text
        return text
    # Return lemmatized word
    return wnl.lemmatize(text, tag)

# Remove accets from text
def remove_accents(text):
    return ud.unidecode(text)

# States wether a text contains non-standard symbols
def has_symbols(text):
    return bool(re.match(r'[^\w-]', text))

# States wether it is a stopword
def is_stopword(text):
    return bool(text.lower() in set(SET_STOPWORDS))

# States wether it is a pronoun
def is_pronoun(text):
    return bool(text.lower() in set(SET_PRONOUNS))

# Clean an entry, applying various
def clean_entity(text, tag, conf):
    # Remove accents
    text = remove_accents(text)
    # Change tag if is pronoun
    if is_pronoun(text):
        tag = 'O'
    # Case the entry is a stopword
    # Return cleaned entity attributes
    return text, tag, conf

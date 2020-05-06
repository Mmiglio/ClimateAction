# Dependencies
from resources.CMUTweetTagger import runtagger_parse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from modules.dataset.dataset import Dataset
import itertools as iter
import unidecode as ud
import numpy as np
import re

# Constants
# Path to tagger executable
TAG_RUN = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar resources/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'
# Load set of stopwords
SET_STOPWORDS = set( stopwords.words('english') +
                    ['would', 'could', 'cannot', "can't", 'must', 'might']
                    )
# Load set of pronouns
SET_PRONOUNS = set(['i', 'you', 'it', 'she', 'he', 'we', 'they', 'me', 'her',
                    'hers', 'him', 'us', 'them', 'my', 'your', 'yours', 'his',
                    'mine', 'its', 'our', 'ours', 'their', 'myself', 'yourself',
                    'himself', 'herself', 'itself', 'ourselves', 'yourselves',
                    'themselves', 'theirs'])


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
        # Get list of tweet id
        tweet_ids = tweets.df.tweet_id.tolist()
        # Get tweet text
        tweet_text = tweets.df.tweet_text.tolist()
        tweet_text = [re.sub(r'[\n\r]', ' ', txt) for txt in tweet_text]
        tweet_text = [re.sub(r'[ ]+', ' ', txt) for txt in tweet_text]
        # Tag tweets text
        tweet_tags = runtagger_parse(tweet_text, run_tagger_cmd=TAG_RUN)
        # Define new dataset content
        entities = []
        # Loop through each tagged tweet
        for i, tweet_id in enumerate(tweet_ids):
            # Loop through each entity for current tweet
            for j, tweet_tag in enumerate(tweet_tags[i]):
                # Get attributes for j-th tagged entity of i-th tweet
                text, tag, conf = tweet_tag
                # Append new entry to dataset
                entities.append({
                    'tweet_id': tweet_id,
                    'entity_index': j,
                    'entity_text': text,
                    'entity_tag': tag,
                    'entity_conf': conf
                })
        # Set new dataset content
        self.df = self.df.append(entities, ignore_index=True)

    # Define function for cleaning entities text
    def clean_entities(self):
        # Apply clean entity function to each row
        self.df = self.df.apply(clean_entity, axis=1)
        # Define entries which are either stopwords or contain symbols
        are_stopwords = self.df.entity_text.apply(is_stopword)  # Stopwords
        have_symbols = self.df.entity_text.apply(has_symbols)  # Symbols
        # Subset dataset excluding invalid entries
        self.df = self.df.loc[~are_stopwords & ~have_symbols]


# Lemmatizing a word, given text and pos tag
def lemmatize(text, tag):
    # Instantiate a word lemmatizer instance
    wnl = WordNetLemmatizer()
    # Pronouns don't need lemmatization
    if tag not in {'N', 'V', 'R', 'A'}:
        # Return plain text
        return text
    # Return lemmatized word
    return wnl.lemmatize(text, tag.lower())

# Remove accets from text
def remove_accents(text):
    text = ud.unidecode(text)
    text = re.sub("`","'", text)
    return text

# States wether a text contains non-standard symbols apart from -
def has_symbols(text):
    return len(re.findall(r'[^\w-]', text)) != 0

# States wether it is a stopword
def is_stopword(text):
    return bool(text.lower() in set(SET_STOPWORDS) - set(SET_PRONOUNS))

# States wether it is a pronoun
def is_pronoun(text):
    return bool(text.lower() in set(SET_PRONOUNS))

# Clean an entry, applying various
def clean_entity(row):
    # Change tag if is pronoun
    if is_pronoun(row.entity_text):
        row.entity_tag, row.entity_conf = 'O', 1.0
    # Remove - symbol at the beginning and at the end of a word
    row.entity_text = re.sub(r'^-', '', row.entity_text)
    row.entity_text = re.sub(r'-$', '', row.entity_text)
    # Convert the entry in lowercase
    row.entity_text = row.entity_text.lower()
    # Lemmatize
    row.entity_text = lemmatize(row.entity_text, row.entity_tag)
    return row

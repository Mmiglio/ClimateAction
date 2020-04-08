# Standard modules
import numpy as np
import pandas as pd
import re
import nltk
import enchant
import wordninja
import unidecode as ud
import matplotlib.pyplot as plt

# Twitter tagger APIs
from resources.CMUTweetTagger import runtagger_parse
# Contractions dictionary
from modules.contractions import contractions_dict
# Hashtag spliting dictionary
from modules.contractions import contractions_dict

# Constants
years = [2018,2019]
# Path to POS tagger java application
ARK_TWEET_NLP_PATH = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar resources/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'
# Define the American English set of allowed words
EN_DICT = enchant.Dict("en_US")
# Pronouns list
PRONOUNS = ['I', 'you', 'it', 'she', 'he', 'we', 'they',
            'me', 'her', 'hers', 'him', 'us', 'them',
            'my', 'your', 'yours', 'his', 'its', 'our', 'ours', 'their',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves']
# Lemmatizer
nltk.download('wordnet')
WNL = nltk.stem.WordNetLemmatizer()
# Stopwords
nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('english')


def load_data(path):
    # Data loader
    tweets = pd.read_json(path, dtype = { 'id': np.unicode_})
    return tweets

    '''if topic == 'metoo':
        # Load tweets dataset
        tweets = pd.read_csv('./data/tweets_metoo.csv', dtype={
            'id_str': np.unicode_,
            'created_at': np.unicode_
        })
        # Parse created_at attribute to Datetime format
        tweets.created_at = pd.to_datetime(tweets.created_at, format='%a %b %d %H:%M:%S %z %Y')
        # rename id_str
        tweets.rename(columns={'id_str':'id'}, inplace = True)
        return tweets'''


def split_hashtag(hashtag):
    # Splitting hashtag
    hash_words = wordninja.split(hashtag)
    # Check if all the words make sense
    for word in hash_words:
        if EN_DICT.check(word) == False:
            # Delete hashtag with no meaningful words
            return []
    return hash_words


def clean_tweet(row, tagged_tweets):

    # Find hashtag not tagged with #
    # Sub with splitting_dict value (the dict contains ALL the tags as keys, no if needed)
    # return the row

    '''# Cleaning part
    tweet = row.text.split(" ")
    # Iterate over the tweet words
    for i, word in enumerate(tweet):
        if "#" in word:
            # Initialize the hashtag
            tweet[i] = " "
            # Extract eventually stacked hashtags
            for hashtag in word.split("#")[1:]:
                h_words = split_hashtag(hashtag)
                tweet[i] = tweet[i].join(h_words)
    # Save cleaned tweet
    cleaned_string = " ".join(tweet)
    # check if a stringa xe voda
    # if a xe vòda aeora mettaghe un bel punto
    if cleaned_string.strip() == "":
        row.text = "."
    else:
        row.text = cleaned_string
    return row'''


def create_words_df(tweets):

    # Run tagger
    tagged_tweets = runtagger_parse(tweets['text'].values, run_tagger_cmd=ARK_TWEET_NLP_PATH)
    # Define new list containing tagged words
    words = list()
    # Loop through each tagged tweet
    for i, tags in enumerate(tagged_tweets):
        tweet_id = tweets.loc[i, 'id']
        for j, tag in enumerate(tags):
            text, pos, conf = tag
            # Keep only nouns (N), verbs (V), adverbs (R), adjectives (A),
            # pronouns (O), possessives (S)
            if pos in ['N', 'V', 'R', 'A', 'O', 'S']:
                words.append({
                    'id': tweet_id, # Id of tweet containing word
                    'index': j, # Word index in sentence
                    'text': text.lower(), # Actual word text
                    'pos': pos, # Part Of Speech tag
                    'conf': conf # Confidence for POS tag
                })
    print('Words dataframe\n\n', pd.DataFrame(words).head(), end = '\n\n')
    return pd.DataFrame(words)


def lemmatize(x):
    # pronouns don't need lemmatization
    if x.pos in ['O', 'S']:
        return x.text
    else:
        return WNL.lemmatize(x.text, x.pos.lower())


def decontract(x):
    # Manage contractions
    if x.text.lower() in contractions_dict.keys():
        return contractions_dict[x.text.lower()]
    else: return x.text


def pronouns_finder(x):
  # extract pronouns, change tag and conf = 1
    w = x.text.split()
    for y in w:
        if y in PRONOUNS:
            return [y, 'O', 1]
    # ignore composite words
    return [x.text, x.pos, x.conf]


def stop_finder(x):
    w = x.split()
    for y in w:
        # drop not interesting words
        if y in set(STOPWORDS) - set(PRONOUNS):
            return False
    return True


def clean_words(words):
    # Cleaning pipeline for words

    # Lemmatize
    words.text = words.apply(lemmatize, axis=1)
    # Convert punctuation
    words.text = words.apply(lambda x: ud.unidecode(x.text), axis=1)
    # Contractions
    words.text = words.apply(decontract, axis=1)
    # Identify remaining entries with symbols
    symb_mask = words.text.apply(lambda x: len(re.findall(r'[^\w-]', x))) != 0
    # Extract pronouns
    words.loc[ symb_mask, ['text', 'pos', 'conf'] ] = words.loc[symb_mask,
            ['text', 'pos', 'conf']].apply(pronouns_finder, axis = 1).values.tolist()
    # Remove useless entries (at least one stopword contained)
    words = words[words.text.apply(stop_finder)]
    # Drop entries with symbols
    words = words.loc[words.text.apply(lambda x: len(re.findall(r'[^\w-]', x))) == 0]
    return words

def words_extraction(tweets, conf, print = False):
    # Apply here 1st tagger
    tagged_tweets = runtagger_parse(tweets['text'].values, run_tagger_cmd=ARK_TWEET_NLP_PATH)
    # Split hashtag (in clean_tweets)
    tweets_clean = tweets.apply(clean_tweets, tagged_tweets, axis=1)

    # Create the words dataset as before - pipeline below

    # Exctract dataset of words
    words = create_words_df(tweets_clean)
    # Plot distribution of the confidence
    fig, ax = plt.subplots(figsize=(15, 5))
    _ = ax.set_title('Confidence distribution for POS tags',fontsize=15)
    _ = ax.hist(words.conf, bins=100)
    _ = ax.set_xlim(left=0.9, right=1.0)
    _ = ax.axvline(x=conf, c='r', ls='--')
    if print:
        _ = plt.savefig('images/preprocessing/tag_conf.png')
    _ = plt.show()
    # info
    print('There are {:d} ({:.02f}%) words under {:.2f} confidence interval'.format(
        sum(words.conf < conf), sum(words.conf < conf) / words.shape[0] * 100, conf
    ))
    # info
    print('There will be {:d} ({:.02f}%) words remaining inside {:.2f} confidence interval'.format(
        sum(words.conf >= conf), sum(words.conf >= conf) / words.shape[0] * 100, conf
    ))
    # Remove tags whose POS tag confidence is below 0.96
    words = words[words.conf >= conf]
    # Apply cleaning pipeline
    words = clean_words(words)
    # Plot words count
    word_counts = words.text.value_counts()
    word_counts = word_counts[:50]
    fig, ax = plt.subplots(figsize=(15, 5))
    _ = ax.set_title('Words distribution',fontsize=15)
    _ = ax.bar(x=word_counts.keys().tolist(), height=word_counts.tolist())
    _ = ax.tick_params(axis='x', rotation=90)
    if print:
        _ = plt.savefig('images/preprocessing/words_counts.png')
    _ = plt.plot()
    # info
    print('\n\nFinal number of words (after cleaning): ', words.shape[0])

    # Output to file
    words.to_csv('./data/words_climatechange.csv', index=False)

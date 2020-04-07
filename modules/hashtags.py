import re
import datetime
import numpy as np
import pandas as pd
import wordninja
import enchant
import matplotlib.pyplot as plt


# Constants
years = [2018, 2019]
# 1st line: neutral hashtags
# 2nd line: specific hastags
words_list=["climatechange", "climatecrisis", "parisagreement",
            "gretathunberg", "climatestrike", "fridays4future"]
# Define the American English set of allowed words
EN_DICT = enchant.Dict("en_US")

def count_words_in_row(row, words_dict):
    """
    NB: The counter is updated (+1) if the hashtag is met at least one time.
    """
    # Find the set of hashtags present in the tweet
    text = row.text.lower().replace("#", " #")
    hash_tweet = set(re.findall(r"#(\w+)", text))
    hash_dict = set(words_dict.keys())
    # Update the number of hashtag in the dictionary
    for word in (hash_tweet & hash_dict):
        words_dict[word] += 1


def count_words_in_list(df, words_list):
    # Initialize the dictionary to store counts
    words_count = dict(zip(words_list, [0]*len(words_list)))
    # Scan each row updating the counters
    df.apply(count_words_in_row, words_dict=words_count, axis=1)
    return words_count


def find_hash_distribution(row, d):
    # Find all the hashtag in the tweet
    hash_list = re.findall(r"#(\w+)", row.text.lower())

    # Increase the counter or define a new key with value 1
    for hashtag in hash_list:
        if hashtag in d:
            d[hashtag] += 1
        else:
            d[hashtag] = 1


def plot_hash_distribution(df, words, n_hash_max, n_hash_min=0):
    # Initialize the dictionary to store counts
    hash_dict = {}
    # Scan each row of the df
    df.apply(find_hash_distribution, d=hash_dict, axis=1)

    # Convert the dictionary in a df, sort it by frequency
    hash_df = pd.DataFrame.from_dict(hash_dict.items())
    hash_df.columns = ["hashtag", "occurrences"]
    hash_df.sort_values(by="occurrences", ascending=False, inplace=True)

    # Plot the barplot with the specified interval of the most frequent hashtags
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.set_title('Conditional counts on {} '.format(words), fontsize=15)
    hash_df[n_hash_min : n_hash_max].plot(x="hashtag", y="occurrences", kind="bar", ax = ax)
    plt.show()
    return hash_df


def define_mask(row, mask, words):
    hash_tweet = set(re.findall(r"#(\w+)", row.text.lower()))
    mask.append((hash_tweet & words) != set())


def filter_dataset(tweets, year, words, n_max=25, n_min=1):
    """
    Return a dataframe that contains tweets in which is present at least
    one hashtags in list "words" written in the specified year.
    """
    mask = []
    if words:
        tweets.apply(define_mask, mask=mask, words=set(words), axis=1)
        df = tweets[mask].copy()
    else:
        df = tweets
    count_words_in_list(df, words)
    top_hash = plot_hash_distribution(df[df.created_at.dt.year == year], words, n_hash_max=n_max, n_hash_min=n_min)
    return top_hash


def show_counts(tweets, words):
    for y in years:
        print('hashtag frequencies in {}: \n\n'.format(y),
                                                count_words_in_list( tweets[y], words_list),
                                                end = '\n\n')


def words_in_dict(words, dict_=EN_DICT):
    """ Check if all the words belong to dict
    """
    for word in words:
        if dict_.check(word) == False:
            return False
    return True


def splitter(hash_pre, hash_post, save = False):
    hashtags = list(set(hash_pre.hashtag) | set(hash_post.hashtag))
    splittable_words = {}

    for i, hashtag in enumerate(hashtags):
        # define the list of words obtained by splitting the hashtag
        hashtag_splitted = wordninja.split(hashtag)
        # check if the split consists in more than one word
        if (len(hashtag_splitted) > 1) and (words_in_dict(hashtag_splitted)):
            # save the word and the suggested splitting
            splittable_words[i] = [ hashtag,
                                    str(hashtag_splitted),
                                    *hash_pre[hash_pre.hashtag == hashtag].occurrences.values,
                                    *hash_post[hash_post.hashtag == hashtag].occurrences.values]
        else:
            # save the word with a None
            splittable_words[i] = [ hashtag,
                                    pd.np.nan,
                                    *hash_pre[hash_pre.hashtag == hashtag].occurrences.values,
                                    *hash_post[hash_post.hashtag == hashtag].occurrences.values]

    splittable_words = pd.DataFrame.from_dict(splittable_words,
                        orient='index',
                        columns=["word", "proposed_splitting", "count_pre", "count_post"])
    if save:
        print(splittable_words.head())
        splittable_words.to_csv('./data/splittable_words.csv')
        
    return splittable_words

# Dependencies
import numpy as np
import pandas as pd

# Load words dataset table
def load_words(path):
    return pd.read_csv(path, dtype={
        'tweet': np.unicode_,
        'index': np.int,
        'text': np.unicode_,
        'pos': np.unicode_,
        'conf': np.float        
    })
                       
# Load tweets dataset
def load_tweets(path):
    return pd.read_csv(
        path, 
        parse_dates=['created_at'], 
        dtype={
           'id_str': np.unicode_,
           'text': np.unicode_
        })
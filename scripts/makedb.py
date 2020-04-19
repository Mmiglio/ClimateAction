# Set root directory
import sys, os; sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Dependencies
from modules.dataset.tweets import Tweets
import argparse
import json


# Main
if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser()
    # Must the output files be overwritten? (T/F)
    parser.add_argument('--overwrite', type=bool, default=True)
    # Raw tweets input file (.jsonl format)
    parser.add_argument('--in_tweets', type=str, required=True)
    # Tweets formatted table output file (.json format)
    parser.add_argument('--out_tweets', type=str, required=True)
    # Hashtags formatted table output file (.json format)
    parser.add_argument('--out_hashtags', type=str, required=True)
    # Words formatted table output file (.json format)
    parser.add_argument('--out_words', type=str, required=True)
    # List of substitutions dictionaries (.json format)
    parser.add_argument('--in_subs', nargs='+', type=str, default=[])
    # Parse arguments
    args = parser.parse_args()

    # Instantiate new tweets table
    tweets = Tweets()
    # Parse tweets from input .jsonl file
    tweets.from_json_list(in_path=args.in_tweets)
    # Store tweets table to .json formatted file
    tweets.to_json(out_path=args.out_tweets)

    # Show tweets DataFrame head
    print('Tweets table:')
    print(tweets.df.head())
    print()

    # Load substitution dictionaries
    subs = {}
    # Loop through each substitutions dictionary .json file
    for subs_path in args.in_subs:
        # Open substitution dictionary file
        with open(subs_path, 'r') as subs_file:
            subs = {**subs, **json.load(subs_file)}

    # Retrieve words and hashtags from tweets
    hashtags, words = tweets.get_entities(subs=subs)
    # Store hashtags table to .json formatted file
    hashtags.to_json(out_path=args.out_hashtags)
    # Store words table to .json formatted file
    words.to_json(out_path=args.out_words)

    # Show hashtags DataFrame head
    print('Hashtags table:')
    print(hashtags.df.head())
    print()

    # Show words DataFrame head
    print('Words table:')
    print(words.df.head())
    print()

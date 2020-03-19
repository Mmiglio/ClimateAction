from langdetect import detect
import json_lines
import pandas as pd
import os

group = "post"
data_path = "../data/{}Greta.jsonl".format(group)
output_path = "../data/tweets_{}Greta.csv".format(group)

def main():
    tweets = pd.DataFrame(load_jsonl(data_path))

    # save tweets to file
    tweets.to_csv(output_path, index=False)


def load_jsonl(file):
    with open(file, 'rb') as f:
        # extract relevant fields from tweets. Be aware that replies
        # have a different structure. For example, assuming we would
        # like to extract hashtags we need to distinguish between different cases
        # (other fields return truncated text and hastags)
        list_tweets = list()
        cnt = 0
        for tweet in json_lines.reader(f, broken=True):          
            # tweet is a reply / retweet
            if 'retweeted_status' in tweet:
                # get retweets of original tweet
                original_retweets = tweet['retweeted_status'].get('retweet_count', None)
                original_favorite = tweet['retweeted_status'].get('favorite_count', None)
                try:
                    full_tweet = tweet['retweeted_status']['extended_tweet']
                    tweet_text = full_tweet['full_text']
                except:
                    # text and hashtags have not been truncated
                    full_tweet = tweet['retweeted_status']
                    tweet_text = full_tweet['text']
            # no RT
            else:
                original_retweets = None
                original_favorite = None
                try:
                    full_tweet = tweet['extended_tweet']
                    tweet_text = full_tweet['full_text']
                except:
                    full_tweet = tweet
                    tweet_text = full_tweet['text']
            
            # filter tweets with lang != eng
            # this operation is extremely slow...  
            if detect(tweet_text) != 'en':
                continue
            else:
                list_tweets.append({
                    'created_at': tweet['created_at'],
                    'id': tweet['id'],
                    'text': tweet_text,
                    'retweet_count': tweet.get('retweet_count', None),
                    'favorite_count': tweet.get('favorite_count', None),
                    'original_retweet_count': original_retweets,
                    'original_favorite': original_favorite
                })
            
            # logging
            cnt += 1
            if cnt % 500 == 0:
                print("Processed {} tweets".format(cnt))
        return list_tweets


if __name__ == "__main__":
    # set working directory
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    
    main()


# Dependencies
import sys, os; sys.path.insert(1, os.path.join(sys.path[0], '..'))
from modules.dataset.tweets import API_PRODUCT_30DAY, API_PRODUCT_FULL
from modules.dataset.tweets import Tweets
from datetime import datetime, date, timedelta
from random import randrange
import argparse
import re


# Constants
AUTH_PATH = 'data/auth.json'  # Path to deafult authentication credentials file


# Define function for sampling datetime intervals (from date - to date)
def sample_intervals(from_date, to_date, window=(1, 0, 0)):
    """
    Sample tweets from every day betweet start date and end date, using
    a temporal window of a spciefied width (hours, minutes, seconds).

    Input
    1. from_date: starting interval date. Datetime object;
    2. to_date: ending interval date. Datetime object;
    3. window_size: width of the sampling window;

    Output
    1. samples: list of windows. Tuple (start, end) of datetime objects;
    """
    # Initialize list of samples
    samples = list()
    # Define datetime bottom scale (earliest available date)
    bottom_scale = datetime(1970, 1, 1)
    # Retireve window size parameters
    hours, minutes, seconds = window
    # Define window as timedelta (time difference) object
    window = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    # Initialize current date as interval's start date
    curr_date = from_date
    # Loop through each day between start and end date
    while curr_date < to_date:
        # Define next date as next day
        next_date = curr_date + timedelta(days=1)
        # Define first available sampling datetime for current day
        first_datetime = datetime.combine(curr_date, datetime.min.time())
        # Define last available sampling datetime as next day 00:00:00
        last_datetime = datetime.combine(next_date, datetime.min.time())
        last_datetime = last_datetime - window
        # Express first and last datetimes as seconds from Jan 01 1970 (standard)
        first_seconds = int((first_datetime - bottom_scale).total_seconds())
        last_seconds = int((last_datetime - bottom_scale).total_seconds())
        # Randomly sample one datetime between first and last datetime (seconds from first available datetime)
        ws_seconds = randrange(start=0, stop=last_seconds-first_seconds, step=1)
        # Compute window start as datetime adding seconds from first available date
        ws_datetime = first_datetime + timedelta(seconds=ws_seconds)
        # Compute window end as datetime
        we_datetime = ws_datetime + window
        # Store current sample
        samples.append((ws_datetime, we_datetime))
        # Update current date
        curr_date = next_date
    # Return samples
    return samples


# Main
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser()
    # Must the output file be overwritten? (T/F)
    parser.add_argument('--overwrite', type=bool, default=False)
    # Start date of download period (iso format YYYY-mm-dd)
    parser.add_argument('--from_date', type=str, required=True)
    # End date of download period (iso format YYYY-mm-dd)
    parser.add_argument('--to_date', type=str, required=True)
    # Window expressed as hours, minutes, seconds (default 1 hour)
    parser.add_argument('--window', nargs='+', type=int, default=[])
    # Keywords list to be used in query
    parser.add_argument('--keywords', nargs='+', type=str, default=[])
    # Filter tweets language (according to twitter language)
    parser.add_argument('--language', type=str, default='en')
    # Number of tweets retrieved for each request
    parser.add_argument('--batch_size', type=int, default=100)
    # Output file path, where to store data (.json formatted)
    parser.add_argument('--out_path', type=str, required=True)
    # Authentication credentials file path
    parser.add_argument('--auth_path', type=str, default=AUTH_PATH)
    # Twitter-level application name
    parser.add_argument('--label', type=str, required=True)
    # Twitter.level product name
    parser.add_argument('--product', type=str, default=API_PRODUCT_30DAY)
    # Parse arguments to dictionary
    args = parser.parse_args()

    # Get start date and interval (in days)
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    # Get sampling window
    hours, minutes, seconds = 1, 0, 0  # Define default window
    window_len = len(args.window)  # Get window size
    # Get hours, minutes and seconds from parameters
    hours = args.window[0] if window_len >= 1 else hours
    minutes = args.window[1] if window_len >= 2 else minutes
    seconds = args.window[2] if window_len >= 3 else seconds
    # Define window
    window = (hours, minutes, seconds)

    # Clean and add hashtag to keywords
    keywords = args.keywords
    keywords = [re.sub(r'[\#\@ ]', '', kw) for kw in keywords]  # Remove hashtags and whitespaces
    keywords = ['#' + kw for kw in keywords]  # Add hashtags at the beginning of the keyword

    # Get language
    language = args.language

    # Make query (docs here: https://developer.twitter.com/en/docs/tweets/search/overview/premium#AvailableOperators)
    query = ''  # Initialize query
    query = query + (' OR '.join(keywords) if keywords else '')  # Add keywords
    query = query + (' lang:{0:s}'.format(language) if language else '')  # Add language

    # Get API label and product
    label = args.label
    product = args.product

    # Get retrieved tweets batch size
    batch_size = args.batch_size

    # Get output file path
    out_path = args.out_path
    # Get flag to decide to overwrite existing file or not
    overwrite = args.overwrite
    # Get authentication credentials files
    auth_path = args.auth_path

    # Get sampled intervals
    samples = sample_intervals(from_date, to_date, window=window)

    # Instantiate new Tweets dataset
    tweets = Tweets()
    # Authenticate using auth file
    tweets.auth_from_json(in_path=auth_path)

    # Check if output file must be overwritten
    if overwrite:
        # Create empty file
        open(out_path, 'w', encoding='utf-8').close()

    # Open output file connection
    with open(out_path, 'a', encoding='utf-8'):
        # Loop through each sampling interval
        for i, (ws_datetime, we_datetime) in enumerate(samples):
            # Get tweets for the sampled interval
            tweets.search_tweets(
                query=query,
                from_date=ws_datetime,
                to_date=we_datetime,
                batch_size=batch_size,
                label=label,
                product=product
            )

    # Store tweets to file
    tweets.to_json(out_path=out_path)

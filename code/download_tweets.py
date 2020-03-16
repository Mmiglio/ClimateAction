from TwitterAPI import TwitterAPI, TwitterPager
import numpy as np
import datetime
import json
import time

# Product: 30day or fullarchive
# Label: label used during the creation of the dev-environment
PRODUCT = 'fullarchive'
LABEL = 'collectPreGreta'

# Start day for the requests
# Format of dates <yyyy-mm-dd>
# Interval: number of days between two requests (1 = 1 request per day)
START_DATE = '2018-04-22'
INTERVALL_REQUESTS = 1

# Hashtags used to create the query 
KEYWORD_HASHTAGS = [
    "climatechange",
    "gretathunberg",
    "climatestrike",
    "fridays4future",
    "climatecrisis",
    "parisagreement"
]

# File use to stored tweets
RESULTS_FILE = 'results_22april2018.jsonl'

# number of total requests
NUM_REQUESTS = 50

# path to the json containing keys for the api
SECRETS_FILE = 'secrets.json'


def authenticate(secrets_file_path=SECRETS_FILE):
    """
    Load credentials stored in secrets_file_path.json
    Return TwitterAPI object created using credentials
    """
    with open(secrets_file_path, 'r') as secrets_file:
        secrets = json.load(secrets_file)

    api = TwitterAPI(
        consumer_key=secrets['consumer_key'],
        consumer_secret=secrets['consumer_secret'],
        access_token_key=secrets['access_token'],
        access_token_secret=secrets['access_token_secret']
        )
    return api


def get_tweets(api, query, from_date, to_date):
    """
    Return 100 tweets for a specific date
    date format is <yyyymmddhhmm> (201912250000)
    """
    r= api.request('tweets/search/{}/:{}'.format(PRODUCT, LABEL), 
                    {
                        'query':query,
                        'maxResults':100, 
                        'fromDate': from_date, 
                        'toDate': to_date
                    }
                    )
    return r


# create list of dates (1 request per day at random hour)
def create_dates(start_date, num_requests=50):
    """
    Return list of dates to be used in the query
    1 request (100 tweets) per day at a random hour/minute
    params: start date in the form <yyyy-mm-gg>
    """
    year, month, day = [int(x) for x in start_date.split('-')]
    from_date = datetime.datetime(year=year, month=month, day=day) 
    to_date = from_date + datetime.timedelta(days=INTERVALL_REQUESTS) 

    dates_list = list()
    for i in range(num_requests):
        hour = np.random.randint(24)
        minute = np.random.randint(60)

        # Add to the list date in the format <yyyymmddhhmm>
        dates_list.append(
            {
                'from_date': from_date.strftime('%Y%m%d')+'{0:0=2d}{1:0=2d}'.format(hour, minute),
                'to_date' :  to_date.strftime('%Y%m%d')+'{0:0=2d}{1:0=2d}'.format(hour, minute)           
            }
        )
        from_date = to_date
        to_date += datetime.timedelta(days=INTERVALL_REQUESTS) 

    return dates_list


def main():
    dates_list = create_dates(start_date=START_DATE,num_requests=NUM_REQUESTS)

    # create query, list of available operators here: 
    # https://developer.twitter.com/en/docs/tweets/search/overview/premium#AvailableOperators
    query = " OR ".join(list(map(lambda kw: "#"+kw, KEYWORD_HASHTAGS)))
    query += " lang:en"

    # authenticate
    api = authenticate()

    # loop over dates
    counter = 1
    for date in dates_list:
        print("Iteration {}/{}".format(counter, NUM_REQUESTS))
        print("Getting tweets from: {}".format(
            datetime.datetime.strptime(date['from_date'], "%Y%m%d%H%M")
            ))
        num_tweets = 0
        with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
            # get tweets
            tweets = get_tweets(
                api,
                query,
                from_date=date['from_date'],
                to_date=date['to_date']
            )
            # write tweets to file
            for tweet in tweets:
                num_tweets += 1
                json.dump(tweet, f)
                f.write('\n')          
        print('\tDownloaded {} tweets'.format(num_tweets))
        counter += 1
        time.sleep(5)


if __name__ == "__main__":
    main()
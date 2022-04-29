# Importing Libraries
import os
import shutil
import datetime

import pandas as pd
import snscrape.modules.twitter as sntwt
from snscrape.base import ScraperException

from tqdm import tqdm

project_path = ".."


# Defining Required Functions
class DummyProgressBar:
    def update(self, n=1): pass
    def close(self): pass

def get_new_path(path):
    if not os.path.isfile(path):
        return path
    suffix = 1
    name, ext = path.rsplit('.', 1)
    while True:
        path = f"{name} ({suffix}).{ext}"
        if not os.path.isfile(path):
            return path
        suffix += 1

def get_tweets_of_day(keyword: str, date: datetime.datetime, directory: str, verbose: bool = False, 
                      max_number: int = -1):
    if max_number == -1:
        max_number = float('inf')

    if not os.path.isdir(directory):
        os.makedirs(directory)

    start_date = date.strftime("%Y-%m-%d")
    end_date = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    TWEET_PER_FILE = 10000

    tweets = pd.DataFrame(columns=['url', 'date', 'content', 'tweet_id', 'user_id'])
    tweet_count = 0
    # users = pd.DataFrame(columns=['user_id', 'username'])

    scraper = sntwt.TwitterSearchScraper(f"{keyword} since:{start_date} until:{end_date}")
    tweet_generator = scraper.get_items()
    generator_is_empty = False

    progress_bar = tqdm() if verbose else DummyProgressBar()
    try:
        while True:
            tweet_count = min(max_number, TWEET_PER_FILE)
            tweets = tweets[0:0]

            for _ in range(tweet_count):
                try:
                    tweet = next(tweet_generator)
                    tweets = tweets.append({
                        'url': tweet.url, 
                        'date': tweet.date, 
                        'content': tweet.content, 
                        'tweet_id': str(tweet.id), 
                        'user_id': tweet.username
                    }, ignore_index=True)
                    progress_bar.update()
                    tweet_count += 1
                except StopIteration:
                    generator_is_empty = True
                    break

            file_name = os.path.join(directory, f"{tweet.date.strftime('%Y-%m-%d %H-%M-%S')}.csv")
            file_name = get_new_path(file_name)
            tweets.to_csv(file_name, index=False)
            
            if max_number < TWEET_PER_FILE or generator_is_empty:
                break
            
            max_number -= TWEET_PER_FILE
        
        progress_bar.close()
    except KeyboardInterrupt as error:
        progress_bar.close()
        raise error
    except ScraperException as error:
        progress_bar.close()
        print(f"Error occured on date: {start_date}")
        print(f"Number of tweets downloaded: {tweet_count}")
        print(f"Last time: {tweet.date.strftime('%Y-%m-%d %H-%M-%S')}")
        print()
        print(error)
    
    shutil.make_archive(f'{keyword} {start_date}', 'zip', directory)
    
    if os.path.isfile(f'{keyword} {start_date}.zip'):
        shutil.rmtree(directory)

    shutil.copy2(f'{keyword} {start_date}.zip', os.path.join(project_path, 'data'))
    os.remove(f'{keyword} {start_date}.zip')


# Collecting Data
# February 2022
for d in range(24, 29):
    get_tweets_of_day(
        keyword='russia ukraine', 
        date=datetime.datetime(2022, 2, d), 
        directory='./data/russia ukraine', 
        verbose=True
    )

# March 2022
for d in range(1, 10):
    get_tweets_of_day(
        keyword='russia ukraine', 
        date=datetime.datetime(2022, 3, 1), 
        directory='./data/russia ukraine', 
        verbose=True
    )

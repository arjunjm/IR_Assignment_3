import sys
import tweepy
import time

class TwitterCrawler():
    consumer_key    = "WAjAojXmneZ9BMVQDBWHA"
    consumer_secret = "B5w0yACGDDGifjHzOkSECublalsEH6Ugf1AmcEg"
    access_token    = "77746210-hSPuLRHhFFbXCC1plctjMs4THogZP8VtlffZHycD7"
    access_token_secret = "ADDf2jk3t8AhPNL9A4GpM3LC8WpwwPsXIrQc3Nz9kvVO0"
    auth            = None
    api             = None

    def __init__(self):
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth)

    def get_first50_tweets(self, query):
        tweet_list = []
        i = 0
        for tweet in tweepy.Cursor(self.api.search,
                           q=query,
                           rpp=100,
                           result_type="recent",
                           include_entities=True,
                           lang="en").items():
            i += 1
            tweet_list.append(tweet.text)
            if (i == 50):
                break
        return tweet_list

def main():
    tweet_crawler = TwitterCrawler()
    tweet_list = tweet_crawler.get_first50_tweets('#katyperry')
    print tweet_list

if __name__ == '__main__':
    main()

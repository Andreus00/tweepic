import tweepy
import json
import datetime
import twint
from src.dataset.tweet_filter import TweetFilter

class TweetRetriever:

    def __init__(self) -> None:
        with open("data/token.json") as f:
            self.token = json.load(f)
        self.api = tweepy.Client(**self.token)
        self.tf = TweetFilter()

    def tweet_to_list(self, tweet):
        return [tweet.id, tweet.created_at.year, tweet.created_at.month, tweet.created_at.day, tweet.text]

    def get_tweet(self, id):
        # return self.tweet_to_list(self.api.get_tweet(id=id, tweet_fields=["created_at", "text"]).data)
        tweet = self.api.get_tweet(id=id, tweet_fields=["created_at", "text"]).data
        if tweet == None:
            return None
        text, hashtags, mentions = self.tf.filter_tweet(tweet.text)
        return [tweet.id, tweet.created_at.year, tweet.created_at.month, tweet.created_at.day, text, hashtags, mentions]
    
    def search(self, query, count=10, toDate=datetime.datetime.now() - datetime.timedelta(seconds=9), tweet_fields=["created_at", "text"]):
        print(toDate)
        result = self.api.search_recent_tweets(query=query, max_results=count, end_time=toDate, tweet_fields=tweet_fields)
        return [self.tweet_to_list(tweet) for tweet in result.data]
        

if __name__ == "__main__":

    twert_ret = TweetRetriever()
    tweets = twert_ret.search("python", count=10)
    print(tweets)
    
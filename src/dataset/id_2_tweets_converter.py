import pyspark
import os
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
import config
import src.tweet_stream.tweet_stream as ts
import tqdm


class Id2TweetsConverter:
    def __init__(self, folder_path="data/twitter-events-2012-2016/"):
        self.folder_path = folder_path
        self.files = os.listdir(self.folder_path)
        print(len(self.files))
        # Create the session
        conf = SparkConf().\
        set('spark.ui.port', "4050").\
        set('spark.executor.memory', '4G').\
        set('spark.driver.memory', '45G').\
        set('spark.driver.maxResultSize', '10G').\
        set('spark.sql.randomSeed', config.SEED).\
        setAppName("PySparkTutorial").\
        setMaster("local[*]")

        # Create the context
        self.sc = pyspark.SparkContext(conf=conf)
        self.spark = SparkSession.builder.getOrCreate()

        log4jLogger = self.sc._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(__name__)
        self.logger.info("pyspark script logger initialized")

        self.tweet_retriever = ts.TweetRetriever()

    def _open_file(self, path):
        return self.sc.textFile(path)

    def open_all_files(self):
        self.logger.warn("Opening all files")
        files = []
        for file in self.files:
            opened_file = self._open_file(self.folder_path + file)
            # add the name of the file to each line
            files.append(opened_file.map(lambda x: (file[:-4], x)))
        return self.sc.union(files)
        # return self.sc.union([self._open_file(self.folder_path + file) for file in self.files])
    
    def select_n_tweets(self, tweets, n=10, seed=config.SEED):
        self.logger.warn(f"Selecting {n} tweets")
        return tweets.sample(False, n/tweets.count(), seed=seed)
    
    def _id_2_tweet(self, id):
        return self.tweet_retriever.get_tweet(id)

    def id_2_tweet(self, list_of_tweets):
        self.logger.warn(f"Converting ids to tweets")
        pbar = tqdm.tqdm(list_of_tweets, total=len(list_of_tweets))
        return self.sc.parallelize([[row[0]] + self._id_2_tweet(row[1]) for row in pbar if self._id_2_tweet(row[1]) is not None])

    def collect_tweets(self, rdd):
        self.logger.warn(f"Collecting tweets")
        return rdd.collect()
    
    
    def get_n_tweets(self, n=10):
        tweets = self.open_all_files()
        sampled_tweets = self.select_n_tweets(tweets, n=n)
        sampled_tweets = self.collect_tweets(sampled_tweets)
        sampled_tweets = self.id_2_tweet(sampled_tweets)
        return sampled_tweets


def test():
    converter = Id2TweetsConverter()
    sampled_tweets = converter.get_n_tweets(n=10_000)
    df = converter.spark.createDataFrame(sampled_tweets, ["label", "id", "year", "month", "day", "text", "mentions", "hashtags"])

    # save the dataframe
    df.write.parquet("data/parquet/tweets.parquet")
    print("Sampled tweets:")


    print("Dataframe:")
    df.show()



    print(sampled_tweets)
    

    
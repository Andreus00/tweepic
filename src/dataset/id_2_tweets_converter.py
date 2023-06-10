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
        self.tweets = None
        self.files = os.listdir(self.folder_path)
        self.files.remove("2014-sydneysiege.ids")
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

    def _open_file(self, path, file):
        return self.sc.textFile(path + file).map(lambda x: (file, x))

    def open_all_files(self):
        self.logger.warn("Opening all files")
        files = []
        for file in self.files:
            print(file)
            opened_file = self._open_file(self.folder_path, file)
            # add the name of the file to each line
            files.append(opened_file)
        return self.sc.union(files)
    
    def select_n_tweets(self, rdd, n=10, seed=config.SEED):
        self.logger.warn(f"Selecting {n} tweets")
        return rdd.takeSample(False, n, seed=seed)
    
    def _id_2_tweet(self, id):
        return self.tweet_retriever.get_tweet(id)

    def id_2_tweet(self, list_of_tweets):
        self.logger.warn(f"Converting ids to tweets")
        pbar = tqdm.tqdm(list_of_tweets, total=len(list_of_tweets))

        tweets = []
        for row in pbar:
            tweet = self._id_2_tweet(row[1])
            if tweet is not None:
                tweets.append([row[0]] + tweet)
        return self.sc.parallelize(tweets)

    def collect_tweets(self, rdd):
        self.logger.warn(f"Collecting tweets")
        return rdd.collect()
    
    def get_n_tweets(self, n=10):
        if self.tweets is None:
            self.tweets = self.open_all_files()
        sampled_tweets = self.select_n_tweets(self.tweets, n=n)
        sampled_tweets = self.id_2_tweet(sampled_tweets)
        return sampled_tweets
    
    def tweets_fetcher(self, n, batch_size = 300):
        self.tweets = self.open_all_files()
        sampled_tweets = self.select_n_tweets(self.tweets, n=n)
        cur_idx = 0
        while True:
            tw = sampled_tweets[cur_idx:cur_idx+batch_size]
            cur_idx += batch_size
            yield self.id_2_tweet(tw)



def generate_dataset():
    converter = Id2TweetsConverter()
    df_name = "data/parqurt_dataset"
    df = converter.spark.createDataFrame(data=[["2012-obama-romney.ids", "266035659821703169", 2012, 11, 7, "i dont think many people round these parts are particularly content with obama getting re-elected but i mean its just my guess"]],
                                         schema=["label", "id", "year", "month", "day", "text"])
    for i, new_tweets in enumerate(converter.tweets_fetcher(100_000, 300)):
        df_new = converter.spark.createDataFrame(new_tweets, ["label", "id", "year", "month", "day", "text"])
        df_new = df_new.withColumn("text", regexp_replace("text", "^RT ", ""))

        df = df.union(df_new)
        df.write.parquet(df_name, mode="overwrite")

        print("Dataframe:")
        df.show(truncate=False)
    
def test():
    converter = Id2TweetsConverter()
    tweets = converter.open_all_files()
    sampled_tweets = converter.select_n_tweets(tweets, n=100)
    sampled_tweets = converter.id_2_tweet(sampled_tweets)
    df = converter.spark.createDataFrame(sampled_tweets, ["label", "id", "year", "month", "day", "text"])

    print("Dataframe:")
    df.show()

    # check statistics of the dataframe
    df.describe().show()

def load_and_show():
    converter = Id2TweetsConverter()
    df_name = "data/parqurt_dataset"
    df = converter.spark.read.parquet(df_name)
    print("Dataframe:")
    df.show(truncate=False)

    # check statistics of the dataframe
    df.describe().show()
    


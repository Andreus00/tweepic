import os
import pyspark
from pyspark import SparkConf



folder_path = "data/twitter-events-2012-2016/"


files_path = os.listdir(folder_path)

conf = SparkConf().\
set('spark.ui.port', "4050").\
set('spark.executor.memory', '4G').\
set('spark.driver.memory', '45G').\
set('spark.driver.maxResultSize', '10G').\
set('spark.sql.randomSeed', 1).\
setAppName("PySparkTutorial").\
setMaster("local[*]")

# Create the context
sc = pyspark.SparkContext(conf=conf)

spark = pyspark.sql.SparkSession.builder.getOrCreate()


files = []
for file in files_path:
    print(file)
    opened_file = sc.textFile(folder_path + file).map(lambda x: (file, x)).toDF()
    # add the name of the file to each line
    files.append(opened_file.rdd)
files = sc.union(files).toDF(["file", "id"])
print(files.take(1))

old_df = spark.read.parquet("data/parquet_bkp/tweets.parquet")

# join the two dataframes where the tweet ids are the same

old_df = old_df.join(files, old_df.id == files.id, how="inner")

# delete column id from the second dataframe
old_df = old_df.drop(files.id)
old_df = old_df.drop("label")

# rename the column file to label
old_df = old_df.withColumnRenamed("file", "label")

old_df = old_df.select("label", "id", "year", "month", "day", "text", "mentions", "hashtags")

old_df.show()

old_df.write.parquet("data/parquet_big/tweets.parquet")
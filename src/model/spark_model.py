import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark
import numpy as np
from scipy.spatial import distance
from pyspark.ml.feature import VectorAssembler, HashingTF
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyspark.sql.functions import lit 
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from sklearn.decomposition import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import BucketedRandomProjectionLSH
from src.model import pipeline
from pyspark.storagelevel import StorageLevel
from src.utils import *
from src.model import pipeline_v2 
from src.model import graph_pipeline
from igraph import *
import config


def main():

    SAVE_EMBEDDINGS_INTERMEDIATE = False
    LOAD_EMBEDDINGS_INTERMEDIATE = True
    SAVE_PROXIMITY_INTERMEDIATE = True
    LOAD_PROXIMITY_INTERMEDIATE = False
    # create label to index dictionary
    l2c = init_labels()
    
    # init spark session
    spark = sparknlp.start(gpu=True, memory=config.spark_memory, params={"spark.jars.packages": "graphframes:graphframes:0.8.1-spark3.0-s_2.12",
                                                           "spakr.driver.memory": config.spark_memory,
                                                           "spark.executor.memory": config.executor_memory,
                                                           "spark.driver.maxResultSize": config.max_result_memory,
                                                           "spark.memory.fraction": config.in_memory_fraction,
                                                           "spark.sql.adaptive.enabled": "true",
                                                           "spark.sql.adaptive.skewJoin.enabled": "true",
                                                           "spark.sql.shuffle.partitions": "40",})
    
    from graphframes import GraphFrame


    # df = spark.read.parquet("test_save")
    # df.show()
    # exit()

    # g = GraphFrame(v, e)
    # load data
    if LOAD_EMBEDDINGS_INTERMEDIATE:
        result = spark.read.parquet(config.intermediate_embeddings_path)
        print("## LOADED ##")
    else:
        df = load_data(config.dataset_path, spark)
        # cast year, month e day to int
        df = df.withColumn("year", F.col("year").cast("int"))
        df = df.withColumn("month", F.col("month").cast("int"))
        df = df.withColumn("day", F.col("day").cast("int"))
        
        # get label count
        label_count = 31 #  len(df.groupBy("label").count().collect())


        # create the embedding pipeline
        nlp_pipeline = pipeline_v2.create_embeddings_pipeline(n_buckets=config.n_buckets)

        # fit and transform pipeline
        pipeline_model = nlp_pipeline.fit(df)
        result = pipeline_model.transform(df)
        result.show()   # output: | label | id |
                        #         text | document | language |
                        #         sentence | token | hashtag_idxs |
                        #         mentions_idxs | punctuation_idxs | hashtags |
                        #         word_embeddings | sentence_embeddings | hashtags_embeddings| date_embeddings  | bucket |
        
        # free memory from the old df
        df.unpersist(blocking=True)


        if SAVE_EMBEDDINGS_INTERMEDIATE:
            result.write.mode("overwrite").parquet(config.intermediate_embeddings_path)
            print("## SAVED ##")
        

    ### SENTENCE PROXIMITY
    if LOAD_PROXIMITY_INTERMEDIATE:
        sentence_proximity_input = spark.read.parquet(config.intermediate_proximity_path)
        sentence_proximity_input.show()
    else:
        sentence_proximity_pipeline = graph_pipeline.create_sentence_proximity_pipeline(n_neighbors=config.n_neighbors)
        sentence_proximity_input = result.select("id", "text",  "sentence_embeddings", "word_embeddings", "hashtags_embeddings", "time_bucket")
        result.unpersist(blocking=True)
        sentence_proximity_input.cache()
        sentence_proximity_pipeline_model = sentence_proximity_pipeline.fit(sentence_proximity_input)
        sentence_proximity_pipeline_result = sentence_proximity_pipeline_model.transform(sentence_proximity_input)
    

        ### WORD AND HASHTAGS PROXIMITY
        word_and_hashtag_proximity_pipeline = graph_pipeline.create_word_and_hashtag_proximity_pipeline(n_words=config.n_words, n_hashtags=config.n_hashtags)
        word_and_hashtag_proximity_input = sentence_proximity_pipeline_result.select("id", "text", "word_embeddings", "hashtags_embeddings", "neighbors")
        word_and_hashtag_proximity_pipeline_model = word_and_hashtag_proximity_pipeline.fit(word_and_hashtag_proximity_input)
        word_and_hashtag_proximity_pipeline_result = word_and_hashtag_proximity_pipeline_model.transform(word_and_hashtag_proximity_input)
        if SAVE_PROXIMITY_INTERMEDIATE:
            word_and_hashtag_proximity_pipeline_result.write.parquet(config.intermediate_proximity_path, mode="overwrite")
            exit()


    ### GENERATE THE GRAPH
    proximity_graph = GraphFrame(result.select("id", "label", "text"), word_and_hashtag_proximity_pipeline_result)

    proximity_graph.inDegrees.show()
    ig = Graph.TupleList(proximity_graph.edges.collect(), directed=True)
    plot(ig).save("graph.png")

    # print("## GRAPH DONE ##")
    # print stuff
    # graph_result.show(truncate=False)
    # sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect())
    # get sentence embeddings
    # sentence_embeddings = sentence_embeddings.reshape(sentence_embeddings.shape[0], -1)

    # hashtag_embeddings = result.select("hashtags_embeddings").collect()
    # for each element get the average of the embeddings if there are at least 1 hashtag otherwise put a zero vector
    # hashtag_embeddings = np.asarray([np.mean(x[0], axis=0) if len(x[0]) > 0 else np.zeros((768,)) for x in hashtag_embeddings])
    


    # remove texts where there are no hashtags
    texts = result.select("text").collect()
    # remove texts where there are no hashtags
    labels = [l2c[row.label] for row in result.select("label").collect()]

    # get texts, clusters and rf predictions
    # clusters = [x.cluster for x in result.select("cluster").collect()]
    # rf_pred = result.select("prediction").collect()

    # print(result.filter(result.id == "1234").select("text").collect())
    # get labels

    # plots
    # plot_cluster(sentence_embeddings, labels, None, rf_pred=None)
    # text_similarity(texts, sentence_embeddings, labels, q=10)


def francesco_stuff():


    # create label to index dictionary
    l2c = init_labels()
    
    # init spark session
    spark = sparknlp.start(gpu=True, memory="25G", params={"spark.jars.packages": "graphframes:graphframes:0.8.1-spark3.0-s_2.12",
                                                           "spakr.driver.memory": "25G",
                                                           "spark.executor.memory": "1G",
                                                           "spark.driver.maxResultSize": "25G",
                                                           "spark.memory.fraction": "0.8",
                                                           "spark.sql.adaptive.enabled": "true",
                                                           "spark.sql.adaptive.skewJoin.enabled": "true",
                                                           "spark.sql.shuffle.partitions": "40",})

    df = load_data(spark).drop("hashtags", "mentions", "text_with_info")
    
    # get label count
    label_count = 31 #  len(df.groupBy("label").count().collect())

    
    # create pipeline
    # for each entry in df add one hashtag
    df = df.withColumn("hashtags", F.array(F.lit("hashtag")))
    hashtag_pipeline = pipeline.create_pipeline2(label_count=label_count, l2c=l2c)
    nlp_pipeline = pipeline.create_embedding_pipeline(label_count=label_count, l2c=l2c)
    pipeline_model = nlp_pipeline.fit(df)
    result = pipeline_model.transform(df)
    hashtag_pipeline_model = hashtag_pipeline.fit(result)
    hashtag_result = hashtag_pipeline_model.transform(result)
    sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect())
    hashtags_embeddings = np.asarray(hashtag_result.select("hashtags_embeddings").collect())
    # merge embeddings
    final_embeddings = np.concatenate((sentence_embeddings, hashtags_embeddings), axis=1)
    print(final_embeddings.shape)
    print(final_embeddings)
    print(result.select("informations").collect()[4])
    print(result.take(1))
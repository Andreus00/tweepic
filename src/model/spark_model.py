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
import src.model.embedding_average as embedding_average
from sklearn.decomposition import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import BucketedRandomProjectionLSH
from src.model.cluster import TweetEmbeddingPreprocessing, LabelsToIndex
from src.model import pipeline
from pyspark.storagelevel import StorageLevel
from src.utils import *
from src.model import pipeline_v2 




def main():

    FRANCESCO = False # Always True 
    SAVE_INTERMEDIATE = False
    LOAD_INTERMEDIATE = False
    print("FRANCESCO MODE: ", FRANCESCO)
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
    
    from graphframes import GraphFrame

    # g = GraphFrame(v, e)

    # load data
    if LOAD_INTERMEDIATE:
        result = spark.read.parquet("intermediate_result.parquet")
        result = result.drop("document", "sentence", "rawPrediction", "probability", "hashtags", "mentions")#.sample(False, 0.01, seed=0)
        print(result.columns)
        print("## LOADED ##")
    else:
        df = load_data(spark).drop("hashtags", "mentions", "text_with_info")
        
        # get label count
        label_count = 31 #  len(df.groupBy("label").count().collect())

        
        # create pipeline
        if FRANCESCO:
            # for each entry in df add one hashtag
            df = df.withColumn("hashtags", F.array(F.lit("hashtag")))
            hashtag_pipeline = pipeline.create_pipeline2(label_count=label_count, l2c=l2c)
            nlp_pipeline = pipeline.create_embedding_pipeline(label_count=label_count, l2c=l2c)
        else:
            nlp_pipeline = pipeline_v2.create_final_pipeline()

            # fit and transform pipeline
            pipeline_model = nlp_pipeline.fit(df)
            result = pipeline_model.transform(df)
            result.show()
            
            # free memory from the old df
            df.unpersist(blocking=True)


    if SAVE_INTERMEDIATE:
        result.write.mode("overwrite").parquet("intermediate_result.parquet")
        print("## SAVED ##")


    if FRANCESCO:
         hashtag_pipeline_model = hashtag_pipeline.fit(result)
         hashtag_result = hashtag_pipeline_model.transform(result)
    else:
        graph_input = result.select("id", "sentence_embeddings", "word_embeddings")
        graph_input.cache()
        result.unpersist(blocking=True)
        graph_pipeline = pipeline.create_graph_pipeline()
        graph_pipeline_model = graph_pipeline.fit(graph_input)
        graph_result = graph_pipeline_model.transform(graph_input)


        # todo: fix this
        # word_distance_pipeline = pipeline.create_word_distance_pipeline()
        # word_distance_input = result.select("id", "sentence_embeddings", F.col("year").cast("int"),  F.col("month").cast("int"),  F.col("day").cast("int"))
        # word_distance_input.cache()
        # result.persist(storageLevel=StorageLevel.DISK_ONLY)
        # word_distance_pipeline_model = word_distance_pipeline.fit(word_distance_input)
        # word_distance_pipeline_result = word_distance_pipeline_model.transform(word_distance_input)
        # word_distance_pipeline_result.show(truncate=False)

        print("## GRAPH DONE ##")


    # print stuff
    result.show()   
    if FRANCESCO:
        sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect())
        hashtags_embeddings = np.asarray(hashtag_result.select("hashtags_embeddings").collect())
        # merge embeddings
        final_embeddings = np.concatenate((sentence_embeddings, hashtags_embeddings), axis=1)
        print(final_embeddings.shape)
        print(final_embeddings)
        print(result.select("informations").collect()[4])
        print(result.take(1))
    else:
        # for el in (graph_result.select("hashes").collect()[0:3]):
        #     print(el)
        graph_result.show(truncate=False)
        
    sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect())
    # get sentence embeddings
    sentence_embeddings = sentence_embeddings.reshape(sentence_embeddings.shape[0], -1)

    # get texts, clusters and rf predictions
    texts = result.select("text").collect() 
    clusters = None if FRANCESCO else [x.cluster for x in result.select("cluster").collect()]
    rf_pred = None if FRANCESCO else result.select("prediction").collect()

    # print(result.filter(result.id == "1234").select("text").collect())
    # get labels
    labels = [l2c[row.label] for row in df.select("label").collect()]

    # plots
    # plot_cluster(sentence_embeddings, labels, clusters, rf_pred=rf_pred)
    # text_similarity(texts, sentence_embeddings, labels, q=10)



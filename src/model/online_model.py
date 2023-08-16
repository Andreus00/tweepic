'''
The purpose of the online model is to be able to run the model in a streaming fashion.

## offline model setup and training

The first thing is to initialize the four pipelines:
    - embeddings pipeline
    - sentence proximity pipeline
    - word and hashtag proximity pipeline
    - edge classifier pipeline

Then, we need to load the intermediate results from the disk, if they exist, otherwise we need to create them.
We can then pass the data to the sentence proximity pipeline and the word and hashtag proximity pipeline, which
will return the sentence, the words and the hashtags distances.
The fourth step is to train the edge classifier pipeline, which will return the edges with the predicted labels for the training data.

# online model

We need a streaming source that delivers tweets in a streaming fashion. We can use the Twitter API for this.
We can then pass the tweets to the embeddings pipeline.
For the sentence proximity pipeline, we have to use the cross join considering only tweets from past and future
based on a window size. This means that we have to keep a buffer of tweets in memory.
After we can easily pass the data to the word and hashtag proximity pipeline and finally update the
graph with the new vertices and edges.
'''

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
from src.model import edge_classifier
from igraph import *
import config


def main():

    # OFFLINE MODEL SETUP AND TRAINING

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

    
    # create the four pipelines
    nlp_pipeline: Pipeline = pipeline_v2.create_embeddings_pipeline(n_buckets=config.n_buckets)
    word_and_hashtag_proximity_pipeline: Pipeline = graph_pipeline.create_word_and_hashtag_proximity_pipeline(n_words=config.n_words, n_hashtags=config.n_hashtags)
    sentence_proximity_pipeline: Pipeline = graph_pipeline.create_sentence_proximity_pipeline(n_neighbors=config.n_neighbors)
    edge_classifier_pipeline: Pipeline = edge_classifier.create_edge_classifier_pipeline()


    ### FIT THE PIPELINES 
    
    # embedding pipeline model
    pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

    # sentence proximity pipeline model
    result = spark.read.parquet(config.intermediate_embeddings_path).sample(False, 0.03, seed=42)
    print("## LOADED EMBEDDINGS INTERMEDIATE ##")
    if config.NUM_FEATURES_CLASSIFICATION == 3:
        result = result.filter(F.size(F.col("hashtags_embeddings")) > 0)
    sentence_proximity_input = result.select("id", "text",  "sentence_embeddings", "word_embeddings", "hashtags_embeddings", "time_bucket")
    sentence_proximity_pipeline_model = sentence_proximity_pipeline.fit(sentence_proximity_input)
    sentence_proximity_pipeline_result = sentence_proximity_pipeline_model.transform(sentence_proximity_input)

    ### word and hashtag proximity pipeline model
    word_and_hashtag_proximity_input = sentence_proximity_pipeline_result.select("id", "text", "word_embeddings", "hashtags_embeddings", "neighbors")
    word_and_hashtag_proximity_pipeline_model = word_and_hashtag_proximity_pipeline.fit(word_and_hashtag_proximity_input)
    word_and_hashtag_proximity_pipeline_result = word_and_hashtag_proximity_pipeline_model.transform(word_and_hashtag_proximity_input)

    edges = word_and_hashtag_proximity_pipeline_result
    vertices = result.select("id", "label", "text")

    ### delete weak edges
    train_edges, test_edges = edges.randomSplit([0.8, 0.2], seed=42)
    edge_classifier_pipeline_model = edge_classifier_pipeline.fit({"edges": train_edges, "vertices": vertices})
    edge_classifier_pipeline_result = edge_classifier_pipeline_model.transform({"edges": test_edges, "vertices": vertices})

    ### GENERATE THE GRAPHc
    proximity_graph = GraphFrame(vertices, edge_classifier_pipeline_result)
    
    proximity_graph.inDegrees.show()
    ig = Graph.TupleList(proximity_graph.edges.collect(), directed=False)
    
    color_labels = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "lime", "olive", "maroon", "navy", "teal", "silver", "gold", "indigo", "violet", "beige", "turquoise", "salmon", "plum", "khaki", "orchid", "tan", "lavender", "coral"]*2

    classes = config.classes
    classes.sort()

    label_to_color = {label: color_labels[i] for i, label in enumerate(classes)}

    vertex_colors = [label_to_color[result.select("label").filter(F.col("id") == int(vertex['name'])).collect()[0][0]] for vertex in ig.vs]


    plot(ig, vertex_color=vertex_colors).save("colored_graph.png")

    connected_components: VertexClustering= ig.connected_components()
    plot(connected_components).save("connected_components.png")


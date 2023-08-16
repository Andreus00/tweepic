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

    # EMBEDDINGS SAVE
    SAVE_EMBEDDINGS_INTERMEDIATE = False
    # EMBEDDINGS LOAD
    LOAD_EMBEDDINGS_INTERMEDIATE = True

    # PROXIMITY SAVE
    SAVE_PROXIMITY_INTERMEDIATE = True
    # PROXIMITY LOAD
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
                                                           "spark.sql.shuffle.partitions": "200",})
    
    from graphframes import GraphFrame


    # df = spark.read.parquet("test_save")
    # df.show()
    # exit()

    # g = GraphFrame(v, e)
    # load data
    if LOAD_EMBEDDINGS_INTERMEDIATE:
        result = spark.read.parquet(config.intermediate_embeddings_path).sample(False, 0.05, seed=42)
        # sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect()).reshape(-1, 768)
        # labels = [l2c[row.label] for row in result.select("label").collect()]
        # plot_cluster(sentence_embeddings, labels)
        if config.NUM_FEATURES_CLASSIFICATION == 3:
            result = result.filter(F.size(F.col("hashtags_embeddings")) > 0)
        print("## LOADED EMBEDDINGS INTERMEDIATE ##")
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

        if SAVE_EMBEDDINGS_INTERMEDIATE:
            result.write.mode("overwrite").parquet(config.intermediate_embeddings_path)
            print("## SAVED ##")
        

    ### SENTENCE PROXIMITY
    if LOAD_PROXIMITY_INTERMEDIATE:
        word_and_hashtag_proximity_pipeline_result = spark.read.parquet(config.intermediate_word_and_hashtags_proximity_path)
        print("## LOADED WORD AND HASHTAGS ##")
    else:
        sentence_proximity_pipeline = graph_pipeline.create_sentence_proximity_pipeline(n_neighbors=config.n_neighbors)
        sentence_proximity_input = result.select("id", "text",  "sentence_embeddings", "word_embeddings", "hashtags_embeddings", "time_bucket")
        sentence_proximity_input.cache()
        sentence_proximity_pipeline_model = sentence_proximity_pipeline.fit(sentence_proximity_input)
        sentence_proximity_pipeline_result = sentence_proximity_pipeline_model.transform(sentence_proximity_input)
    

        ### WORD AND HASHTAGS PROXIMITY
        word_and_hashtag_proximity_pipeline = graph_pipeline.create_word_and_hashtag_proximity_pipeline(n_words=config.n_words, n_hashtags=config.n_hashtags)
        word_and_hashtag_proximity_input = sentence_proximity_pipeline_result.select("id", "text", "word_embeddings", "hashtags_embeddings", "neighbors")
        word_and_hashtag_proximity_pipeline_model = word_and_hashtag_proximity_pipeline.fit(word_and_hashtag_proximity_input)
        word_and_hashtag_proximity_pipeline_result = word_and_hashtag_proximity_pipeline_model.transform(word_and_hashtag_proximity_input)
        if SAVE_PROXIMITY_INTERMEDIATE:
            word_and_hashtag_proximity_pipeline_result.write.parquet(config.intermediate_word_and_hashtags_proximity_path, mode="overwrite")
            print("## SAVED WORD AND HASHTAGS ##")
            exit()

    edges = word_and_hashtag_proximity_pipeline_result
    vertices = result.select("id", "label", "text")

    ### DELETE WEAK EDGES
    train_edges, test_edges = edges.randomSplit([0.8, 0.2], seed=42)
    edge_classifier_pipeline: Pipeline = edge_classifier.create_edge_classifier_pipeline()
    edge_classifier_pipeline_model = edge_classifier_pipeline.fit({"edges": train_edges, "vertices": vertices})
    edge_classifier_pipeline_result = edge_classifier_pipeline_model.transform({"edges": test_edges, "vertices": vertices})

    # TP = edge_classifier_pipeline_result.select("label", "prediction").filter((F.col("label") == 1) & (1 == F.col("prediction"))).count()
    # TN = edge_classifier_pipeline_result.select("label", "prediction").filter((F.col("label") == 0) & (0 == F.col("prediction"))).count()
    # FN = edge_classifier_pipeline_result.select("label", "prediction").filter((F.col("label") == 1) & (0 == F.col("prediction"))).count()
    # FP = edge_classifier_pipeline_result.select("label", "prediction").filter((F.col("label") == 0) & (1 == F.col("prediction"))).count()    # si ma vedi quanti hanno il label 0

    P0 = edge_classifier_pipeline_result.select("label").filter((F.col("label") == 0)).count()    # si ma vedi quanti hanno il label 0
    edge_classifier_pipeline_result = edge_classifier_pipeline_result

    # edge_classifier_pipeline_result = edge_classifier_pipeline_result.union(
    #                                         edge_classifier_pipeline_result.select(
    #                                                                     F.col("dst").alias("src"), 
    #                                                                     F.col("src").alias("dst"), 
    #                                                                     "relationship"))

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

    print("## GRAPH DONE ##")
    
    # print stuff
    # graph_result.show(truncate=False)
    sentence_embeddings = np.asarray(result.select("sentence_embeddings").collect())
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
    plot_cluster(sentence_embeddings, labels, None, rf_pred=None)
    # text_similarity(texts, sentence_embeddings, labels, q=10)


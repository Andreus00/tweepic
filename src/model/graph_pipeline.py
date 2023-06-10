from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import BucketedRandomProjectionLSH
from src.model.cluster import *
from pyspark.ml.feature import Bucketizer
import numpy as np


def create_graph_pipeline():
    l = [
    GetTopNNeighborsTest(n_neigh=3),
    CalculateWordDistance(),
    ]
    return Pipeline(stages=l)


def create_sentence_proximity_pipeline():
    # l = [
    # GetTopNNeighborsTest(n_neigh=20),
    # CalculateWordDistance(),
    # ]
    # return Pipeline(stages=l)s
    cross_join = CrossJoin()
    calculate_distance = CalculateDistance()
    aggregate_neighbors = AggregateNeighbors()
    reorder_neighbors = ReorderNeighbors()

    return Pipeline(stages=[cross_join, calculate_distance, aggregate_neighbors, reorder_neighbors])

def create_word_proximity_pipeline(inputCol, outputCol):

    explode_column = ExplodeColumn()
    
    


    return Pipeline(stages=[])



class CrossJoin(Transformer):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CROSS JOIN ##")

        df2 = df.alias("df2").selectExpr("id as id_2", "sentence_embeddings as sentence_embeddings_2", "word_embeddings as word_embeddings_2", "hashtags_embeddings as hashtags_embeddings_2", "time_bucket")
        df = df.join(df2, (df["time_bucket"] == df2["time_bucket"]) & (df["id"] != df2["id_2"]), how="inner")
        
        return df.select("id", "sentence_embeddings", "word_embeddings", "hashtags_embeddings", "id_2", "sentence_embeddings_2", "word_embeddings_2", "hashtags_embeddings_2")

class CalculateDistance(Transformer):

    def __init__(self, outputCol="euclidean_distance"):
        super().__init__()
        self.outputCol = outputCol
        self.udf_func = udf(self._udf_func, FloatType())

    def _udf_func(self, emb1, emb2):
        return float(distance.euclidean(emb1, emb2))

    def _transform(self, df: DataFrame):
        print("## CALCULATING DISTANCE ##")
        return df.withColumn(self.outputCol, self.udf_func(F.col("sentence_embeddings"), F.col("sentence_embeddings_2")))
        return df
    
class AggregateNeighbors(Transformer):

    def __init__(self, inputcols = ["euclidean_distance", "id_2", "word_embeddings_2", "hashtags_embeddings_2"], outputCol = "neighbors"):
        self.inputCols = inputcols
        self.outputCol = outputCol

    def _transform(self, df: DataFrame):
        print("## AGGREGATING NEIGHBORS ##")
        df = df.groupBy("id", "word_embeddings", "hashtags_embeddings").agg(F.collect_list(F.struct(*self.inputCols)).alias(self.outputCol))
        return df
# check if there are duplicates in the neighbors
# df.select("id", F.size("neighbors").alias("size")).groupBy("size").count().show()

class ReorderNeighbors(Transformer):

    def __init__(self, n_neighbors=3, outputCol = "neighbors"):
        self.outputCol = outputCol
        self.n_neighbors = n_neighbors
        self.udf_func = udf(self._udf_func, ArrayType(StructType(
                                                [
                                                    StructField("euclidean_distance", FloatType()),
                                                    StructField("id_2", StringType()),
                                                    StructField("word_embeddings_2", ArrayType(ArrayType(FloatType()))),
                                                    StructField("hashtags_embeddings_2", ArrayType(ArrayType(FloatType())))
                                                ]
                                            )))

    def _udf_func(self, neighbors):
        # best_neighbors = []
        neigh_len = len(neighbors)
        if neigh_len <= self.n_neighbors:
            return neighbors
        scores = np.array([n[0] for n in neighbors])
        # return sorted(neighbors, key=lambda x: x[0])[:self.n_neighbors]< 
        idxs = np.argpartition(scores, self.n_neighbors)[:self.n_neighbors]
        return [neighbors[i] for i in idxs]

    
    def _transform(self, df: DataFrame):
        print("## REORDERING NEIGHBORS ##")
        return df.withColumn(self.outputCol, self.udf_func(F.col("neighbors"))) #  F.sort_array(F.col("neighbors"), asc = False))



################## CALCULATE DISTANCE BETWEEN WORDS/HASHTAGS #####################


class CalculateWordDistance(Transformer):

    def __init__(self, k_words=3):
        super().__init__()
        self.k_words = k_words
        self.udf_func = udf(self._calculate_word_distance, StructType([
                                                                StructField("distance", FloatType(), True), 
                                                                StructField("word_couples", ArrayType(
                                                                                                StructType([
                                                                                                        StructField("word_1", IntegerType(), True),
                                                                                                        StructField("word_2", IntegerType(), True)
                                                                                                ]), True
                                                                                            ))
                                                            ]))

    def _calculate_word_distance(self, word_emb_1, word_emb_2):
        # wordemb1 and wordemb2 are two lists of word embeddings
        # for each word in the first embedding, find the closest word in the second embedding
        # then take the best three and average the distances

        # calculate the distances between the embeddings
        distances = distance.cdist(word_emb_1, word_emb_2, metric="cosine")

        # get the closest neighbors
        closest_words_idxs = np.argsort(distances, axis=None)[:self.k_words]

        words_sent_1, words_sent_2 = np.unravel_index(closest_words_idxs, distances.shape)


        # get the distances of the closest neighbors
        distances = distances[words_sent_1, words_sent_2]

        # return the average of the distances
        return float(np.mean(distances.flatten())), [(int(a),int(b)) for a, b in zip(words_sent_1, words_sent_2)]
    
    
    def _transform(self, df: DataFrame):
        print("## CALCULATING WORD DISTANCES ##")
        # find the word embeddings for the neighbors
        id_and_word_embeddings = df.select(F.col("id").alias("n_id"), F.col("word_embeddings").alias("word_embeddings_2"))
        # explode the list of neighbors
        df = df.withColumn("neighbors", F.explode(F.col("neighbors")))
        # add the word embeddings to the dataframe
        df = df.join(id_and_word_embeddings, df.neighbors == id_and_word_embeddings.n_id, "left").drop("n_id")
        # calculate the distance between the word embeddings 
        return df.withColumn("word_distance", self.udf_func(F.col("word_embeddings"), F.col("word_embeddings_2"))).drop("sentence_embeddings", "word_embeddings", "word_embeddings_2")

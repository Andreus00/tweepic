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


def create_sentence_proximity_pipeline():
    cross_join = CrossJoin()
    calculate_distance = CalculateDistance()
    aggregate_neighbors = AggregateNeighbors()
    reorder_neighbors = ReorderNeighbors(n_neighbors=20)

    return Pipeline(stages=[cross_join, calculate_distance, aggregate_neighbors, reorder_neighbors])

def create_word_and_hashtag_proximity_pipeline():

    explode_column = ExplodeColumn(inpCol="neighbors")
    closest_words = CalculateArrayDistance(inpCols=["word_embeddings", "neighbors.word_embeddings_2"], outCol="closest_words")
    closest_hashtags = CalculateArrayDistance(inpCols=["hashtags_embeddings", "neighbors.hashtags_embeddings_2"], outCol="closest_hashtags", outFields=["distance", "hashtag_couples", "hashtag_1", "hashtag_2"])
    create_edge = CreateEdge()

    return Pipeline(stages=[explode_column, closest_words, closest_hashtags, create_edge])



class CrossJoin(Transformer):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CROSS JOIN ##")

        df2 = df.alias("df2").selectExpr("id as id_2", "text as text_2", "sentence_embeddings as sentence_embeddings_2", "word_embeddings as word_embeddings_2", "hashtags_embeddings as hashtags_embeddings_2", "time_bucket")
        df = df.join(df2, (df["time_bucket"] == df2["time_bucket"]) & (df["id"] != df2["id_2"]), how="inner")
        
        return df.select("id", "text", "sentence_embeddings", "word_embeddings", "hashtags_embeddings", "id_2", "text_2", "sentence_embeddings_2", "word_embeddings_2", "hashtags_embeddings_2")

class CalculateDistance(Transformer):

    def __init__(self, outputCol="cosine_distance"):
        super().__init__()
        self.outputCol = outputCol
        self.udf_func = udf(self._udf_func, FloatType())

    def _udf_func(self, emb1, emb2):
        return float(distance.cosine(emb1, emb2))

    def _transform(self, df: DataFrame):
        print("## CALCULATING DISTANCE ##")
        return df.withColumn(self.outputCol, self.udf_func(F.col("sentence_embeddings"), F.col("sentence_embeddings_2")))
        return df
    
class AggregateNeighbors(Transformer):

    def __init__(self, inputcols = ["cosine_distance", "id_2", "text_2", "word_embeddings_2", "hashtags_embeddings_2"], outputCol = "neighbors"):
        self.inputCols = inputcols
        self.outputCol = outputCol

    def _transform(self, df: DataFrame):
        print("## AGGREGATING NEIGHBORS ##")
        df = df.groupBy("id", "word_embeddings", "hashtags_embeddings", "text").agg(F.collect_list(F.struct(*self.inputCols)).alias(self.outputCol))
        return df
# check if there are duplicates in the neighbors
# df.select("id", F.size("neighbors").alias("size")).groupBy("size").count().show()

class ReorderNeighbors(Transformer):

    def __init__(self, n_neighbors=3, outputCol = "neighbors"):
        self.outputCol = outputCol
        self.n_neighbors = n_neighbors
        self.udf_func = udf(self._udf_func, ArrayType(StructType(
                                                [
                                                    StructField("cosine_distance", FloatType()),
                                                    StructField("id_2", IntegerType()),
                                                    StructField("text_2", StringType()),
                                                    StructField("word_embeddings_2", ArrayType(ArrayType(FloatType()))),
                                                    StructField("hashtags_embeddings_2", ArrayType(ArrayType(FloatType())))
                                                ]
                                            )))

    def _udf_func(self, neighbors):
        # best_neighbors = []
        # return sorted(neighbors, key=lambda x: x[0])[:self.n_neighbors]
        neigh_len = len(neighbors)
        if neigh_len <= self.n_neighbors:
            return neighbors
        scores = np.array([n[0] for n in neighbors])
        idxs = np.argpartition(scores, self.n_neighbors)[:self.n_neighbors]
        return [neighbors[i] for i in idxs]

    
    def _transform(self, df: DataFrame):
        print("## REORDERING NEIGHBORS ##")
        return df.withColumn(self.outputCol, self.udf_func(F.col("neighbors"))) #  F.sort_array(F.col("neighbors"), asc = False))



################## CALCULATE DISTANCE BETWEEN WORDS/HASHTAGS #####################


class ExplodeColumn(Transformer):

    def __init__(self, inpCol="neighbors") -> None:
        super().__init__()
        self.inputCol = inpCol

    def _transform(self, df: DataFrame):
        print("## EXPLODING COLUMN ##")
        return df.withColumn(self.inputCol, F.explode(self.inputCol))


class CalculateArrayDistance(Transformer):

    def __init__(self,inpCols, outCol, outFields=["distance", "word_couples", "word_1", "word_2"], k_el=3):
        super().__init__()
        self.inputCols = inpCols
        self.outFields = outFields
        self.outCol = outCol
        self.k_el = k_el
        self.udf_func = udf(self._calculate_array_distance, StructType([
                                                                StructField(self.outFields[0], FloatType(), True), 
                                                                StructField(self.outFields[1], ArrayType(
                                                                                                StructType([
                                                                                                        StructField(self.outFields[2], IntegerType(), True),
                                                                                                        StructField(self.outFields[3], IntegerType(), True)
                                                                                                ]), True
                                                                                            ), True)
                                                            ]))

    def _calculate_array_distance(self, array_emb_1, array_el_2):
        # wordemb1 and wordemb2 are two lists of word embeddings
        # for each word in the first embedding, find the closest word in the second embedding
        # then take the best three and average the distances
        len_1, len_2 = len(array_emb_1), len(array_el_2)
        if len_1 == 0 or len_2 == 0:
            return None
        # calculate the distances between the embeddings
        distances = distance.cdist(array_emb_1, array_el_2, metric="cosine")
        k_el  = min(len_1 * len_2, self.k_el)

        # get the closest neighbors
        closest_el_idxs = np.argsort(distances, axis=None)[:k_el]
        # closest_el_idxs = np.argpartition(distances.flatten(), k_el, axis=None)[:k_el]

        array_el_1, array_el_2 = np.unravel_index(closest_el_idxs, distances.shape)


        # get the distances of the closest neighbors
        distances = distances[array_el_1, array_el_2]

        # return the average of the distances
        return float(np.mean(distances.flatten())), [(int(a),int(b)) for a, b in zip(array_el_1, array_el_2)]
    
    
    def _transform(self, df: DataFrame):
        print("## CALCULATING DISTANCES ##")
        # calculate the distance between the word embeddings 
        return df.withColumn(self.outCol, self.udf_func(F.col(self.inputCols[0]), F.col(self.inputCols[1]))).drop(*self.inputCols)



class CreateEdge(Transformer):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CREATING EDGES ##")
        return df.select(F.col("id").alias("src"), F.col("neighbors.id_2").alias("dst"), F.array(F.col("neighbors.cosine_distance").alias("sentence_distance"), F.col("closest_words.distance").alias("word_distance"), F.col("closest_hashtags.distance").alias("hashtag_distance")).alias("relationship"))
         



################## CREATE THE GRAPH #####################
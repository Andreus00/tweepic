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


def create_sentence_proximity_pipeline(n_neighbors=5):
    cross_join = CrossJoin()
    calculate_distance = CalculateDistance()
    aggregate_neighbors = AggregateNeighbors()
    reorder_neighbors = ReorderNeighbors(n_neighbors=n_neighbors)

    return Pipeline(stages=[cross_join, calculate_distance, aggregate_neighbors, reorder_neighbors])

def create_online_sentence_proximity_pipeline(n_neighbors=5):
    cross_join = OnlineCrossJoin()
    calculate_distance = CalculateDistance()
    aggregate_neighbors = AggregateNeighbors()
    reorder_neighbors = ReorderNeighbors(n_neighbors=n_neighbors)

    return Pipeline(stages=[cross_join, calculate_distance, aggregate_neighbors, reorder_neighbors])

def create_word_and_hashtag_proximity_pipeline(n_words=3, n_hashtags=3):

    explode_column = ExplodeColumn(inpCol="neighbors")
    closest_words = CalculateArrayDistance(inpCols=["word_embeddings", "neighbors.word_embeddings_2"], outCol="closest_words", k_el=n_words)
    closest_hashtags = CalculateArrayDistance(inpCols=["hashtags_embeddings", "neighbors.hashtags_embeddings_2"], outCol="closest_hashtags", outFields=["distance", "hashtag_couples", "hashtag_1", "hashtag_2"], k_el=n_hashtags)
    create_edge = CreateEdge()

    return Pipeline(stages=[explode_column, closest_words, closest_hashtags, create_edge])


def create_online_word_and_hashtag_proximity_pipeline(n_words=3, n_hashtags=3):

    # explode_column = ExplodeColumn(inpCol="neighbors")
    closest_words = CalculateArrayDistance(inpCols=["word_embeddings", "word_embeddings_2"], outCol="closest_words", k_el=n_words)
    closest_hashtags = CalculateArrayDistance(inpCols=["hashtags_embeddings", "hashtags_embeddings_2"], outCol="closest_hashtags", outFields=["distance", "hashtag_couples", "hashtag_1", "hashtag_2"], k_el=n_hashtags)
    create_edge = CreateEdge(inpCols = ["id", "id_2", "cosine_distance", "closest_words.distance", "closest_hashtags.distance"])

    return Pipeline(stages=[closest_words, closest_hashtags, create_edge])

# def create_online_proximity_pipeline(n_neighbors=5, n_words=3, n_hashtags=3):
#     cross_join = CrossJoin()
#     calculate_distance = CalculateDistance()
#     aggregate_neighbors = AggregateNeighbors()
#     reorder_neighbors = ReorderNeighbors(n_neighbors=n_neighbors)
#     closest_words = CalculateArrayDistance(inpCols=["word_embeddings", "word_embeddings_2"], outCol="closest_words", k_el=n_words)
#     closest_hashtags = CalculateArrayDistance(inpCols=["hashtags_embeddings", "hashtags_embeddings_2"], outCol="closest_hashtags", outFields=["distance", "hashtag_couples", "hashtag_1", "hashtag_2"], k_el=n_hashtags)
#     create_edge = CreateEdge(inpCols = ["id", "id_2", "cosine_distance", "closest_words.distance", "closest_hashtags.distance"])

#     return Pipeline(stages=[cross_join, calculate_distance, aggregate_neighbors, reorder_neighbors, closest_words, closest_hashtags, create_edge])


class CrossJoin(Transformer):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CROSS JOIN ##")

        buckets = df.select("time_bucket").distinct().collect()
        buckets.sort(key=lambda x: x["time_bucket"])

        first_bucket = buckets[0]["time_bucket"]

        df_bucket = df.filter((df["time_bucket"] == first_bucket) | (df["time_bucket"] == first_bucket + 1)).alias("df_bucket")
        df2 =  df_bucket.filter(df_bucket["time_bucket"] == first_bucket).select(F.col("id").alias("id_2"), F.col("sentence_embeddings").alias("sentence_embeddings_2")).alias("df2")
        df_bucket = df_bucket.drop("time_bucket").join(df2, (df_bucket["id"] != df2["id_2"]), how="inner")
        df_final = df_bucket.select("id", "sentence_embeddings", "id_2", "sentence_embeddings_2")


        all_dfs = [df_final]


        for bucket in buckets[1:]:
            df_bucket = df.filter((df["time_bucket"] == bucket["time_bucket"]) | (df["time_bucket"] == bucket["time_bucket"] - 1)).alias("df_bucket")
            df2 = df_bucket.filter(df_bucket["time_bucket"] == bucket["time_bucket"]).select(F.col("id").alias("id_2"), F.col("sentence_embeddings").alias("sentence_embeddings_2")).alias("df2")
            df_bucket = df_bucket.drop("time_bucket").join(df2, (df_bucket["id"] != df2["id_2"]), how="inner")
            df_bucket = df_bucket.select("id", "sentence_embeddings", "id_2", "sentence_embeddings_2")
            all_dfs.append(df_bucket)
            # print(df_final.rdd.count())
        
        df_final = all_dfs[0]
        for df in all_dfs[1:]:
            df_final = df_final.union(df)

        return df_final.select("id", "sentence_embeddings", "id_2", "sentence_embeddings_2")
    



class OnlineCrossJoin(Transformer):

    '''
    Takes tweets from two buckets and joins them together.
    The input MUST be tweets from two buckets.
    '''

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CROSS JOIN ##")

        buckets = df.select("time_bucket").distinct().collect()
        buckets.sort(key=lambda x: x["time_bucket"])

        first_bucket = buckets[0]["time_bucket"]
        second_bucket = buckets[1]["time_bucket"]

        df2 =  df.filter(F.col("time_bucket") == 1).alias("df2")
        df3 = df.drop("time_bucket").select(F.col("id").alias("id_2"), F.col("sentence_embeddings").alias("sentence_embeddings_2"))
        df = df2.join(df3, (df3["id_2"] != df2["id"]), how="inner")
        df_final = df.select("id", "sentence_embeddings", "id_2", "sentence_embeddings_2")
        
        return df_final
    



class OnlineDistanceCalculator(Transformer):

    def __init__(self, n_neigh) -> None:
        super().__init__()
        self.n_neigh = n_neigh

    def _transform(self, df: DataFrame):
        print("## GETTING EMBEDDINGS ##")
        embeddings = df.select("sentence_embeddings").collect()
        ids = np.asarray(df.select("id").collect())
        word_embeddings = df.select("word_embeddings").collect()
        hashtags_embeddings = df.select("hashtags_embeddings").collect()
        embeddings = np.array([x[0] for x in embeddings]).reshape(-1, 768)
        print(embeddings.shape)
        
        print("## CALCULATING DISTANCES ##")
        distances = distance.pdist(embeddings, metric="cosine")
        distances = distance.squareform(distances)
        best_idxs_pos = distances.argsort(axis=1)[:,1:self.n_neigh  + 1]

        print("## GETTING NEIGHBORS ##")
        
        # get the ids of the closest neighbors
        best_idxs = ids[best_idxs_pos].reshape(-1, self.n_neigh).tolist()
        
        neighbor_distances = [distances[i, best_idxs_pos[i]].tolist() for i in range(len(best_idxs_pos))]
        # create a dataframe with the distances
        
        print("## CREATING THE FINAL DATAFRAME ##")
        
        def get_neighbors(index):
             return best_idxs[index-1]
        
        def get_distances(index):
            return neighbor_distances[index-1]
        
        def get_word_embeddings(index):
            return word_embeddings[index-1]
        
        def get_hashtags_embeddings(index):
            return hashtags_embeddings[index-1]
        
        get_neighbors_udf = udf(get_neighbors, ArrayType(StringType()))
        get_distances_udf = udf(get_distances, ArrayType(FloatType()))
        get_word_embeddings_udf = udf(get_word_embeddings, ArrayType(ArrayType(FloatType())))
        get_hashtags_embeddings_udf = udf(get_hashtags_embeddings, ArrayType(ArrayType(FloatType())))

        w = Window.partitionBy(F.lit(1)).orderBy("id")

        # Add an index column
        df = df.withColumn("index", F.row_number().over(w))

        # Now you can use these UDFs to add your new columns
        df = df.withColumn("cosine_distance", get_distances_udf(F.col("index")))
        df = df.withColumn("id_2", get_neighbors_udf(F.col("index")))
        df = df.withColumn("word_embeddings_2", get_word_embeddings_udf(F.col("index")))
        df = df.withColumn("hashtags_embeddings_2", get_hashtags_embeddings_udf(F.col("index")))
        df = df.drop("index", "time_bucket")
        df = df.withColumn("neighbors", F.arrays_zip("id_2", "cosine_distance", "word_embeddings_2", "hashtags_embeddings_2"))
        return df






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


class AggregateNeighbors(Transformer):

    def __init__(self, inputcols = ["cosine_distance", "id_2"], outputCol = "neighbors"):
        self.inputCols = inputcols
        self.outputCol = outputCol

    def _transform(self, df: DataFrame):
        print("## AGGREGATING NEIGHBORS ##")
        df = df.groupBy("id").agg(F.collect_list(F.struct(*self.inputCols)).alias(self.outputCol))
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
                                                    StructField("id_2", StringType()) # ,
                                                    # StructField("word_embeddings_2", ArrayType(ArrayType(FloatType()))),
                                                    # StructField("hashtags_embeddings_2", ArrayType(ArrayType(FloatType())))
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

    def __init__(self, inpCols = ["id", "neighbors.id_2", "neighbors.cosine_distance", "closest_words.distance", "closest_hashtags.distance"]) -> None:
        super().__init__()
        self.inpCols = inpCols

    def _transform(self, df: DataFrame):
        print("## CREATING EDGES ##")
        return df.select(F.col(self.inpCols[0]).alias("src"), F.col(self.inpCols[1]).alias("dst"), F.array(F.col(self.inpCols[2]).alias("sentence_distance"), F.col(self.inpCols[3]).alias("word_distance"), F.col(self.inpCols[4]).alias("hashtag_distance")).alias("relationship"))
         



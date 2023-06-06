from pyspark.ml import Transformer, Estimator
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StructField, StructType
from scipy.spatial import distance
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

THRESHOLD = 0.1

class TweetClusterPreprocessing(Transformer):

    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.udf_func = udf(self._udf, VectorUDT())

    # def _udf(self, *cols):
    #     # Concatenate the vectors
    #     return Vectors.dense(*(c[0] for c in cols))

    # def _transform(self, df):
    #     return df.withColumn(self.outputCol, self.udf_func(*self.inputCols))
    def _udf(self, emb):
        return Vectors.dense(*emb[0])

    def _transform(self, df):
        print(df[self.inputCol])
        return df.withColumn(self.outputCol, self.udf_func(self.inputCol))


class LabelsToIndex(Transformer):

    def __init__(self, l2c) -> None:
        super().__init__()
        self.l2c = l2c
        self.udf_func = udf(lambda x: l2c[x], IntegerType())
    
    def _transform(self, df):
        return df.withColumn("label_idx", self.udf_func(df.label))

class CrossJoin(Transformer):

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, df: DataFrame):
        print("## CALCUATING CARTESIAN PRODUCT ##")
        # print("# entries before:", df.count())
        # make the cartesian product of the dataframe with itself
        return df.crossJoin(df.select(F.col("id").alias("id_2"),
                                    F.col("sentence_embeddings").alias("sentence_embeddings_2"))) \
                .filter("id != id_2")
        # print("# entries after:", joined.count())


class CalculateDistance(Transformer):

    def __init__(self):
        super().__init__()

    def udf_func(self, emb1, emb2):
        return float(distance.cosine(emb1, emb2))

    def _transform(self, df: DataFrame):
        print("## CALCULATING DISTANCE ##")
        return df.withColumn("cosine_similarity", F.udf(self.udf_func, FloatType())(F.col("sentence_embeddings"), F.col("sentence_embeddings_2")))

        # df_cos_sim = df.withColumn("dot_prod", F.lit(sum([F.col("sentence_embeddings").getItem(i) * F.col("sentence_embeddings_2").getItem(i) for i in range(768)])))
        # df_cos_sim = df_cos_sim.withColumn("norm_1", F.lit(F.sqrt(sum([F.col("sentence_embeddings").getItem(i) * F.col("sentence_embeddings").getItem(i) for i in range(768)]))))
        # df_cos_sim = df_cos_sim.withColumn("norm_2", F.lit(F.sqrt(sum([F.col("sentence_embeddings_2").getItem(i) * F.col("sentence_embeddings_2").getItem(i) for i in range(768)]))))
        # df_cos_sim = df_cos_sim.withColumn("cosine_similarity", F.lit(F.col("dot_prod") / (F.col("norm_1") * F.col("norm_2"))))
        # df_cos_sim = df_cos_sim.drop("sentence_embeddings", "sentence_embeddings_2", "dot_prod", "norm_1", "norm_2")
        # return df_cos_sim
    

class FilterDistance(Transformer):

    def __init__(self, threshold = THRESHOLD):
        super().__init__()
        self.threshold = threshold
    
    def _transform(self, df: DataFrame):
        print("## FILTERING ##")
        return df.filter(F.col("cosine_similarity") >= self.threshold).repartition(1000)


class AggregateNeighbors(Transformer):

    def __init__(self, outputCol = "neighbors"):
        self.outputCol = outputCol

    def _transform(self, df: DataFrame):
        print("## AGGREGATING NEIGHBORS ##")
        
        return df.groupBy("id").agg(F.collect_list(F.struct("cosine_similarity", "id_2")).alias(self.outputCol)).repartition(1000)


# class ReorderNeighbors(Transformer):

#     def __init__(self, outputCol = "neighbors"):
#         self.outputCol = outputCol

#     def _transform(self, df: DataFrame):
#         print("## REORDERING NEIGHBORS ##")
#         return df.withColumn(self.outputCol, F.sort_array(F.col("neighbors"), asc = False))

class GetClosestNeighbors(Transformer):

    def __init__(self, n_neighbors=3, outputCol = "closest_neighbors"):
        self.outputCol = outputCol
        self.n_neighbors = n_neighbors

    def _transform(self, df: DataFrame):
        print("## GETTING CLOSEST NEIGHBORS ##")
        return df.withColumn(self.outputCol, F.slice(F.col("neighbors"), 1, self.n_neighbors))
    




class GetTopNNeighbors(Transformer):
    '''
    This class gets the embeddings of each tweet and finds the closest n neighbors.
    It then calculates the weight of each edge based on the distance between the best words of the tweets.
    '''

    def __init__(self, n_neighbors, wordEmbCol, sentEmbCol, outCol) -> None:
        super().__init__()
        self.wordEmbCol = wordEmbCol
        self.sentEmbCol = sentEmbCol
        self.outCol = outCol
        self.n_neighbors = n_neighbors
        self.udf_func = udf(self._get_top_n_words, ArrayType(FloatType()))


    def _get_top_n_words(self, scores):
        return scores[:self.n_neighbors]

    
    def _transform(self, df: DataFrame):
        # for each element, find the n closest neighbors
        # for each neighbor, calculate the weight of the edge based on the k closest words
        # return a dataframe with the edges

        print("## CALCUATING CARTESIAN PRODUCT ##")
        # print("# entries before:", df.count())
        # make the cartesian product of the dataframe with itself
        joined = df.crossJoin(df.select(F.col("id").alias("id_2"), 
                                    F.col("word_embeddings").alias("word_embeddings_2"),
                                    F.col("sentence_embeddings").alias("sentence_embeddings_2"))) \
                .filter("id != id_2")
        # print("# entries after:", joined.count())

        print("## CALCUATING DISTANCES ##")
        # calculate the distances between the embeddings
        distances = joined.select("id", "id_2", self.wordEmbCol, "word_embeddings_2", self.sentEmbCol, "sentence_embeddings_2") \
                    .rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], float(distance.cosine(x[4], x[5])))).filter(lambda x: x[6] < THRESHOLD) \
                        .toDF(["id", "id_2", self.wordEmbCol, "word_embeddings_2", self.sentEmbCol, "sentence_embeddings_2", "distance"])

        # print("# entries after:", distances.count())

        print("## CALCUATING SORTING ##")
        # get the closest neighbors
        closest_neighbors = distances.groupBy("id", self.sentEmbCol) \
                                    .agg(F.collect_list(F.struct("id_2", "word_embeddings_2", "sentence_embeddings_2", "distance")).alias("closest_neighbors")) \
                                    .repartition(1000) \
                                    .rdd.map(lambda x: (x[0], x[1], [y[0] for y in sorted(x[2], key=lambda z: z[3])[:self.n_neighbors]])) \
                                    .toDF(["id", self.sentEmbCol, "closest_neighbors"])
        # print("# entries after:", closest_neighbors.count())
        
        return closest_neighbors
    

class CreateGraph(Transformer):

    def __init__(self, k_words, inpCol, outCol) -> None:
        super().__init__()
        self.k_words = k_words
        self.inpCol = inpCol
        self.outCol = outCol

    def _transform(self, df: DataFrame):
        # I receive a dataframe with the closest neighbors in the format: (id, word_embeddings, sentence_embedding, [id_2, word_embeddings_2, sentence_embeddings_2, distance])
        # I need to output a dataframe that contains the edges in the format: (id, id_2, weight)
        # The weight is given by those wotds that are closest to each other across the two sentences

        # first split the list of neighbors into a list of tuples
        df = df.withColumn("closest_neighbors", F.explode(F.col("closest_neighbors"))).select("id", "word_embeddings", "closest_neighbors.id_2", "closest_neighbors.sentence_embeddings_2", "closest_neighbors.distance")



        pass
    



# def assign_tweet_to_cluster(tweet_embedding, cluster_centers, thresholds):
#     distances = cdist(tweet_embedding[np.newaxis, :], cluster_centers)
#     for i, distance in enumerate(distances[0]):
#         if distance < thresholds[i]:
#             return i  # return the cluster index
#     return -1  # return -1 if the tweet doesn't belong to any existing

# from scipy.spatial.distance import cdist
# # Choose the number of clusters for K-Means. This might require some trial and error.
# n_clusters = 10

# # Perform K-Means clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)

# # The cluster centers can be accessed with kmeans.cluster_centers_
# # These will be used to calculate the distance to new tweets
# cluster_centers = kmeans.cluster_centers_

# # The labels_ attribute contains the cluster index for each tweet
# cluster_labels = kmeans.labels_
# # Calculate the distance of each tweet to its cluster center
# distances = cdist(embeddings, cluster_centers[cluster_labels])

# # Calculate the average distance within each cluster
# average_distances = distances.mean(axis=0)

# # You could use these average distances as your thresholds for adding new tweets to the clusters
# thresholds = average_distances


class FindHashtags(Transformer):

    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.udf_func = udf(lambda x: [y for y in x if y.startswith("#")])

    def _transform(self, df):
        # find the hashtags
        return df.withColumn(self.outputCol, self.udf_func(self.inputCol))
    
    
class FindHashtags(Transformer):

    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.udf_func = udf(lambda x: [y for y in x if y.startswith("#")])
        
    def _transform(self, df):
        # find the hashtags
        return df.withColumn(self.outputCol, self.udf_func(self.inputCol))
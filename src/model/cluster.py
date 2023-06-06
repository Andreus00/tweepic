from pyspark.ml import Transformer, Estimator
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from sklearn.cluster import KMeans
from pyspark.sql.types import IntegerType, FloatType
from scipy.spatial import distance
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


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

class GraphGenerator(Transformer):
    '''
    This class gets the embeddings of each tweet and finds the closest n neighbors.
    It then calculates the weight of each edge based on the distance between the best words of the tweets.
    '''

    def __init__(self, n_neighbors, k_words, wordEmbCol, sentEmbCol, outCol) -> None:
        super().__init__()
        self.wordEmbCol = wordEmbCol
        self.sentEmbCol = sentEmbCol
        self.outCol = outCol
        self.n_neighbors = n_neighbors
        self.k_words = k_words
    
    def _get_edges(self, word_embeddings, closest_neighbors, k_words):
        # get the distance matrix for each neighbor and the tweet
        pass
    
    def _transform(self, df: DataFrame):
        # for each element, find the n closest neighbors
        # for each neighbor, calculate the weight of the edge based on the k closest words
        # return a dataframe with the edges

        # make the cartesian product of the dataframe with itself
        df = df.crossJoin(df.select(F.col("id").alias("id_2"), F.col("sentence_embeddings").alias("sentence_embeddings_2")))

        # calculate the distances between the embeddings
        embeddings = df.select(self.sentEmbCol, "sentence_embeddings_2").rdd.map(lambda x: (x[0], x[1], float(distance.cosine(x[0], x[1])))).toDF([self.sentEmbCol, "sentence_embeddings_2", "distance"])
        # get the closest neighbors
        closest_neighbors = embeddings.groupBy(self.sentEmbCol).agg(F.collect_list(F.struct("sentence_embeddings_2", "distance")).alias("closest_neighbors")).rdd.map(lambda x: (x[0], [y[0] for y in sorted(x[1], key=lambda x: x[1])[:self.n_neighbors]])).toDF([self.sentEmbCol, "closest_neighbors"])

        # get the edges
        
        # create the dataframe
        return closest_neighbors # df.withColumn(self.outCol, edges)
    



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
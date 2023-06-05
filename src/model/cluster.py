from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT


class TweetClusterPreprocessing(Transformer):

    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.udf_func = udf(self._udf, VectorUDT())

    def _udf(self, emb):
        return Vectors.dense(*emb[0].embeddings)

    def _transform(self, df):
        return df.withColumn(self.outputCol, self.udf_func(self.inputCol))


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
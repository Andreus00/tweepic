from pyspark.ml import Transformer, Estimator
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StructField, StructType, StringType, LongType
from scipy.spatial import distance
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import Normalizer
import pandas as pd
import pyspark
from pyspark.sql.window import Window


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
        # df.drop("document", "sentence", "token", "rawPrediction", "probability", "year", "month", "day", "hashtags", "mentions")
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
        print("# entries before:", df.count())
        # make the cartesian product of the dataframe with itself
        # df =  df.join(df.select(F.col("id").alias("id_2"),
        #                             F.col("sentence_embeddings").alias("sentence_embeddings_2")), "id < id_2", how="inner")
        df = df.join(
            df.selectExpr("id as id_2", "sentence_embeddings as sentence_embeddings_2"),
            F.expr("id < id_2"),
            how="inner"
        )
        return df


class CalculateDistance(Transformer):

    def __init__(self):
        super().__init__()
        self.udf_func = udf(self._udf_func, FloatType())

    def _udf_func(self, emb1, emb2):
        return float(distance.euclidean(emb1, emb2))

    def _transform(self, df: DataFrame):
        print("## CALCULATING DISTANCE ##")
        df =  df.withColumn("euclidean_distance", self.udf_func(F.col("sentence_embeddings"), F.col("sentence_embeddings_2")))
        return df
    

class FilterDistance(Transformer):

    def __init__(self, threshold = THRESHOLD):
        super().__init__()
        self.threshold = threshold
    
    def _transform(self, df: DataFrame):
        print("## FILTERING ##")
        df = df.filter(F.col("euclidean_distance") >= self.threshold)
        return df


class AggregateNeighbors(Transformer):

    def __init__(self, outputCol = "neighbors"):
        self.outputCol = outputCol
        self.udf_func = udf(self.comparator, IntegerType())

    def comparator(self, a, b):
        c = a - b
        return 1 if c > 0 else -1 if c < 0 else 0


    # def func(self, a, b):
    #     return a - b


    def _transform(self, df: DataFrame):
        print("## AGGREGATING NEIGHBORS ##")
        df = df.groupBy("id").agg(F.array_sort(F.collect_list(F.struct("euclidean_distance", "id_2")), self.udf_func).alias(self.outputCol))
        return df


# class ReorderNeighbors(Transformer):

#     def __init__(self, outputCol = "neighbors"):
#         self.outputCol = outputCol

#     def _transform(self, df: DataFrame):
#         print("## REORDERING NEIGHBORS ##")
#         return df.withColumn(self.outputCol, F.sort_array(F.col("neighbors"), asc = False))

class GetClosestNeighbors(Transformer):

    def __init__(self, n_neighbors=3, outputCol = "neighbors"):
        self.outputCol = outputCol
        self.n_neighbors = n_neighbors

    def _transform(self, df: DataFrame):
        print("## GETTING CLOSEST NEIGHBORS ##")
        df =  df.withColumn(self.outputCol, F.slice(F.col("neighbors"), 1, self.n_neighbors))
        return df
    




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
    



    

class GetTopNNeighborsTest(Transformer):

    '''
    version that uses scipy.spatial.distance.cdist
    '''

    def __init__(self, n_neigh=3) -> None:
        super().__init__()
        self.n_neigh = n_neigh

    def _transform(self, df: DataFrame):
        print("## GETTING EMBEDDINGS ##")
        embeddings = df.select("sentence_embeddings").collect()
        ids = np.asarray(df.select("id").collect())
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
        
        get_neighbors_udf = udf(get_neighbors, ArrayType(StringType()))
        get_distances_udf = udf(get_distances, ArrayType(FloatType()))

        w = Window.partitionBy(F.lit(1)).orderBy("id")

        # Add an index column
        df = df.withColumn("index", F.row_number().over(w))

        # Now you can use these UDFs to add your new columns
        df = df.withColumn("neighbors", get_neighbors_udf(F.col("index")))
        df = df.withColumn("distances", get_distances_udf(F.col("index")))
        df = df.drop("index")

        return df



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
        return df.withColumn("word_distance", self.udf_func(F.col("word_embeddings"), F.col("word_embeddings_2")))





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
    

class SplitInformations(Transformer):

    def __init__(self, inputCol, outputCols):
        self.inputCol = inputCol
        self.outputCols = outputCols
        
    def _transform(self, df):
        # find the hashtags
        informations = df.select("Informations").collect()
        hashtags = []
        mentions = []

        for info in informations:
            list_of_hashtags = []
            list_of_mentions = []
            for item in info[0]:
                hashtag = item.result
                # id = item.metadata.identifier
                # if id == "hashtag":
                list_of_hashtags.append(hashtag[1:].lower())
                # elif id == "mention":
                #     list_of_mentions.append(hashtag[1:].lower())
            hashtags.append(list_of_hashtags)
            mentions.append(list_of_mentions)
        print(hashtags)
        # hashtags = [y for x in informations for y in x[0] if y.startswith("#")]
        # mentions = [y for x in informations for y in x[0] if y.startswith("@")]

        return df.withColumn(self.outputCols, F.array(hashtags)) \
    


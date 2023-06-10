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
from src.model.cluster import *  # TweetClusterPreprocessing, GetTopNNeighbors, LabelsToIndex, FindHashtags, CreateGraph
from pyspark.ml.feature import Bucketizer



def create_embedding_pipeline(label_count, l2c):
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())


    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
        .setInputCols(["document"])\
        .setOutputCol("language")
    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")
    tokenizer = Tokenizer() \
        .setInputCols("sentence") \
        .setOutputCol("token")
    embeddingsWord = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
        .setInputCols("document", "token") \
        .setOutputCol("word_embeddings") \
        .setCaseSensitive(True)
    hashtagsEmbeddings = HashtagEmbeddings(
        inputCols=["token", "word_embeddings"],
        outputCol="hashtags_embeddings",
    )
    embeddingsSentence = SentenceEmbeddings() \
        .setInputCols(["document", "word_embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")
    wordEmbeddingsFinisher = WordEmbeddingsFinisher(
        inputCol="word_embeddings",
        outputCol="word_embeddings",
    ) 
    embeddingsFinisher = EmbeddingsFinisher() \
        .setInputCols("sentence_embeddings") \
        .setOutputCols("sentence_embeddings") \
        .setCleanAnnotations(False)    
    label2idx = LabelsToIndex(l2c)
    randomForest =  RandomForestClassifier() \
        .setLabelCol("label_idx") \
        .setFeaturesCol("sentence_embeddings") \
        .setNumTrees(10)
    embeddingPreprocessing = TweetEmbeddingPreprocessing(
        inputCol="sentence_embeddings", 
        outputCol="sentence_embeddings")
    # BisectingKMeans()
    kmeans  = KMeans() \
        .setK(label_count) \
        .setSeed(1) \
        .setFeaturesCol("sentence_embeddings") \
        .setPredictionCol("cluster") \
        .setDistanceMeasure("cosine")
    

    return Pipeline(stages=[document_assembler, languageDetector, sentence_detector, tokenizer, embeddingsWord, embeddingsSentence, wordEmbeddingsFinisher, embeddingsFinisher, hashtagsEmbeddings, embeddingPreprocessing, label2idx, randomForest])


def create_graph_pipeline():
    l = [
    GetTopNNeighborsTest(n_neigh=3),
    CalculateWordDistance(),
    ]
    return Pipeline(stages=l)


def create_word_distance_pipeline():
    # l = [
    # GetTopNNeighborsTest(n_neigh=20),
    # CalculateWordDistance(),
    # ]
    # return Pipeline(stages=l)
    import datetime

    dateToFeatures = DateToFeatures(["year", "month", "day"], "dateFeature")
    dateBucketizer = DateBucketizer("dateFeature", "bucket", 12, datetime.date(2023, 6, 8).toordinal())
    crossJoin = CrossJoin()
    calculateDistance = CalculateDistance()
    aggregateNeighbors = AggregateNeighbors()
    reorderNeighbors = ReorderNeighbors()

    return Pipeline(stages=[dateToFeatures, dateBucketizer, crossJoin, calculateDistance, aggregateNeighbors, reorderNeighbors])


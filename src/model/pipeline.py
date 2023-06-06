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


def create_pipeline(label_count, l2c):
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())


    document_assembler = DocumentAssembler() \
        .setInputCol("text_with_info") \
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
    embeddingsSentence = SentenceEmbeddings() \
        .setInputCols(["document", "word_embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")
    embeddingsFinisher = EmbeddingsFinisher() \
        .setInputCols("sentence_embeddings") \
        .setOutputCols("sentence_embeddings") \
        .setCleanAnnotations(False)
    graphPreprocessing = TweetClusterPreprocessing(
        inputCol="sentence_embeddings", 
        outputCol="sentence_embeddings")
    # BisectingKMeans()
    kmeans  = KMeans() \
        .setK(label_count) \
        .setSeed(1) \
        .setFeaturesCol("sentence_embeddings") \
        .setPredictionCol("cluster") \
        .setDistanceMeasure("cosine")
    label2idx = LabelsToIndex(l2c)
    randomForest =  RandomForestClassifier() \
        .setLabelCol("label_idx") \
        .setFeaturesCol("sentence_embeddings") \
        .setNumTrees(10)
    
    # class ApproxEuclideanDistance(Estimator):
        
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.model = BucketedRandomProjectionLSH(
    #             inputCol="sentence_embeddings",
    #             outputCol="hashes",
    #             bucketLength=0.1,
    #             numHashTables=3
    #         )
            
    #     def _fit(self, dataset):
    #         self.model = self.model.fit(dataset)
    #         return self

    #     def transform(self, dataset):
    #         df = self.model.transform(dataset)
    #         return self.model.approxSimilarityJoin(df, df, 0.2, distCol="distance")
        
    # graph_processing = GetTopNNeighbors(5, "word_embeddings", "sentence_embeddings", "graph_edges")



    # createGraph = CreateGraph(5, "word_embeddings", "sentence_embeddings")

    l = [
        CrossJoin(),
        CalculateDistance(),
        FilterDistance(),
        AggregateNeighbors(),
        GetClosestNeighbors(n_neighbors=3)
    ]


    
    # approxEuclideanDistance = ApproxEuclideanDistance()
    

    return Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddingsWord, embeddingsSentence, embeddingsFinisher, graphPreprocessing, label2idx, randomForest]), Pipeline(stages=l)
    









    



def create_pipeline2(label_count, l2c):

    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    
    # hasthag_assembler = DocumentAssembler() \
    #     .setInputCol("hashtags") \
    #     .setOutputCol("hashtag_document")
    
    # date_assembler = DocumentAssembler() \
    #     .setInputCol("date") \
    #     .setOutputCol("date")

    # hashtag_sentence = SentenceDetector() \
    #     .setInputCols(["hashtag_document"]) \
    #     .setOutputCol("hashtag_sentence")
    
    hashtag_extractor = FindHashtags(
        inputCol="text",
        outputCol="hashtag",
    )
    
    # hashtag_tokenizer = Tokenizer() \
    #     .setInputCols(["hashtag"]) \
    #     .setOutputCol("hashtag_tokens")

    # hashtag_embeddings = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
    #     .setInputCols(["document", "hashtag_tokens"]) \
    #     .setOutputCol("hashtag_embeddings")    
    
    # average_embeddings = SentenceEmbeddings() \
    #     .setInputCols(["document", "hashtag_embeddings"]) \
    #     .setOutputCol("average_hashtag_embeddings") \
    #     .setPoolingStrategy("AVERAGE")
    
    # mention_extractor = RegexMatcher() \
    #     .setInputCols(["document"]) \
    #     .setStrategy("MATCH_ALL") \
    #     .setExternalPattern("(\@[\w]+)") \
    #     .setOutputCol("mention")
    
    # url_extractor = RegexMatcher() \
    #     .setInputCols(["document"]) \
    #     .setStrategy("MATCH_ALL") \
    #     .setExternalPattern("((www\.|http://|https://)[^\s]+)") \
    #     .setOutputCol("url")
    
    # document_normalizer = DocumentNormalizer() \
    #     .setInputCols(["document"]) \
    #     .setOutputCol("normalized") \
    #     .setLowercase(True) \
    #     .setCleanupPatterns(["[^\w\d\s]"]) \
    #     .setReplacementPatterns([["\s+", " "], ["^\s+", ""]])
    


    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")
    
    language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
        .setInputCols(["document"])\
        .setOutputCol("language")
    
    word_tokenizer = Tokenizer() \
        .setInputCols("sentence") \
        .setOutputCol("word_token")
    
    # hasthag_tokenizer = Tokenizer() \
    #     .setInputCols("hashtag_sentence") \
    #     .setOutputCol("hashtag_token")





    # lemmatizer = LemmatizerModel.pretrained("lemma_antbnc") \
    #     .setInputCols(["normal"]) \
    #     .setOutputCol("lemma")
    # stopwords_cleaner = StopWordsCleaner() \
    #     .setInputCols(["lemma"]) \
    #     .setOutputCol("clean_lemma") \
    #     .setCaseSensitive(False)
    # finisher = Finisher() \
    #     .setInputCols(["clean_lemma"]) \
    #     .setOutputCols(["token_features"]) \
    #     .setOutputAsArray(True) \
    #     .setCleanAnnotations(False)
    
    embeddings_word = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
        .setInputCols("document", "word_token") \
        .setOutputCol("word_embeddings") \
        .setCaseSensitive(True) 
    
    embeddings_sentence = SentenceEmbeddings() \
        .setInputCols(["document", "word_embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")
    
    # embeddings_hasthag = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
    #     .setInputCols("hashtag_document", "hashtag_token") \
    #     .setOutputCol("hashtag_embeddings") \
    #     .setCaseSensitive(True)
    
    # embeddings_union = VectorAssembler() \
    #     .setInputCols(["sentence_embeddings", "average_hashtag_embeddings"]) \
    #     .setOutputCol("sentence_embeddings")
        
    embeddings_finisher = EmbeddingsFinisher() \
        .setInputCols("sentence_embeddings") \
        .setOutputCols("sentence_embeddings") \
        .setCleanAnnotations(False)
        
    # hashtag_embeddings_finisher = EmbeddingsFinisher() \
    #     .setInputCols("average_hashtag_embeddings") \
    #     .setOutputCols("average_hashtag_embeddings") \
    #     .setCleanAnnotations(False)
    
    # graph_preprocessing = TweetClusterPreprocessing(inputCol="sentence_embeddings", outputCol="final_embeddings")
    
    # BisectingKMeans()
    kmeans  = KMeans() \
        .setK(label_count) \
        .setSeed(1) \
        .setFeaturesCol("final_embeddings") \
        .setPredictionCol("cluster") \
        .setDistanceMeasure("cosine")
    
    label2idx = LabelsToIndex(l2c)

    random_forest =  RandomForestClassifier() \
        .setLabelCol("label_idx") \
        .setFeaturesCol("final_embeddings") \
        .setNumTrees(10) \


    return Pipeline(stages=[document_assembler,
                            hashtag_extractor,
                            # hashtag_tokenizer,
                            # hashtag_embeddings,
                            # average_embeddings,
                            # hasthag_assembler,
                            # hasthag_extractor, 
                            # hashtag_sentence,
                            sentence_detector,
                            language_detector, 
                            word_tokenizer,
                            # hasthag_tokenizer, 
                            embeddings_word, 
                            embeddings_sentence,
                            # embeddings_hasthag,
                            # embeddings_union, 
                            embeddings_finisher,
                            # hashtag_embeddings_finisher,
                            # graph_preprocessing 
                            ])
    
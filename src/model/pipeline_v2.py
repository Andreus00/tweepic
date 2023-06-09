import sparknlp
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
import re





def create_final_pipeline():
    http_filter = HttpCleaner(inputCol="text", outputCol="text")
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
        .setInputCols(["document"])\
        .setOutputCol("language")
    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")   
    tokenizer = Tokenizer() \
        .setInputCols("sentence") \
        .setOutputCol("token") 
    
    # INDIVIDUAZIONE @ # E punctuation
    hashtag_regex_matcher = CustomRegexMatcher(
        inputCol="token",
        outputCol="hashtag_idxs",
        regex=r"^#.*"
    )
    mentions_regex_matcher = CustomRegexMatcher(
        inputCol="token",
        outputCol="mentions_idxs",
        regex=r"^@.*"
    )
    punctuation_regex_matcher = CustomRegexMatcher(
        inputCol="token",
        outputCol="punctuation_idxs",
        regex=r"^[^\w\s]$"
    )

    # ESTRAZIONE HASHTAGS
    hashtag_extractor = ElementSelector(inputCols=["token", "hashtag_idxs"], outputCol="hashtags", attribute="result")
    
    # BERT
    embeddings_word = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
        .setInputCols("document", "token") \
        .setOutputCol("word_embeddings") \
        .setCaseSensitive(True)
    
    # SENTENCE BERT
    embeddings_sentence = SentenceEmbeddings() \
        .setInputCols(["document", "word_embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")
    embeddings_finisher = EmbeddingsFinisher() \
        .setInputCols("sentence_embeddings") \
        .setOutputCols("sentence_embeddings") \
        .setCleanAnnotations(False)
    embeddings_final_form = TweetEmbeddingPreprocessing(
        inputCol="sentence_embeddings", 
        outputCol="sentence_embeddings")
    
    # HASTHAGS
    hashtags_embeddings = ElementSelector(
        inputCols=["word_embeddings", "hashtag_idxs"],
        outputCol="hashtags_embeddings",
        attribute="embeddings"
    )

    # PUNCTUATION REMOVER
    embeddings_punctuation_remover = ElementRemover(
        inputCols=["word_embeddings", "punctuation_idxs"],
        outputCol="word_embeddings",
        idxs="punctuation_idxs"
    )
    
    
    return Pipeline(stages=[
        http_filter,
        document_assembler,
        language_detector,
        sentence_detector,
        tokenizer,
        hashtag_regex_matcher,
        mentions_regex_matcher,
        punctuation_regex_matcher,
        hashtag_extractor,
        embeddings_word,
        embeddings_sentence,
        embeddings_finisher,
        embeddings_final_form,
        hashtags_embeddings,
        embeddings_punctuation_remover
    ])


class HttpCleaner(Transformer):
    def __init__(self, inputCol, outputCol) -> None:
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.udf_f = udf(self.clean_http, StringType())
    
    def clean_http(self, text):
        return re.sub(r"http\S+", "http", text)
    
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(self.outputCol, self.udf_f(F.col(self.inputCol)))
        return dataset


class CustomRegexMatcher(Transformer):

    def __init__(self, inputCol, outputCol, regex) -> None:
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.regex = regex
        self.udf_f = udf(self.get_match_idxs, ArrayType(IntegerType()))

    def get_match_idxs(self, tokens, regex):
        return [idx for idx, token in enumerate(tokens) if re.match(regex, token.result)]
    

    def _transform(self, dataset: DataFrame) -> DataFrame:
        regex = self.regex
        dataset = dataset.withColumn(self.outputCol, self.udf_f(F.col(self.inputCol), F.lit(regex)))
        return dataset



class CustomRegexReplacer(Transformer):

    def __init__(self, inputCol, outputCol, regex, replace) -> None:
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.regex = regex
        self.replace = replace
        self.udf_f = udf(self.get_match_idxs, ArrayType(IntegerType()))

    def get_match_idxs(self, tokens, regex, replace):
        return [token.result if re.match(regex, token.result) else replace for token in tokens]
    

    def _transform(self, dataset: DataFrame, replace) -> DataFrame:
        regex = self.regex
        replace = self.replace
        dataset = dataset.withColumn(self.outputCol, self.udf_f(F.col(self.inputCol), F.lit(regex), F.lit(replace)))
        return dataset

class ElementSelector(Transformer):

    def __init__(self, inputCols, outputCol, attribute="resut") -> None:
        '''
        Gets two columns and returns a new column with the elements of the
        first column at the indexes specified in the second column modified
        by the function specified in the input
        '''
        super().__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol
        self.attribute = attribute
        self.udf_f = udf(self.get_match_idxs, ArrayType(StringType()))

    def get_match_idxs(self, tokens, idxs, attribute):
        return [getattr(tokens[idx], attribute) for idx in range(len(tokens)) if idx in idxs]
    
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(self.outputCol, self.udf_f(F.col(self.inputCols[0]), F.col(self.inputCols[1]), F.lit(self.attribute)))
        return dataset  
    

class ElementRemover(Transformer):

    def __init__(self, inputCols, outputCol, idxs) -> None:
        '''
        Gets two columns and returns a new column with the elements of the 
        first column at the indexes specified in the second column modified 
        by the function specified in the input
        '''
        super().__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol
        self.idxs = idxs
        self.udf_f = udf(self.get_match_idxs, ArrayType(StringType()))

    def get_match_idxs(self, tokens, idxs):
        return [tokens[idx].embeddings for idx in range(len(tokens)) if idx not in idxs]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset = dataset.withColumn(self.outputCol, self.udf_f(F.col(self.inputCols[0]), F.col(self.inputCols[1])))
        dataset.take(1)
        return dataset  
    
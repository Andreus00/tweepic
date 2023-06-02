import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark



spark = sparknlp.start(gpu=True)
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")
sentence_detector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")
tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")



embeddings = BertSentenceEmbeddings.pretrained("sent_electra_base_uncased", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I think covid has ruined our lives']], ["text"]))
result2 = pipeline_model.transform(spark.createDataFrame([['Superbowl is a beautiful event']], ["text"]))
result3 = pipeline_model.transform(spark.createDataFrame([['My balls are blue because of this project']], ["text"]))
emb1 = result.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings
emb2 = result2.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings
emb3 = result3.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings

import numpy as np
print(np.dot(emb1, emb2))
print(np.dot(emb1, emb3))
print(np.dot(emb2, emb3))
# # print the number of rows and columns
# print(result.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings)
# print(len(result.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings))
# # print(result.embeddings.count())

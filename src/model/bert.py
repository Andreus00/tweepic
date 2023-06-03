import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark
import numpy as np
from scipy.spatial import distance
from pyspark.ml.feature import VectorAssembler, HashingTF
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyspark.sql.functions import lit 
import pyspark.sql.functions as F


spark = sparknlp.start(gpu=True)

# load dataset

df = spark.read.parquet("data/parquet_multi_labels/tweets.parquet")
df = df.withColumn("text_with_date", F.concat_ws(" <sep> ", df["text"], df["year"].cast("string"), df["month"].cast("string"), df["day"].cast("string"), df["hashtags"], df["mentions"]))
df.show()
print(df.count())

document_assembler = DocumentAssembler() \
.setInputCol("text_with_date") \
.setOutputCol("document")
sentence_detector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")
tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")
normalizer = Normalizer() \
.setInputCols(["token"]) \
.setOutputCol("normal")
languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
    .setInputCols(["sentence"])\
    .setOutputCol("language")
# lemmatizer = LemmatizerModel.pretrained("lemma_antbnc") \
# .setInputCols(["normal"]) \
# .setOutputCol("lemma")
# stopwords_cleaner = StopWordsCleaner() \
# .setInputCols(["lemma"]) \
# .setOutputCol("clean_lemma") \
# .setCaseSensitive(False)
# hashingTF = HashingTF(inputCol="normal", outputCol="tf")

# https://github.com/JohnSnowLabs/spark-nlp/issues/7357         FOR THE PIPELINE

# https://www.johnsnowlabs.com/understanding-the-power-of-transformers-a-guide-to-sentence-embeddings-in-spark-nlp/
# sent_xlm_roberta_base, sent_roberta_base
# embeddings = RoBertaSentenceEmbeddings.pretrained("sent_roberta_base", "en") \

# embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base", "en") \
# embeddings = RoBertaSentenceEmbeddings.pretrained("sent_roberta_base", "en") \
embeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx") \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler,sentence_detector, tokenizer, normalizer, languageDetector, embeddings])
pipeline_model = nlp_pipeline.fit(df)
texts = df.select("text_with_date").collect()
print(texts[0].text_with_date)
result = pipeline_model.transform(df)

embeddings = result.select("sentence_embeddings").collect()
# TAKE THE EMBEDDINGS FROM THE ROWS
# print(len(embeddings))
# print(type(embeddings))
# print(embeddings[0].sentence_embeddings[0].embeddings)

# emb_df = spark.createDataFrame([embeddings], ["sentence_embeddings"])

# emb_size = len(embeddings)

# emb_df.show()
embs = []
# [Row(sentence_embeddings = [(end = 1942, ..., embeddings = [12, 41,23 ])])]
for row in embeddings:
    embs.append((Vectors.dense(row.sentence_embeddings[0].embeddings),))
pca_df = spark.createDataFrame(embs,["sentence_embeddings"])
pca = pyspark.ml.feature.PCA(k=2, inputCol="sentence_embeddings", outputCol="pca_features")
pca_result = pca.fit(pca_df).transform(pca_df).select("pca_features").toPandas()

classes = [
    '2016-brexit.ids',
    '2014-gazaunderattack.ids',
    '2013-boston-marathon-bombing.ids',
    '2015-charliehebdo.ids',
    '2015-germanwings-crash.ids',
    '2014-ebola.ids',
    '2012-uselection.ids',
    '2016-sismoecuador.ids',
    '2012-obama-romney.ids',
    '2014-indyref.ids',
    '2012-sxsw.ids',
    '2015-parisattacks.ids',
    '2016-hijacked-plane-cyprus.ids',
    '2015-refugeeswelcome.ids',
    '2014-stpatricksday.ids',
    '2012-mexican-election.ids',
    '2012-superbowl.ids',
    '2016-panamapapers.ids',
    '2016-irish-ge16.ids',
    '2015-nepalearthquake.ids',
    '2016-brussels-airport-explossion.ids',
    '2015-hurricanepatricia.ids',
    '2014-ferguson.ids',
    '2014-ottawashooting.ids',
    '2014-hongkong-protests.ids',
    '2014-typhoon-hagupit.ids',
    '2012-hurricane-sandy.ids',
    '2012-euro2012.ids',
    '2016-lahoreblast.ids'
]
l2c = {l:i for i,l in enumerate(classes)}
print(l2c)
labels = [l2c[row.label] for row in df.select("label").collect()]


plt.scatter(pca_result["pca_features"].apply(lambda v: v[0]), pca_result["pca_features"].apply(lambda v: v[1]), c=labels, cmap='viridis', label=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")
# plt.legend(
#     handles=[mpatches.Patch(color='C'+str(i), label=classes[i]) for i in range(len(classes))],
#     bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=1.0
# )
plt.show()

# result1 = pipeline_model.transform(spark.createDataFrame([["Messi ha vinto il mondiale."]], ["text"]))
# result2 = pipeline_model.transform(spark.createDataFrame([["Argentina won his third world cup! Messi was the MVP."]], ["text"]))
# result3 = pipeline_model.transform(spark.createDataFrame([[texts[3000].text]], ["text"]))


# emb1 = result1.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings
# emb2 = result2.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings
# emb3 = result3.select("sentence_embeddings").collect()[0].sentence_embeddings[0].embeddings

# similarity_a = 1 - distance.cosine(emb1, emb2)
# similarity_b = 1 - distance.cosine(emb2, emb3)
# similarity_c = 1 - distance.cosine(emb1, emb3)

# print("first sentence: ", df.select("text").collect()[1].text)
# print("second sentence: ", df.select("text").collect()[2].text)
# print("third sentence: ", df.select("text").collect()[3000].text)
# print(similarity_a)
# print(similarity_b)
# print(similarity_c)

# # eucl distance

# print(np.linalg.norm(np.asarray(emb1) - np.asarray(emb2)))
# print(np.linalg.norm(np.asarray(emb2) - np.asarray(emb3)))
# print(np.linalg.norm(np.asarray(emb1) - np.asarray(emb3)))


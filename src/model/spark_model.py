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
from pyspark.sql.functions import udf
import src.model.embedding_average as embedding_average
from sklearn.decomposition import PCA


def main():
    spark = sparknlp.start(gpu=True)

    # load dataset
    df = spark.read.parquet("data/parquet_multi_labels_no_filters/tweets.parquet")
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
    .setInputCols("sentence") \
    .setOutputCol("token")
    # languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
    #     .setInputCols(["sentence"])\
    #     .setOutputCol("language")
    # lemmatizer = LemmatizerModel.pretrained("lemma_antbnc") \
    # .setInputCols(["normal"]) \
    # .setOutputCol("lemma")
    # stopwords_cleaner = StopWordsCleaner() \
    # .setInputCols(["lemma"]) \
    # .setOutputCol("clean_lemma") \
    # .setCaseSensitive(False)
    # hashingTF = HashingTF(inputCol="normal", outputCol="tf")



    embeddings = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
    .setInputCols("document", "token") \
    .setOutputCol("word_embeddings")

    embeddingsSentence = SentenceEmbeddings() \
                .setInputCols(["document", "word_embeddings"]) \
                .setOutputCol("sentence_embeddings") \
                .setPoolingStrategy("AVERAGE")
    nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, embeddingsSentence])
    pipeline_model = nlp_pipeline.fit(df)
    texts = df.select("text_with_date").collect()
    hashtags = df.select("hashtags").collect()
    result = pipeline_model.transform(df)
    result.show()
    sentence_embeddings = []
    sentence_embeddings_row =result.select("sentence_embeddings").collect()
    for i, row in enumerate(sentence_embeddings_row):
        sentence_embeddings.append(row[0][0].embeddings)
    embeddings = result.select("word_embeddings").collect()

    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(sentence_embeddings)
    
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

    #plot
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA")
    ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], cmap='viridis', c=labels, label=labels)
    ax.grid()
    plt.show()
    plt.savefig("pca.png")

    q = 12
    query = texts[q]
    d = {}
    for i,tweet in enumerate(texts):
        if i == q:
            continue
        sim = 1-distance.cosine(sentence_embeddings[q],sentence_embeddings[i])
        d[tweet] = (labels[i], sim)
        
    print('Most similar to: ',query, "Class: ", labels[q])
    print('----------------------------------------')
    for idx,x in enumerate(sorted(d.items(), key=lambda x: x[1][1], reverse=True)):
        print(idx+1, "Sim: ", round(x[1][1],2), "Class: ", x[1][0], "Tweet: ", x[0])


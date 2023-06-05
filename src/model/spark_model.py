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
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from src.model.cluster import TweetClusterPreprocessing


def load_data(spark):
    # load dataset
    df = spark.read.parquet("data/parquet_big/tweets.parquet")
    data = [
    ["chatgpt","1123", "2016", "7", "31", "ChatGPT's sophisticated natural language processing capabilities enable it to generate human-like responses to a wide range of queries.", "", ""],
    ["chatgpt","1124", "2016", "7", "31", "With its comprehensive training on diverse topics, ChatGPT can understand and generate text on a wide range of subjects.", "", ""],
    ["2016-panamapapers.ids","1123", "2016", "6", "25", "A diabetic food is any pathology that results directly from peripheral arterial disease.", "", ""],
    ["chatgpt","1126", "2019", "4", "20", "L'intelligenza artificiale è in grado di capire il linguaggio umano e fornire risposte complesse.", "", ""],
    ["chatgpt", "12345", "2023", "06", "05", "L'intelligenza artificiale sta cambiando il futuro, aprendo nuove opportunità e sfidando i confini dell'innovazione. #IA #TecnologiaAvanzata", "IA, TecnologiaAvanzata",""],
    ["chatgpt", "67890", "2023", "07", "31", "Formula One is harnessing the power of artificial intelligence to enhance performance, optimize strategies, and push boundaries. #AI #F1", "AI, F1",""],
    ["chatgpt", "54321", "2023", "08", "01", "Une brève explication d'un réseau neuronal : un modèle de l'intelligence artificielle inspiré du fonctionnement du cerveau humain, capable d'apprendre et de résoudre des problèmes complexes. #IA #RéseauNeuronal", "IA, RéseauNeuronal",""],
    ["chatgpt", "09876", "2023", "06", "05", "Künstliche Intelligenz birgt potenzielle Gefahren, da sie mit zunehmender Autonomie und Komplexität ethische Herausforderungen mit sich bringt. Wir müssen verantwortungsbewusst damit umgehen. #KI #Ethik", "KI, Ethik",""],
    ["chatgpt", "65432", "1999", "01", "23", "Los bots en Rocket League pueden ser divertidos, pero a veces sus decisiones son simplemente absurdas. ¡Pero eso los hace únicos y entretenidos! #RocketLeague #Bots", "RocketLeague, Bots",""]
    ]
    # data = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9]
    
    df_test = spark.createDataFrame(data, ["label", "id", "year", "month", "day", "text", "hashtags", "mentions"])
    df = df_test.union(df)
    # df = df.withColumn("text_with_info", F.concat_ws(" <sep> ", df["text"], df["year"].cast("string"), df["month"].cast("string"), df["day"].cast("string"), df["hashtags"], df["mentions"]))
    df = df.withColumn("text_with_info", F.concat_ws(" <sep> ", df["text"], df["hashtags"], df["mentions"]))
    df.show()
    print(df.count())
    return df


def create_pipeline(label_count):
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())


    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    # document_assembler2 = DocumentAssembler() \
    #     .setInputCol("hashtags") \
    #     .setOutputCol("hashtags_document")
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
    # embeddings_bert = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base", "xx") \
    # .setInputCols("document") \
    # .setOutputCol("sentence_embeddings")\
    # .setCaseSensitive(True) \
    # .setMaxSentenceLength(512)

    
    cluster_preprocess = TweetClusterPreprocessing(inputCol="sentence_embeddings", outputCol="cluster_features")
    
    # BisectingKMeans()
    kmeans  = BisectingKMeans() \
        .setK(label_count) \
        .setSeed(1) \
        .setFeaturesCol("cluster_features") \
        .setPredictionCol("cluster") \
        .setDistanceMeasure("cosine")

    # kmeans_evaluator = ClusteringEvaluator(
    #     predictionCol='cluster',
    #     featuresCol='cluster_features'
    # )
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

    return Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddingsWord, embeddingsSentence, cluster_preprocess, kmeans])
    



def main():
    spark = sparknlp.start(gpu=True)

    df = load_data(spark)

    df = df.filter(df["text"] == "")

    df = df.limit(313)

    df.subtract(df.limit(309)).show()
    
    label_count = len(df.groupBy("label").count().collect())
    print(df.groupBy("label").count().collect())

    print(label_count)

    nlp_pipeline = create_pipeline(label_count=label_count)

    pipeline_model = nlp_pipeline.fit(df)

    texts = df.select("text").collect()
    hashtags = df.select("hashtags").collect()
    
    result = pipeline_model.transform(df)
    result.show()
    
    sentence_embeddings = np.asarray(result.select("cluster_features").collect())
    sentence_embeddings = sentence_embeddings.reshape(sentence_embeddings.shape[0], sentence_embeddings.shape[2])
    
    word_embeddings = result.select("word_embeddings").collect()
    
    clusters = [x.cluster for x in result.select("cluster").collect()]


    classes = [
        '2014-gazaunderattack.ids',
        '2013-boston-marathon-bombing.ids',
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
        '2015-hurricanepatricia.ids',
        '2014-ferguson.ids',
        '2014-ottawashooting.ids',
        '2014-hongkong-protests.ids',
        '2014-typhoon-hagupit.ids',
        '2012-hurricane-sandy.ids',
        '2012-euro2012.ids',
        '2015-charliehebdo.ids',
        '2015-germanwings-crash.ids',
        '2016-euro2016.ids',
        '2016-brexit.ids',
        '2016-brussels-airport-explossion.ids',
        '2016-lahoreblast.ids',
        'chatgpt'
    ]
    classes.sort()

    l2c = {l:i for i,l in enumerate(classes)}
    labels = [l2c[row.label] for row in df.select("label").collect()]

    plot_cluster(sentence_embeddings, labels, clusters)

    text_similarity(texts, sentence_embeddings, labels, q=10)




def plot_cluster(sentence_embeddings, labels, clusters):

    # two plots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(sentence_embeddings)
    fig, (ax1,ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='3d'), figsize=(12, 4))
    # ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("PCA")
    ax1.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], cmap='viridis', c=labels, label=labels)
    ax1.grid()

    # ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("Clusters")
    ax2.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], cmap='viridis', c=clusters, label=clusters)
    ax2.grid()

    plt.legend()
    plt.show()
    plt.savefig("pca.png")


def text_similarity(texts, sentence_embeddings, labels, q=0):
    # q = 0 # texts.index("Match passionnant entre l'Angleterre et l'Italie aujourd'hui. <sep> 2012 <sep> 6 <sep> 25 <sep> <sep>")
    query = texts[q]  # Orribile attentato alla rambla questa notte. <sep> 2017 <sep> 8 <sep> 17 <sep> <sep>
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


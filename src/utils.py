from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pyspark.sql.functions as F
import config
import numpy as np

def load_data(path, spark):
    # load dataset
    df = spark.read.parquet(path)
    # data = [
    # ["chatgpt","1123", "2016", "7", "31", "#ChatGPT's sophisticated #naturallanguage processing capabilities enable it to generate human-like responses to a wide range of queries."],
    # ["chatgpt","1124", "2016", "7", "31", "With its comprehensive training on diverse topics, ChatGPT can understand and generate text on a wide range of subjects."],
    # ["2016-panamapapers.ids","1127", "2016", "6", "25", "A diabetic food is any pathology that results directly from peripheral arterial disease."],
    # ["chatgpt","1126", "2019", "4", "20", "L'intelligenza artificiale è in grado di capire il linguaggio umano e fornire risposte complesse."],
    # ["chatgpt", "12345", "2023", "06", "05", "L'intelligenza artificiale sta cambiando il futuro, aprendo nuove opportunità e sfidando i confini dell'innovazione. #IA #TecnologiaAvanzata"],
    # ["chatgpt", "67890", "2023", "07", "31", "Formula One is harnessing the power of artificial intelligence to enhance performance, optimize strategies, and push boundaries. #AI #F1"],
    # ["chatgpt", "54321", "2023", "08", "01", "Une brève explication d'un réseau neuronal : un modèle de l'intelligence artificielle inspiré du fonctionnement du cerveau humain, capable d'apprendre et de résoudre des problèmes complexes. #IA #RéseauNeuronal"],
    # ["chatgpt", "09876", "2023", "06", "05", "Künstliche Intelligenz birgt potenzielle Gefahren, da sie mit zunehmender Autonomie und Komplexität ethische Herausforderungen mit sich bringt. Wir müssen verantwortungsbewusst damit umgehen. #KI #Ethik"],
    # ["chatgpt", "65432", "1999", "01", "23", "Los bots en Rocket League pueden ser divertidos, pero a veces sus decisiones son simplemente absurdas. ¡Pero eso los hace únicos y entretenidos! #RocketLeague #Bots"]
    # ]
    # # data = [sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, sentence7, sentence8, sentence9]
    
    # df_test = spark.createDataFrame(data, ["label", "id", "year", "month", "day", "text"])
    # df = df_test.union(df)

    df = df.filter(df["text"] != "")
    # df = df.filter(df["year"] != "")
    # df = df.filter(df["label"] != "")
    # df = df.filter(df["month"] != "")
    # df = df.filter(df["day"] != "")
    # df = df.withColumn("text", F.regexp_replace("text", r"^RT\s+", ""))

    # df = df.withColumn("text_with_info", F.concat_ws(" <sep> ", df["text"], df["year"].cast("string"), df["month"].cast("string"), df["day"].cast("string"), df["hashtags"], df["mentions"]))
    # df = df.withColumn("text_with_info", F.concat_ws(" <sep> ", df["text"], df["hashtags"], df["mentions"]))
    print("Dataframe loaded")
    df.show()
    print("num rows:", df.count())
    return df



def slice_df(df, spark, start, end):
    return spark.createDataFrame(df.limit(end).tail(end - start))


def init_labels():
    classes = config.classes
    classes.sort()

    return {l:i for i,l in enumerate(classes)}


def plot_cluster(sentence_embeddings, labels):

    # two plots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(sentence_embeddings)
    fig, ax1 = plt.subplots(1,1, subplot_kw=dict(projection='3d'), figsize=(12, 4))
    # ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("PCA")
    ax1.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], cmap='viridis', c=labels, label=labels)
    ax1.grid()


    plt.show()
    fig.savefig("images/pca.png")


def text_similarity(texts, sentence_embeddings, labels, q=0, input_tweet=None, sent_emb=None):
    # q = 0 # texts.index("Match passionnant entre l'Angleterre et l'Italie aujourd'hui. <sep> 2012 <sep> 6 <sep> 25 <sep> <sep>")
    query = texts[q]  # Orribile attentato alla rambla questa notte. <sep> 2017 <sep> 8 <sep> 17 <sep> <sep>
    if input_tweet is not None:
        query = input_tweet
        
    d = {}
    for i,tweet in enumerate(texts):
        if i == q:
            continue
        if sent_emb is not None:
            sim = 1-distance.cosine(sent_emb, sentence_embeddings[i][0].toArray())
        else:
            sim = 1-distance.cosine(sentence_embeddings[q],sentence_embeddings[i])
        d[tweet] = (labels[i], sim)
        
    print('Most similar to: ',query) # , "Class: ", labels[q])
    print('----------------------------------------')
    i = 0
    for idx,x in enumerate(sorted(d.items(), key=lambda x: x[1][1], reverse=True)):
        if i == 10:
            break
        print(idx+1, "Sim: ", round(x[1][1],2), "Class: ", x[1][0], "Tweet: ", x[0])
        i += 1




def plot_online_PCA(sentence_embeddings, labels, tweet, tweet_label):

    # add tweet to sentence_embeddings
    sentence_embeddings = [x[0].toArray() for x in sentence_embeddings]
    sentence_embeddings = np.vstack((sentence_embeddings, tweet))
    labels = np.append(labels, tweet_label)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(sentence_embeddings)
    
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("PCA")
    
    # Plot all points in gray
    ax1.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], cmap='gray', c='gray', label='Other')
    
    # Plot points of class "x" in a different color
    x_indices = np.where(labels == tweet_label)[0]
    ax1.scatter(pca_result[x_indices, 0], pca_result[x_indices, 1], pca_result[x_indices, 2], cmap='viridis', c='green', label=tweet_label)

    # plot tweet in red
    ax1.scatter(pca_result[-1, 0], pca_result[-1, 1], pca_result[-1, 2], cmap='viridis', c='red', label='Tweet')
    
    ax1.grid()
    ax1.legend()
    
    plt.show()



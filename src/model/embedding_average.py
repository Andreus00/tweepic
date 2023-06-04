import pyspark
from pyspark.ml import Transformer
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
import numpy as np

udf_func = F.udf(lambda x: map(lambda v: Vectors.dense(v), np.average(x.to_numpy(), axis=0)), ArrayType(FloatType()))

class EmbeddingAverage(Transformer):
    def __init__(self, inputCol, outputCol):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.avg_lists_udf = udf(self.avg_lists, ArrayType(FloatType()))
    # Definisci una UDF per estrarre gli embeddings da una riga
    # @staticmethod
    # @F.udf(returnType=ArrayType(ArrayType(FloatType())))
    # def extract_embeddings(row):
    #     return row[0].embeddings

    # # Definisci una UDF per calcolare la media di una lista di liste
    # @staticmethod
    # @F.udf(returnType=ArrayType(FloatType()))
    # def avg_lists(lists):
    #     return [float(sum(x))/len(x) for x in zip(*lists)]

    # def _transform(self, df):
    #     rows = df.select(df[self.inputCol]).collect()
    #     embeddings = []
    #     for row in rows:
    #         word_embeddings = []
    #         for word in row[0]:
    #             word_embeddings.append(word.embeddings)
    #         embeddings.append(word_embeddings)
    #     embeddings = [np.average(sublist, axis=0) for sublist in embeddings]
    #     spark = pyspark.sql.SparkSession.builder.getOrCreate()
    #     # [[1,3,2][14,32,14]]
    #     return df.withColumn(self.outputCol, F.array([F.lit(sublist) for sublist in embeddings]))
    
    # def _transform(self, df):
    #     # Calcola gli embeddings medi come prima
    #     df_expanded = df.select(F.col("id"), F.col(self.inputCol).embeddings.alias(self.inputCol))
    #     df_avg = df_expanded.withColumn(self.outputCol, udf_func(df_expanded[self.inputCol]))
        
    #     # Crea una nuova colonna con gli embeddings medi esplosi
    #     # df = df.withColumn(self.outputCol, F.explode(df_avg["average"]))
        
    #     return df_avg

        # Definisci una UDF per calcolare la media di una lista
    @staticmethod
    def avg_lists(lists):
        return [float(sum(x))/len(x) for x in zip(*lists)]

    

    def _transform(self, df):
        df_expanded = df.select(F.col("id"), F.col(self.inputCol).embeddings.alias(self.inputCol))
        df_avg = df_expanded.withColumn(self.outputCol, self.avg_lists_udf(df_expanded[self.inputCol]))
        return df_avg
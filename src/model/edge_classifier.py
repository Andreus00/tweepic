from pyspark.ml import Pipeline
from pyspark.sql.functions import lit 
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, ArrayType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.base import Transformer, Estimator
from pyspark.ml.util import MLWritable, MLReadable, MLReader, MLWriter
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from igraph import *
import config


def create_edge_classifier_pipeline():
    stages = [
        Join(),
        LabelPreprocessor(),
        WeightClassifier()
    ]
    
    return Pipeline(stages=stages)
    

class Join(Transformer, MLWritable, MLReadable):
    def __init__(self):
        super(Join, self).__init__()
    
    def _transform(self, data):
        print("## JOIN ##")
        edges = data["edges"]
        vertices = data["vertices"]
        edges = edges.join(vertices, edges.src == vertices.id, "inner").drop("id", "text").withColumnRenamed("label", "src_label")
        edges = edges.join(vertices, edges.dst == vertices.id, "inner").drop("id", "text").withColumnRenamed("label", "dst_label")
        return edges
    
    def write(self):
        return MLWriter()
    
    def read(self):
        return MLReader()
    

class LabelPreprocessor(Transformer, MLWritable, MLReadable):
    def __init__(self):
        super(LabelPreprocessor, self).__init__()
        self.udf_func = udf(self._udf, IntegerType())
        
    def _udf(self, src_label, dst_label):
        return int(src_label == dst_label)
    
    def _transform(self, edges):
        return edges.withColumn("label", self.udf_func(F.col("src_label"), F.col("dst_label"))).drop("src_label").drop("dst_label")
    
    def write(self):
        return MLWriter()
    
    def read(self):
        return MLReader()

class WeightClassifier(Estimator, MLWritable):

    def __init__(self):
        super(WeightClassifier, self).__init__()
        self.model = RandomForestClassifier()\
                        .setFeaturesCol("features") \
                        .setLabelCol("label") \
                        .setPredictionCol("prediction")
        
        self.paramGrid = ParamGridBuilder()\
            .addGrid(self.model.numTrees, [30]) \
            .addGrid(self.model.maxDepth, [15]) \
            .addGrid(self.model.maxBins, [32]) \
            .build()
        self.trainValidationStep = TrainValidationSplit() \
            .setEstimator(self.model) \
            .setEvaluator(BinaryClassificationEvaluator()
                          .setMetricName("areaUnderROC")) \
            .setEstimatorParamMaps(self.paramGrid) \
            .setTrainRatio(0.8) \
            .setParallelism(12)
        self.array_to_vector_udf = udf(lambda l: Vectors.dense(l[:config.NUM_FEATURES_CLASSIFICATION]), VectorUDT())
        self.best_model = None
        
    def _fit(self, edges):
        edges = edges.withColumn("features", self.array_to_vector_udf(edges["relationship"]))
        self.best_model = self.trainValidationStep.fit(edges)
        return self
    
    def transform(self, edges):
        print("## WEIGHT CLASSIFIER ##")
        edges = edges.withColumn("features", self.array_to_vector_udf(edges["relationship"]))
        return self.best_model.transform(edges).drop("rawPrediction", "probability").drop("features")
    
    def write(self):
        wr = MLWriter()
        wr.option("best_model", self.best_model.write())

import sys
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Spark ML").getOrCreate()

test = spark.read.format("csv").option("delimiter", ";").option("header", "true").option("inferSchema", "true").load(sys.argv[1])

new_columns = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar","chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density","pH", "sulphates", "alcohol", "quality"]

for i in range(len(new_columns)):
        test = test.withColumnRenamed(test.columns[i], new_columns[i])

featureAssembler = VectorAssembler(inputCols=new_columns[:-1], outputCol='features')
test = featureAssembler.transform(test)

featureScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
featureScalerModel = featureScaler.fit(test)

test = featureScalerModel.transform(test)

model_path = "saved_dt_model"
saved_model = DecisionTreeClassificationModel.load(model_path)

predictions = saved_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

f1 = evaluator.evaluate(predictions)
print("F1 Score for the prediction is: ", f1)

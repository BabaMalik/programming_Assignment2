#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Spark ML").getOrCreate()


# In[9]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler


# In[12]:


column_names = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar","chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density","pH", "sulphates", "alcohol", "quality"]


# In[14]:


train = spark.read.format("csv").option("delimiter", ";").option("header", "true").option("inferSchema", "true").load("s3://myprobucket001/TrainingDataset.csv")

for i in range(len(column_names)):
        train = train.withColumnRenamed(train.columns[i], column_names[i])
        
featureAssembler = VectorAssembler(inputCols=column_names[:-1], outputCol='features')
train = featureAssembler.transform(train)

featureScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
featureScalerModel = featureScaler.fit(train)

train = featureScalerModel.transform(train)


# In[21]:


validation = spark.read.format("csv").option("delimiter", ";").option("header", "true").option("inferSchema", "true").load("s3://myprobucket001/ValidationDataset.csv")

for i in range(len(column_names)):
        validation = validation.withColumnRenamed(validation.columns[i], column_names[i])
        
featureAssembler = VectorAssembler(inputCols=column_names[:-1], outputCol='features')
validation = featureAssembler.transform(validation)

featureScaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
featureScalerModel = featureScaler.fit(validation)

val = featureScalerModel.transform(validation)


# In[26]:


from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define logistic regression model
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='quality')

# Train logistic regression model
lr_model = lr.fit(train)

# Save logistic regression model
lr_model.save("s3://myprobucket001/saved_lr_model")

# Define decision tree model
dt = DecisionTreeClassifier(featuresCol='scaledFeatures', labelCol='quality')

# Train decision tree model
dt_model = dt.fit(train)

# Save decision tree model
dt_model.save("s3://myprobucket001/saved_dt_model")

# Make predictions using logistic regression and decision tree model
lr_predictions = lr_model.transform(val)
dt_predictions = dt_model.transform(val)

# Define evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")

# Evaluate predictions and print F1 scores
lr_f1 = evaluator.evaluate(lr_predictions)
dt_f1 = evaluator.evaluate(dt_predictions)

print("Logistic Regression F1 Score: ", lr_f1)
print("Decision Tree F1 Score: ", dt_f1)



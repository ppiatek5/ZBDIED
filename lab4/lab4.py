from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("BankChurnMLlib").getOrCreate()
data = spark.read.csv('C:/Users/mrbre/Downloads/bank.csv', header=True, inferSchema=True)

feature_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
data_transformed = assembler.transform(data)

train_data, test_data = data_transformed.randomSplit([0.7, 0.3], seed=42)

rf = RandomForestClassifier(labelCol='churn', featuresCol='features')

pipeline = Pipeline(stages=[rf])
pipeline_model = pipeline.fit(train_data)
predictions = pipeline_model.transform(test_data)
predictions.select('prediction', 'churn').show(5)

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [5, 10])
             .addGrid(rf.numTrees, [20, 50])
             .build())

evaluator = BinaryClassificationEvaluator(labelCol='churn')
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
cv_model = crossval.fit(train_data)

predictions = cv_model.transform(test_data)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
predictions.select('prediction', 'churn').show(5)

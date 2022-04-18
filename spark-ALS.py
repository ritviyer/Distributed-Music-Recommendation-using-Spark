#import findspark
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import *

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
import numpy as np

resultPath = '../results/'
f = open(resultPath + 'resultMillionSong.txt','a')
f.write("\nFor Million Song \n")

ss= SparkSession.builder.appName('Music Recommendation').getOrCreate()
sc = ss._sc

dataPath = '../dataset/Million Song/'
data = ss.read.option('header','true').csv(dataPath+'triplets.csv').limit(100000)

indexers = [StringIndexer(inputCol=column, outputCol=column + '_index').fit(data) for column in ['userID','SongID']]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)
data = data.withColumn("Rating", data["Rating"].cast(DoubleType()))

(train, test) = data.randomSplit([0.8, 0.2])

als = ALS(userCol="userID_index", itemCol="SongID_index", ratingCol="Rating", coldStartStrategy="drop")
paramGrid = (ParamGridBuilder()
             .addGrid(als.rank, [20])
             .addGrid(als.regParam, [0.25])
             .addGrid(als.maxIter, [20])
             .addGrid(als.implicitPrefs, [True])
             .addGrid(als.alpha, [40])
             .build())
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating",predictionCol="prediction")
cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=4, seed = 0)
model = cv.fit(train)

f.write('\n')
f.write("Best Model = " + str(model.bestModel) + '\n')
f.write("CV RMSE of best model = " + str(min(model.avgMetrics)) + '\n')

best_params = model.getEstimatorParamMaps()[np.argmin(model.avgMetrics)]
for i,j in best_params.items():
  f.write(i.name+': '+str(j) + '\n')


predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
f.write("Root-mean-square error = " + str(rmse) + '\n')


ss.stop()
f.close()

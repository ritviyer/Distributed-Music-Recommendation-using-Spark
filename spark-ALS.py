#import findspark
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import *

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType

resultPath = '../results/'
f = open(resultPath + 'resultLastFM.txt','w')

ss= SparkSession.builder.appName('Music Recommendation').getOrCreate()
sc = ss._sc

dataPath = '../dataset/hetrec2011-lastfm-2k/'
data = ss.read.option('header','true').csv(dataPath+'user_artists.csv')
data = data.withColumn("userID", data["userID"].cast(IntegerType()))
data = data.withColumn("artistID", data["artistID"].cast(IntegerType()))
data = data.withColumn("Weight", data["Weight"].cast(IntegerType()))

(train, test) = data.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="artistID", ratingCol="Weight", coldStartStrategy="drop")
model = als.fit(train)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Weight",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
f.write("Root-mean-square error = " + str(rmse) + '\n')


# userRecs = model.recommendForAllUsers(10)
# artistRecs = model.recommendForAllItems(10)

users = data.select(als.getUserCol()).distinct().limit(5)
userSubsetRecs = model.recommendForUserSubset(users, 5)

artists = data.select(als.getItemCol()).distinct().limit(5)
artistSubSetRecs = model.recommendForItemSubset(artists, 5)


userSubsetRecs.toPandas().to_csv(resultPath + 'userSubsetRecs.csv', index=False)
artistSubSetRecs.toPandas().to_csv(resultPath + 'artistSubSetRecs.csv', index = False)


ss.stop()
f.close()

#import findspark
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import *

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import time

start_time = time.time()

resultPath = '../results/'

dataPath = '../dataset/Million Song/'
songOrArtist = 'SongID'
ratingCol = 'Rating'
datasetName = "MillinSongDataset"
dataFile = 'songCSV.csv'
is_Normalized = 'Yes'

core_used = 8
rnk = 16
regP = 0.25
maxI = 15
a = 40

f = open(resultPath + 'resultMillionSong.txt','a')
f.write('\n\n\n' + datasetName + '\n')
f.write("Cores Used: " + str(core_used) + '\n')
f.write("is_Normalized: " + str(is_Normalized) + '\n')
f.write("Rank = " + str(rnk) + '\n')
f.write("RegParam = " + str(regP) + '\n')
f.write("MaxIter = " + str(maxI) + '\n')
f.write("Alpha = " + str(a) + '\n')

ss= SparkSession.builder.appName('Music Recommendation').getOrCreate()
sc = ss._sc

data = ss.read.option('header','true').csv(dataPath+dataFile).limit(100000)

indexers = [StringIndexer(inputCol=column, outputCol=column + '_index').fit(data) for column in ['userID',songOrArtist]]
pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)
data = data.withColumn(ratingCol, data[ratingCol].cast(IntegerType()))

(train, test) = data.randomSplit([0.8, 0.2])

als = ALS(maxIter=maxI, regParam=regP, rank = rnk, alpha=a, userCol="userID_index", itemCol=songOrArtist+"_index", ratingCol=ratingCol, coldStartStrategy="drop")
model = als.fit(train)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol=ratingCol,predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
f.write("Root-mean-square error = " + str(rmse) + '\n')


# userRecs = model.recommendForAllUsers(10)
# artistRecs = model.recommendForAllItems(10)

users = data.select(als.getUserCol()).distinct().limit(5)
userSubsetRecs = model.recommendForUserSubset(users, 10)

artists = data.select(als.getItemCol()).distinct().limit(5)
artistSubSetRecs = model.recommendForItemSubset(artists, 10)


userSubsetRecs.toPandas().to_csv(resultPath + 'userSubsetRecs.csv', index=False)
artistSubSetRecs.toPandas().to_csv(resultPath + 'artistSubSetRecs.csv', index = False)


ss.stop()

end_time = time.time() - start_time
f.write("Total time (seconds): " + str(end_time) + '\n')
f.write("Total time (minutes): " + str(end_time/60) + '\n')
f.close()
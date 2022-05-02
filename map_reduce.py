from cmath import sqrt
from distutils.log import ERROR, error
from operator import index
from numpy import true_divide
from collections import OrderedDict 
import findspark
import math 
import pandas as pd
import time

findspark.init()

df = []
song_id = ""
already_displayed = []

def remove_duplicates(rdd):
    a,b = rdd
    c,d = b
    x,y = c
    u,v = d
    if x == u :
        return False
    else : 
        return True

def justsongpairs(rdd):
    a,b = rdd
    c,d = b
    x,y = c
    u,v = d
    return ((x,u),(y,v))

def findCosineSimilarity(rdd):
    num_of_pairs = 0
    xxsum = 0.0
    yysum = 0.0
    xysum = 0.0

    for pair in rdd:
        X, Y = pair
        ratingX = float(X)
        ratingY = float(Y)
        xxsum = xxsum + ratingX*ratingX
        yysum = yysum + ratingY*ratingY
        xysum = xysum + ratingX*ratingY
        num_of_pairs = num_of_pairs+1
    
    denom = 0.0 + math.sqrt(xxsum)*math.sqrt(yysum)

    result = 0.0 + xysum/denom
    return (result,num_of_pairs)

def filterOnThreshold(rdd):
    cosineThreshold = 0.97
    countThreshold = 0
    a,b = rdd
    x,y = a
    u,v = b
    if (x == song_id or y == song_id) and u > cosineThreshold and v > countThreshold :
        return True
    else : 
        return False

def displayTop10(top10, song_id):
    index = 1
   
    for a,b in top10:
        x,y = a
        u,v = b
        if x == song_id and x not in already_displayed:
            row = df.loc[df['SongID'] == y]
            row = row.drop_duplicates(subset = ['SongID'])
            song_title = row['songTitle'].to_string(index = False)
            artist = row['artistName'].to_string(index = False)
            print(song_title + "," + artist)
            already_displayed.append(x)
            index = index+1
        elif y == song_id and y not in already_displayed:
            row = df.loc[df['SongID'] == x]
            row = row.drop_duplicates(subset = ['SongID'])
            song_title = row['songTitle'].to_string(index = False)
            artist = row['artistName'].to_string(index = False)
            print(song_title + "," + artist)
            already_displayed.append(y)
            index = index+1

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
spark=SparkSession.builder\
    .master("local[*]")\
    .appName("Music recommendation")\
    .getOrCreate()
sc=spark.sparkContext
sc.setLogLevel("ERROR")
data_path = "C:\\Users\\birur\\Downloads\\song_dataset10.csv"
start = time.time()
rdd = sc.textFile(data_path)\
            .map(lambda x: x.split(",")).map(lambda x : ((x[1]),(x[2],x[3]))) 
songpair = rdd.join(rdd)
songpair_withoutdups = songpair.filter(remove_duplicates)
just_songpairs = songpair_withoutdups.map(justsongpairs)
groupOfRatingPairs = just_songpairs.groupByKey()
songPairsAndSimilarityScore = groupOfRatingPairs.mapValues(findCosineSimilarity)
df=pd.read_csv(data_path,usecols=[1,2,3,5,6], names=['userID','SongID','Rating','artistName','songTitle'], header = None)

userids = ["b80344d063b5ccb3212f76538f3d9e43d87dca9e", "7c86176941718984fed11b7c0674ff04c029b480", "76235885b32c4e8c82760c340dc54f9b608d7d7e",
"250c0fa2a77bc6695046e7c47882ecd85c42d748" , "3f73f44560e822344b0fb7c6b463869743eb9860"]
for user in userids:
    row = df.loc[df['userID'] == user]
    row = row.sort_values(by=['Rating'])
    top5_songs = row['SongID'].to_list()
    top5_songs = list(set(top5_songs))
    top5_songs = top5_songs[0:5]
    print("\n \n Loading song suggestions for user " + user)
    print("Here are the top recommendations for you: \n")
    i = 0
    top50 = []
    for song in top5_songs:
        song_id = song
        #print("Song id = " + song_id)
        filteredsongSimilarity = songPairsAndSimilarityScore.filter(filterOnThreshold)
        top10 = filteredsongSimilarity.take(10)
        #print(top10)
        top50.extend(top10)
    top50.sort(key = lambda x: x[1][0])
    top50 = top50[0:20]
    #print(top50)
    already_displayed.clear()
    for song in top5_songs:
        song_id = song
        displayTop10(top50,song_id)
end = time.time()
print(end-start)

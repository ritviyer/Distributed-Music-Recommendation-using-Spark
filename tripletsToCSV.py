import pandas as pd

dataPath = '../dataset/Million Song/'

df = pd.read_csv(dataPath + 'train_triplets.txt', names = ['triplets'])
df = df['triplets']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('\t'))

newList = pd.DataFrame(newList,columns=['userID','SongID','Rating'])
newList.to_csv(dataPath+'triplets.csv', index=False)



df = pd.read_csv(dataPath + 'unique_tracks.txt', names = ['data'])
df = df['data']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('<SEP>'))

newList = pd.DataFrame(newList,columns=['trackID','SongID','artistName','songTitle'])
newList.to_csv(dataPath+'unique_tracks.csv', index=False)


df = pd.read_csv(dataPath + 'unique_artists.txt', names = ['data'])
df = df['data']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('<SEP>'))

newList = pd.DataFrame(newList,columns=['artistID','artistMBID','trackID','artistName'])
newList.to_csv(dataPath+'unique_artists.csv', index=False)
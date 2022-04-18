import pandas as pd

dataPath = '../dataset/Million Song/'

df = pd.read_csv(dataPath + 'train_triplets.txt', names = ['triplets'])
print(df.head())
df = df['triplets']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('\t'))

newList = pd.DataFrame(newList,columns=['userID','SongID','Rating'])
newList.to_csv(dataPath+'triplets.csv', index=False)
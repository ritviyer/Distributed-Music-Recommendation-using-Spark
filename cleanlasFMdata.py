import pandas as pd

dataPath = '../dataset/hetrec2011-lastfm-2k/'

df = pd.read_csv(dataPath + 'user_artists.dat')
df = df['userID\tartistID\tweight']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('\t'))

newList = pd.DataFrame(newList,columns=['userID','artistID','Weight'])
newList = newList.apply(pd.to_numeric)
newList.to_csv(dataPath+'user_artists.csv', index=False)
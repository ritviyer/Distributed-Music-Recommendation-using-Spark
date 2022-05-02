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



df = pd.read_csv(dataPath + 'tags.dat')
df = df['tagID\ttagValue']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('\t'))

newList = pd.DataFrame(newList,columns=['tagID','tagValue'])
newList.to_csv(dataPath+'tags.csv', index=False)



df = pd.read_csv(dataPath + 'user_friends.dat')
df = df['userID\tfriendID']
newList = []
for ind, row in df.iteritems():
    newList.append(row.split('\t'))

newList = pd.DataFrame(newList,columns=['userID','friendID'])
newList = newList.apply(pd.to_numeric)
newList.to_csv(dataPath+'user_friends.csv', index=False)

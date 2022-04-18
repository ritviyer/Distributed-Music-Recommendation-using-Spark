import pandas as pd

dataPath = '../dataset/hetrec2011-lastfm-2k/'

df = pd.read_csv(dataPath + 'user_artists.csv')

df_sum = df.groupby('userID')['Weight'].sum().reset_index()
df['sum'] = df.userID.map(df_sum.set_index('userID').Weight)
df['Weight'] = (df['Weight'] / df['sum']) * 5

df.drop(columns=['sum'],inplace=True)

df.to_csv(dataPath + 'user_artists.csv', index = False)



dataPath = '../dataset/Million Song/'

df = pd.read_csv(dataPath + 'songCSV.csv')

df_sum = df.groupby('userID')['Rating'].sum().reset_index()
df['sum'] = df.userID.map(df_sum.set_index('userID').Rating)
df['Rating'] = (df['Rating'] / df['sum']) * 5

df.drop(columns=['sum'],inplace=True)


df.to_csv(dataPath + 'songCSV.csv', index = False)
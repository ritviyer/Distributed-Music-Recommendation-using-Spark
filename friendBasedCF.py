from cv2 import computeCorrespondEpilines
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


dataPath = '../dataset/hetrec2011-lastfm-2k/'
df = pd.read_csv(dataPath + 'user_friends.csv')
ratings = pd.read_csv(dataPath + 'user_artists.csv')

graph = dict()

def addLinks(df, graph):
    graph[df['userID'].iloc[0]] = set(df['friendID'])
    return

df.groupby('userID').apply(addLinks, graph = graph)


def get_all_connected_groups(graph):
    already_seen = set()
    result = []
    for node in graph:
        if node not in already_seen:
            connected_group, already_seen = get_connected_group(node, already_seen, graph)
            result.append(connected_group)
    return result


def get_connected_group(node, already_seen, graph):
    result = []
    already_seen.add(node)
    result.append(node)
    nodes = graph[node]
    while nodes:
        node = nodes.pop()
        if node in already_seen:
            continue
        already_seen.add(node)
        result.append(node)
        nodes = nodes | graph[node]
    return result, already_seen


def get_second_degree(graph, userID):
    nodes = graph[userID]
    result = set()
    r = set()
    for node in nodes:
      result.update(graph[node])
      r.update(graph[node])
    # for node in r:
    #   result.update(graph[node])
    return result

#components = get_all_connected_groups(graph)
user = 2
components = [get_second_degree(graph,user)]
userCluster0 = components[0]
# idx = 0
# for com in components:
#     print(idx, len(com))
#     print(com)
#     idx+=1

mat_um = ratings.pivot(index='userID',columns='artistID',values='Weight')
mat_um.fillna(0,inplace=True)

ratings0 = ratings.loc[ratings['userID'].isin(userCluster0)]
mat_um0 = ratings0.pivot(index='userID',columns='artistID',values='Weight')
mat_um0.fillna(0,inplace=True)

cosine_sim = cosine_similarity(mat_um)
cosine_sim = pd.DataFrame(cosine_sim, columns=mat_um.index)
cosine_sim = cosine_sim.set_index(mat_um.index)

cosine_sim0 = cosine_similarity(mat_um0)
cosine_sim0 = pd.DataFrame(cosine_sim0, columns=mat_um0.index)
cosine_sim0 = cosine_sim0.set_index(mat_um0.index)

sim_users = cosine_sim[user].nlargest(11).index.tolist()[1:]
sim_users_values = cosine_sim[user].nlargest(11).tolist()[1:]
sim_users_values = [ '%.2f' % elem for elem in sim_users_values ]
print()
print("For All Users")
print(sim_users)
print(sim_users_values)


sim_users = cosine_sim0[user].nlargest(11).index.tolist()[1:]
sim_users_values = cosine_sim0[user].nlargest(11).tolist()[1:]
sim_users_values = [ '%.2f' % elem for elem in sim_users_values ]
print()
print("For Clustered Users")
print(sim_users)
print(sim_users_values)



allMusic = pd.DataFrame(mat_um.columns.tolist(), columns = ['artistID'])
allMusic['Weight'] = 0

for pos in range(len(sim_users)):
    us = sim_users[pos]
    val = sim_users_values[pos]
    music_heard = ratings.groupby('userID').get_group(us)[['artistID','Weight']]
    music_heard = music_heard[music_heard['artistID'].isin(allMusic['artistID'])]
    music_heard['Weight'] = music_heard['Weight'] * float(val)
    allMusic = pd.concat([allMusic, music_heard]).groupby(['artistID']).sum().reset_index()
  
sim_music = allMusic['Weight'].nlargest(5).index.tolist()
sim_music = allMusic.iloc[sim_music]['artistID'].tolist()
print()
print("Top Movies recommended for this user")
for mov in sim_music:
  print(mov)
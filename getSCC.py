from cv2 import computeCorrespondEpilines
import pandas as pd

dataPath = '../dataset/hetrec2011-lastfm-2k/'
df = pd.read_csv(dataPath + 'user_friends.csv')

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

components = get_all_connected_groups(graph)
#components = [get_second_degree(graph,624)]

idx = 0
for com in components:
    print(idx, len(com))
    print(com)
    idx+=1
import numpy as np
import networkx as nx
import math

# An edgelist is a dictionary of sets. The dictionary key is the vertex and dictionary value is
# a set that contains all of its adjacent vertices.


# edgelist -> networkx graph
# converts an edgelist into an networkx graph
def convert_edgelist_to_nx(edgelist):
    G = nx.Graph()
    for vert in edgelist:
        for neighbor in edgelist[vert]:
            G.add_edge(vert, neighbor)
    return G


# 2 column csv_file -> edgelist
# reads a csv_file and returns the edgelist produced by it
def make_edgelist(my_file):
    with open(my_file, "r") as f:
        header = f.readline() # throw away the header
        my_dict = {}
        for line in f:
            data = line.strip('\n')
            data_l = data.split(',')
            data_l = list(map(int, data_l))
            if data_l[0] in my_dict:
                my_dict[data_l[0]].add(data_l[1])
            else:
                my_dict[data_l[0]] = {data_l[1]}
            if data_l[1] in my_dict:
                my_dict[data_l[1]].add(data_l[0])
            else:
                my_dict[data_l[1]] = {data_l[0]}
    return my_dict

# edgelist + vertex -> int
# given an edgelist and a vertex, determines the degree of that vertex
def get_degree(edgelist, vert):
    return len(edgelist[vert])

# edgelist + vertex -> float
# given an edgelist and a vertex, determines the clustering coefficient of that vertex
def clust_c_v(edgelist, vert):
    edge_count = 0
    num_neighbors = len(edgelist[vert])
    for i in edgelist[vert]: # set of neighbors
        for j in edgelist[i]: # neighbor's neighbors
            if j in edgelist[vert]:
                edge_count += 1
    return edge_count/float((num_neighbors)*(num_neighbors-1))

# edgelist + vertex -> float
# given an edgelist, determines the clustering coefficient of the entire graph
# the clustering coefficient of a graph is the average clustering coefficient of its verts
def clust_c_g(edgelist):
    clust_sum = 0
    for vert in edgelist:
        clust_sum += clust_c_v(edgelist, vert)
    return clust_sum/len(edgelist)

# edgelist + vertex -> float
# given an edgelist and a vertex, determines the closeness centrality of that vertex
def closeness_centrality(edgelist, vert):
    short_sum = 0
    short_paths = nx.shortest_path_length(convert_edgelist_to_nx(edgelist), vert)
    for value in short_paths.values():
        short_sum += value
    return 1.0/short_sum


# edgelist + vertex -> float
# given an edgelist and a vertex, determines the betweenness centrality of that vertex
def betweenness_centrality(edgelist, vert):
    bet_sum = 0.0
    for i, val1 in enumerate(edgelist):
        for j, val2 in enumerate(edgelist):
            if j>i and vert != val1 and vert != val2:
                all_shorts = nx.all_shortest_paths(edgelist, val1, val2)
                vert_in_short_count = 0
                short_count = 0
                for short in all_shorts:
                    short_count += 1
                    if vert in short:
                        vert_in_short_count += 1
                bet_sum += (vert_in_short_count)/short_count
    return bet_sum


# edgelist -> float
# given an edgelist, determines the average shortest path length of the graph
def avg_short_path(edgelist):
    num_nodes= len(edgelist)
    short_sum = 0
    short_paths = nx.shortest_path_length(convert_edgelist_to_nx(edgelist))
    for vert in short_paths:
        for path_l in vert[1].values(): #its a tuple of (vertex, {neighbor : length})
            short_sum += path_l
    return short_sum/float((num_nodes)*(num_nodes-1)) #total_sum/tot_nodes



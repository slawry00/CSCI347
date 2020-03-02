import numpy as np
import math

# An edgelist is a dictionary of sets. The dictionary key is the vertex and dictionary value is
# a set that contains all of its adjacent vertices.


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



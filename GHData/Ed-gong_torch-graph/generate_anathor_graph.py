import dgl
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

def read_graph_data(file_path, line_needed_to_skip):
    crs = open(file_path, "r")
    print ("Output of Read function is ")
    #print (crs.read())
    a = [line.split() for line in crs]
    comment = a[0:line_needed_to_skip]
    array = a[line_needed_to_skip:]
    # convert the edge to src and dst
    src = []
    dst = []
    for each_edge in array:
        src.append(int(each_edge[0]))
        dst.append(int(each_edge[1]))
    return src, dst, comment
       
def write_graph_data(comment, src, dst, saved_file_name):
    crs = open(file_path, "r")
    print ("write the graph data now:")
    #print (crs.read())
    edge = []
    for i in range(len(src)):
        temp = []
        temp.append(str(src[i]))
        temp.append(str(dst[i]))
        edge.append(temp)
    # write this things in a python file
    file = open(saved_file_name, "w") 
    for each_line in comment:
        for each_word in each_line:
        #print(each_line)
            file.write(each_word + " ")
        file.write("\n")
    for each_line in edge:
        for each_word in each_line:
        #print(each_line)
            file.write(each_word + " ")
        file.write("\n")
        #file.write(each_line + "\n")
    file.close() 
   

if __name__ == "__main__":
    file_path = "bigworld.txt"
    line_needed_to_skip = 4
    src, dst, comment = read_graph_data(file_path, line_needed_to_skip)
    new_src = [1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33]
    new_dst = [0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32]
    graph = write_graph_data(comment, new_src, new_dst, "try.txt")


import pickle
from glob import glob
import os
import networkx as nx
from model.min_cut import MinCut
import numpy as np

dirs = glob('../data_temp/test_I/*.p')
dirs.sort(key=os.path.abspath)

graph = pickle.load(open('../data_temp/test_I/graph_cycle_nn9_0000001.p', 'rb'))

J = graph['J'].todense()
G = nx.from_numpy_array(J)

map_mincut = MinCut(G, J, graph['b']).inference()
map_gt = graph['map_gt']
print(np.concatenate((map_gt, map_mincut), 1))



import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx #convert the data from graph geometric to a networkx



#defining a undirected graph with 4 nodes

#defining 3 tensors

edge_list = torch.tensor([
                        [0,0,0,1,2,2,3,3], #set source nodes
                        [1,2,3,0,0,3,2,0], #set target nodes
                        ], dtype=torch.long)

#define 6 features for each node (4*6 (no. of nodes*no. of features))

node_features = torch.tensor([
                            [-8,1,5,8,2,-3 ], #features of node 0
                            [-1,0,2,-3,0,1], #features of node 1
                            [1,-1,0,-1,2,1], #features of node 2
                            [0,1,4,-2,3,4], #features of node 3
                            ], dtype=torch.long)

#define 1 weight for each edge 

edge_weights = torch.tensor([
                            [35.], #weight for nodes (0,1)
                            [48.], #weight for nodes (0,2)
                            [12.], #weight for nodes(0,3)
                            [10.], #weight for nodes(1,0)
                            [70.], #weight for nodes(2,0)
                            [5.], #weight for nodes(2,3)
                            [15.], #weight for nodes(3,2)
                            [8.], #weight for nodes(3,0)
                            ], dtype = torch.long)

#make a data object to store graph information

data = Data(x= node_features, edge_index=edge_list, edge_attr= edge_weights)

#print the graph info

print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features per node (Length of feature vector", data.num_node_features)
print("Number of weights per edge(edge features)", data.num_edge_features, "\n")

#plot the graph

G = to_networkx(data)
nx.draw_networkx(G)
plt.show()










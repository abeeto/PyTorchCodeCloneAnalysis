import torch
from torch_geometric.data import Data
import torch_geometric

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# torch_geometric.data utility
print(data.keys)
print(data['x'])

for key, item in data:
    print("{} found in data".format(key))

# check number of nodes
print(data.num_nodes)

# check number of edges
print(data.num_edges)

# check number of node features
print(data.num_node_features)

# check if isolated nodes
print(data.contains_isolated_nodes())

# check if self loops edges
print(data.contains_self_loops())

# check if the graph is directed or not
print(data.is_directed())

# Transfer data object to CUDA
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

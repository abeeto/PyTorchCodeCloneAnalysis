import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, GCNConv


class RandomWireGCN(torch.nn.Module):
    def __init__(self, num_layers: int, features: int, p: float):
        super(RandomWireGCN, self).__init__()
        self.channels = features
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(self.channels, self.channels))
        self.n = len(self.layers)
        self.p = p
        self.architecture, self.sorting = self.create_er(self.n, self.p)

        self.in_nodes = [node for node in self.architecture.nodes if
                         self.architecture.in_degree(node) == 0 and node != ""]
        self.out_nodes = [node for node in self.architecture.nodes if
                          self.architecture.out_degree(node) == 0 and node != ""]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        layer_result = [torch.zeros(len(x), self.channels)] * self.n

        for in_node in self.in_nodes:
            layer_result[in_node] += self.architecture.nodes[in_node]["layer"](x, edge_index).relu() / len(
                self.in_nodes)

        for i in self.sorting:
            tmp = torch.zeros(len(x), self.channels)
            incoming = self.architecture.in_edges(i, data=True)
            for u, v, data in incoming:
                tmp += layer_result[u]
            layer_result[i] += self.architecture.nodes[i]["layer"](tmp / len(incoming), edge_index).relu()

        output = torch.zeros(len(x), self.channels)
        for out_node in self.out_nodes:
            output += self.architecture.nodes[out_node]["layer"](x, edge_index).relu() / len(self.out_nodes)

        return output

    def create_er(self, n: int, p: float):
        G = nx.gnp_random_graph(n, p, directed=True)
        DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
        for node, label in DAG.nodes(data=True):
            DAG.nodes[node]["layer"] = self.layers[node]

        return DAG, list(nx.topological_sort(DAG))


import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def normalized_adj(adj):
    eye = th.sparse_coo_tensor(
        indices=th.stack([th.arange(adj.shape[0]), th.arange(adj.shape[0])], dim=0),
        values=th.ones(adj.shape[0]),
    ).to(adj.device)
    adj = eye + adj
    
    deg = th.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = th.pow(deg, -0.5)
    
    deg_inv_sqrt = th.diag(deg_inv_sqrt).to_sparse()
    
    adj = th.sparse.mm(th.sparse.mm(deg_inv_sqrt, adj), deg_inv_sqrt)

    return adj

class SGC(nn.Module):
    def __init__(self, in_features, out_features, adj, k, bias=True):
        super(SGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = normalized_adj(adj).pow(k)

        self.W = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        support = self.adj.mm(x)
        output = self.W(support)

        return output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_feats, out_feats) -> None:
        super().__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.weight = nn.Parameter(th.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.FloatTensor(out_feats))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = th.mm(x, self.weight)
        output = th.spmm(adj, support)

        return output + self.bias

class GCN(nn.Module):
    """
        Graph Convolutional Network (GCN) https://arxiv.org/abs/1609.02907
    """
    def __init__(self, nfeats, nhids, nclasses, adj) -> None:
        super().__init__()

        self.adj = normalized_adj(adj)

        self.conv1 = GraphConvolution(nfeats, nhids)
        self.conv2 = GraphConvolution(nhids, nclasses)

    def forward(self, x):
        x = F.relu(self.conv1(x, self.adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, self.adj)

        return x

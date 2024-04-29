import torch
import math
import scipy
import random
import numpy as np
import torch_scatter
from collections import Counter
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch_geometric
from torch_geometric.datasets import Planetoid
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse , to_dense_adj
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data

def attacker(dataset, data , ptb):
    #0 < ptb < 1 {0.05 se start}
    dpr_data = Pyg2Dpr(dataset)
    attacked_data = data
    adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
    idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    #surrogate setup
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device)
    model = model.to(device)
    perturbations = int(ptb * (adj.sum() // 2))
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    edges_src = (torch.nonzero( modified_adj, as_tuple=True))[0].int()
    edges_dst = (torch.nonzero( modified_adj, as_tuple=True))[1].int()
    edge_index_corsen = torch.stack((edges_src, edges_dst))
    attacked_data.edge_index = edge_index_corsen
    attacked_data.edge_attr = torch.reshape(modified_adj[torch.nonzero(modified_adj,as_tuple=True)], (-1,1))
    return attacked_data


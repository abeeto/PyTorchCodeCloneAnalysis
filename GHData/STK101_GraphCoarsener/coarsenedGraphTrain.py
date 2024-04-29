
import math
import numpy as np

import random
from collections import Counter
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from torch_geometric.transforms import NormalizeFeatures

import time 
import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GCN_(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def train(model,data_coarsen,optimizer,criterion):
      model.train()
      optimizer.zero_grad() 
      out = model(data_coarsen.x, data_coarsen.edge_index, data_coarsen.edge_attr)  
      loss = criterion(out, data_coarsen.y)   #loss = criterion(out[data_coarsen.train_mask], data_coarsen.y[data_coarsen.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test(model,data):
      model.eval()
      out = model(data.x, data.edge_index, data.edge_attr)
      pred = out.argmax(dim=1)  
      test_correct = pred == data.y  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

def HashFunction(fea,Wl,bin_width, bias):
    h = math.floor((1/bin_width)*((np.dot(fea,Wl)) + bias))
    return h

def clustering(dataset,data,no_of_hash, bin_width):
  feature_size = dataset.num_features
  Wl = np.random.uniform(feature_size, size= (no_of_hash,feature_size))
  no_nodes = data.x.shape[0]
  features = data.x
  bias = [random.uniform(-bin_width,bin_width) for i in range(no_of_hash)]
  dict_hash_indices = {}
  for i in range(no_nodes):
    fea = features[i,:]
    list_ = []
    for _ in range(no_of_hash):
      list_.append(HashFunction(fea,Wl[_,:]/np.linalg.norm(Wl[_,:]),bin_width, bias[_]))  
    occurence_count = Counter(list_)
    dict_hash_indices[i] = occurence_count.most_common(1)[0][0]
  return dict_hash_indices

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def run(dataset,data,bw,name_str,epochs=100):
  logger = logging.getLogger(__name__)
  print("Starting method on " , name_str)
  logger.info(f'Starting method on : {name_str}')
  
  torch.cuda.empty_cache()
  g_coarsened = clustering(dataset,data.to('cpu'),100,bw)
  values = g_coarsened.values() 
  unique_values = set(g_coarsened.values())
  print("reduction ratio - ", (1 - (len(unique_values)/len(values))))
  logger.info(f'reduction ratio -  {(1 - (len(unique_values)/len(values)))}')

  dict_blabla ={}
  C_diag = torch.zeros(len(unique_values), device= device)
  help_count = 0
  for v in unique_values:
      C_diag[help_count],dict_blabla[help_count] = get_key(v, g_coarsened)
      help_count += 1
  val = dict_blabla.values()
  C_diag_lab = torch.clone(C_diag).to(device)
  P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)
  P_hat_lab = torch.clone(P_hat).to(device)
  zero_list = []
  for x in dict_blabla:
      for y in dict_blabla[x]:
          P_hat[y,x] = 1
          if ((data.train_mask)[y] == False):
            C_diag_lab[x] = C_diag_lab[x] - 1
            if (C_diag_lab[x] == 0):
              zero_list.append(x)
              C_diag_lab[x] = 1
          else:
            P_hat_lab[y,x] = 1
  zero_list = sorted(zero_list)

  P_hat_lab = P_hat_lab.to_sparse()
  P_hat = P_hat.to_sparse()
  P_lab = torch.sparse.mm(P_hat_lab,(torch.diag(torch.pow(C_diag_lab, -1/2))))
  P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))
  g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)
  g_adj =torch.squeeze(g_adj)

  i = dense_to_sparse(g_adj)[0]
  v = dense_to_sparse(g_adj)[1]
  shape = g_adj.shape
  g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
  g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
  g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy()
  edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]

  edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
  edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
  edge_index_corsen = torch.stack((edges_src, edges_dst))
  edge_features = torch.from_numpy(edge_weight)
  features =  data.x.to(device = device)
  cor_feat = (torch.sparse.mm(torch.t(P), features)).to('cpu')
  cor_feat_dict = {}
  for x in range(len(cor_feat)):
      cor_feat_dict[x] = cor_feat[x]

  X = np.array(data.y.cpu())
  n_X = np.max(X) + 1
  X = np.eye(n_X)[X]
  X = torch.from_numpy(X).to(device)
  labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P_lab).double() , X.double()).double() , 1).to('cpu')
  for x in zero_list:
    if (x == 0):
      labels_coarse[x] = 0 #labels_coarse[Next_greater(zero_list, x)]
    else:
      labels_coarse[x] = labels_coarse[x-1]

  data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
  data_coarsen.edge_attr = edge_features
  model = GCN_(dataset.num_features, 16, dataset.num_classes)
  print(model)

  model = model.to(device)
  data_coarsen = data_coarsen.to(device)
  train_mask = torch.ones( len(cor_feat), dtype=torch.bool)
  train_mask[zero_list] = False
  learning_rate = 0.01
  decay = 5e-4
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()
  start = time.time()
  for epoch in range(400):
        optimizer.zero_grad()
        out = model(data_coarsen.x, data_coarsen.edge_index, data_coarsen.edge_attr)  
        pred = out.argmax(1)
        criterion = torch.nn.NLLLoss()
        # loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss = criterion(out[train_mask], data_coarsen.y[train_mask])
        

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('In epoch {}, loss: {:.3f}'.format(epoch, loss))
        if epoch % 50 == 0:
          logger.info('In epoch {}, loss: {:.3f}'.format(epoch, loss))

  print("Time taken to train: " , start - time.time())
  logger.info(f'Time taken to train: " , {start - time.time()}')

  data = data.to(device)
  model.eval()
  pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  print(f'Accuracy: {acc:.4f}')
  logger.info(f'Accuracy: {acc:.4f}')

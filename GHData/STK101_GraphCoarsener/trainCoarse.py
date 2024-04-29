import math
import numpy as np
import random
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import time
import logging
import pickle
import learn_graph
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

def HashFunction(fea,Wl,bin_width, bias):
  h = math.floor((1/bin_width)*((np.dot(fea,Wl)) + bias))
  return h


def clustering(data, no_of_hash, bin_width, dataset):
  data = data.to(device)
  feature_size = dataset.num_features
  Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1).to(device)
  no_nodes = data.x.shape[0]
  features = data.x
  bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)]).to(device)
  features.to(device)
  Bin_values = torch.floor((1/bin_width)*(torch.matmul(features, Wl.T) + bias)).to(device)
  cluster, _ = torch.max(Bin_values, dim = 1)
  dict_hash_indices = {}
  for i in range(no_nodes):
    dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
  return dict_hash_indices

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def val(model,data):
    data = data.to(device)
    model.eval()
    pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    correct = (pred == data.y).sum()
    acc = int(correct) / int(data.x.shape[0])
    return acc

def run(dataset,data,bw,name_str,epochs=300, learning_rate = 0.01 , decay= 5e-04, gsp = False, alpha = 1, beta = 1):

  #print("Starting method on " , name_str)

  torch.cuda.empty_cache()
  g_coarsened = clustering(data,100,bw,dataset) #(data, no_of_hash, bin_width, dataset)
  values = g_coarsened.values() 
  unique_values = set(g_coarsened.values())
  rr = 1 - len(unique_values)/len(values)
  dict_blabla ={}
  C_diag = torch.zeros(len(unique_values), device= device)
  help_count = 0
  for v in unique_values:
      C_diag[help_count],dict_blabla[help_count] = get_key(v, g_coarsened)
      help_count += 1


  P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)
  zero_list = torch.ones(len(unique_values), dtype=torch.bool)
  for x in dict_blabla:
      for y in dict_blabla[x]:
          P_hat[y,x] = 1
          zero_list[x] = zero_list[x] and (not (data.train_mask)[y]) 

  P_hat = P_hat.to_sparse()
  P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))

  features =  data.x.to(device = device).to_sparse()
  cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense())).to_sparse()

  g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)
  g_adj =torch.squeeze(g_adj)

  i = dense_to_sparse(g_adj)[0]
  v = dense_to_sparse(g_adj)[1]
  shape = g_adj.shape
  g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
  g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
  C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))
  g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32) 
  if (gsp):
    cor_feat_dense = cor_feat.to_dense().to('cpu').numpy()
    g_coarse_dense = learn_graph.validation_reg(cor_feat_dense, g_coarse_dense, alpha = alpha, beta= beta).astype(np.float32)
    #g_coarse_dense = learn_graph.make_sparse()
  cor_feat = cor_feat.to('cpu')
  edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
  edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
  edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
  edge_index_corsen = torch.stack((edges_src, edges_dst))
  edge_features = torch.from_numpy(edge_weight)
  num_classes = dataset.num_classes

  Y = np.array(data.y.cpu())
  Y = one_hot(Y,dataset.num_classes).to(device)
  Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)]).to(device)
  labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1).to(device)
  data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
  data_coarsen.edge_attr = edge_features

  model = GCN_(dataset.num_features, 16, dataset.num_classes)
  model = model.to(device)
  data_coarsen = data_coarsen.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=decay)

  best_val_acc = 0
  for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_coarsen.x, data_coarsen.edge_index, data_coarsen.edge_attr)  
        pred = out.argmax(1)
        criterion = torch.nn.NLLLoss()
        loss = criterion(out[~zero_list], data_coarsen.y[~zero_list]) 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        val_acc = val(model,data)
        if best_val_acc < val_acc:
            torch.save(model, 'best_model.pt')
            best_val_acc = val_acc
            

        #if epoch % 20 == 0:
            #print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))
        
  model = torch.load('best_model.pt')
  model.eval()
  data = data.to(device)
  pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  correct = (pred == data.y).sum()
  acc = int(correct) / int(data.x.shape[0])
  #print(f'Accuracy: {acc:.4f}')
  return [acc , rr]






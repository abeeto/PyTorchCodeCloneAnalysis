import torch_geometric
import math
import numpy as np
import scipy
import random
from collections import Counter
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data


torch.cuda.empty_cache()
dataset = Reddit(root='data/Reddit', transform=NormalizeFeatures())

data = dataset[0]  
print(data)
print(data.edge_index.T)
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.out(x), dim=1)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
print(model)



model = GCN(num_features = dataset.num_features, hidden_channels=16, num_classes = dataset.num_classes)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      out = model(data.x, data.edge_index)  
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []
for epoch in range(0, 20):
    loss = train()
    losses.append(loss)
    if epoch % 5 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

original_test_acc = test()

def HashFunction(fea,Wl,bin_width, bias):
  h = math.floor((1/bin_width)*((np.dot(fea,Wl)) + bias))
  return h

def clustering(data,no_of_hash, bin_width):
  feature_size = dataset.num_features
  Wl = np.random.uniform(feature_size, size= (no_of_hash,feature_size))
  no_nodes = data.x.shape[0]
  features = data.x
  #bin_width = 45 #for coauthor cs
  #bin_width = 15 #for coauthor physics half
  #bin_width = 60 # for coauthor physics actual
  #bin_width = 0.25 #for citeseer
  #bin_width = 1 #for cora
  #bin_width =  2.562999725341797e-06  #for Pubmed
  #bin_width = 2 #for CoraFull
  bias = [random.uniform(-bin_width,bin_width) for i in range(no_of_hash)]
  dict_hash_indices = {}
  for i in range(no_nodes):
    #fea = np.append(g.ndata['feat'][i]), np.array([degree_mat[i,0], eigen_centrality[i,0]]))
    fea = features[i,:]
    list_ = []
    for _ in range(no_of_hash):
      #list_.append(HashFunction(fea,Wl[_,:],20))
      list_.append(HashFunction(fea,Wl[_,:]/np.linalg.norm(Wl[_,:]),bin_width, bias[_]))
      
    occurence_count = Counter(list_)
    dict_hash_indices[i] = occurence_count.most_common(1)[0][0]
  return dict_hash_indices

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS)

def get_key_subject(val,  g_coarsened):
  KEYS_SUBJECT = []
  for key, value in g_coarsened.items():
    if val == value:
      #KEYS_SUBJECT.append(g.ndata["label"][key])
      KEYS_SUBJECT.append(key)
  return KEYS_SUBJECT


bw = .562999725341797e-06
g_coarsened = clustering(data.to('cpu'),100,bw)
values = g_coarsened.values() 
unique_values = set(g_coarsened.values())
print("reduction ratio - ", (1 - (len(unique_values)/len(values))))
dict_blabla ={}
C_diag = torch.zeros(len(unique_values), device= device)
help_count = 0
for v in unique_values:
    C_diag[help_count] = get_key(v, g_coarsened)
    dict_blabla[help_count] = get_key_subject(v, g_coarsened)
    help_count += 1
val = dict_blabla.values()

P_hat = torch.zeros((data.num_nodes, len(unique_values)), device= device)
for x in dict_blabla:
  for y in dict_blabla[x]:
    P_hat[y,x] = 1

P_hat = P_hat.to_sparse()
P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))
g_adj = to_dense_adj(data.edge_index)
g_adj =torch.squeeze(g_adj)
print(g_adj.shape)
i = dense_to_sparse(g_adj)[0]
v = dense_to_sparse(g_adj)[1]
shape = g_adj.shape
g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device = device)
g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy()
edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
print(g_coarse_dense.shape)
edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
edge_index_corsen = torch.stack((edges_src, edges_dst))
edge_features = torch.from_numpy(edge_weight)
n_nodes = len(unique_values)
n_train = int(n_nodes * 0.6)
n_val = int(n_nodes * 0.2)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train:n_train + n_val] = True
test_mask[n_train + n_val:] = True
features =  data.x.to(device = device)
cor_feat = (torch.sparse.mm(torch.t(P_hat), features)).to('cpu')
cor_feat_dict = {}
for x in range(len(cor_feat)):
    cor_feat_dict[x] = cor_feat[x]
X = np.array(data.y.cpu())
n_X = np.max(X) + 1
X = np.eye(n_X)[X]
X = torch.from_numpy(X).to(device)
labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , X.double()).double() , 1).to('cpu')

data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
data_coarsen.train_mask = train_mask
data_coarsen.test_mask = test_mask
data_coarsen.val_mask = val_mask
data_coarsen.edge_attr = edge_features

class GCN_(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,num_classes):
        super(GCN_, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.out(x), dim=1)
        return x

model = GCN_(dataset.num_features, 16, dataset.num_classes)
print(model)

model = GCN_(num_features = dataset.num_features, hidden_channels=16, num_classes = dataset.num_classes)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_coarsen = data_coarsen.to(device)

learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      out = model(data_coarsen.x, data_coarsen.edge_index, data_coarsen.edge_attr)  
      loss = criterion(out[data_coarsen.train_mask], data_coarsen.y[data_coarsen.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index, data.edge_attr)
      pred = out.argmax(dim=1)  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []
for epoch in range(0, 200):
    loss = train()
    losses.append(loss)
    if epoch % 20 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print("Final Test Acc")
print(test_acc)
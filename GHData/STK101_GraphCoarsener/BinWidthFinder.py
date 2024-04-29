import numpy as np
import random
import torch_geometric
import math
import numpy as np
import scipy
import random
from collections import Counter
import torch
from torch.nn import Linear
import argparse
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import GitHub
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Planetoid
import sys
import logging
import os

from zmq import device
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def HashFunction(fea,Wl,bin_width, bias):
  h = math.floor((1/bin_width)*((np.dot(fea,Wl)) + bias))
  return h

def clustering_vectorized(data, no_of_hash, bin_width, dataset):
  feature_size = dataset.num_features
  Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1).to(device)
  # print(Wl.shape)
  no_nodes = data.x.shape[0]
  features = data.x
  bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)]).to(device)
  # print(bias.shape)
  features.to(device)
  # print(features.shape)
  Bin_values = torch.floor((1/bin_width)*(torch.matmul(features, Wl.T) + bias))
  cluster, _ = torch.max(Bin_values, dim = 1)
  dict_hash_indices = {}
  for i in range(no_nodes):
    dict_hash_indices[i] = cluster[i]
  return dict_hash_indices



def clustering(data,no_of_hash, bin_width, dataset):
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


def Find_Binwidth(dataset, data,coarsening_ratio = 0.05, precision = 0.0005):
  bw = 1
  ratio = 1
  while(abs(ratio - coarsening_ratio) > precision):
    if(ratio > coarsening_ratio):
      bw = bw*0.5
    else:
      bw = bw*1.5
    #print(bw)
    g_coarsened = clustering_vectorized(data.to(device),100, bw,dataset)
    values = g_coarsened.values() 
    unique_values = set(g_coarsened.values())
    ratio = (1 - (len(unique_values)/len(values)))
    # break
  return bw, ratio



def parse_args():
    parser = argparse.ArgumentParser(description='Find Bin Width')
    parser.add_argument('--data_dir',type=str,required=False,help="Path to the dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # args = parse_args()
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('binwidthCoraFull.log', 'a'))
    print = logger.info
    print("hello world")
    print("Loading Dataset Planetoid Cora")
    dataset = Planetoid(root = 'data/PlanetoidCora', name = 'Cora')
    data = dataset[0]
    print(data) 
    bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    print(bw)
    print(ratio)
    # print("Loading Dataset DBLP")
    # dataset = CitationFull(root = 'data/DBLP', name = 'DBLP')
    # data = dataset[0] 
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print()
    # print()

    # print("Loading Dataset Amazon")
    # dataset = Amazon(root = 'data/Amazon_computers', name = 'Computers')
    # data = dataset[0]  
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print(" ")
    # print(" ")

    # print("Loading Dataset Amazon Photo")
    # dataset = Amazon(root = 'data/Amazon_Photo', name = 'Photo')
    # data = dataset[0]
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print(" ")
    # print(" ")

    # print("Loading Dataset Reddit2")
    # dataset = Reddit2(root='data/Reddit2')
    # data = dataset[0]
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print()
    # print()


    # print("Loading Dataset Reddit")
    # dataset = Reddit(root='data/Reddit')
    # data = dataset[0]
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print()
    # print()


    # print("Loading Dataset Citation Cora")
    # dataset = CitationFull(root = 'data/CitationCora', name = 'Cora')
    # data = dataset[0]
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print()
    # print()

    # print("Loading Dataset Flickr")
    # dataset = Flickr(root='data/Flickr')
    # data = dataset[0]
    # print(data) 
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.5, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.7, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)
    # bw, ratio = Find_Binwidth(coarsening_ratio = 0.3, precision = 0.01,dataset = dataset, data= data)
    # print(bw)
    # print(ratio)

    # del data
    # del dataset
    # print()
    # print()c

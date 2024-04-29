# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:39:59 2019

@author: admin
"""
import numpy as np
import gnnio,preprocess
import os,time
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys



def write_result(*args,filename='result.txt'):
    cwd=os.getcwd()
    writepath=os.path.join(cwd,'results',filename)
    #print(writepath)
    file=open(writepath,'a')
    file.write('\n\n'+'*'*10 + str(time.asctime(time.localtime())))
    for item in args:
        file.write(str(item))
    file.close()

def get_dataset(dataset_str):
    if dataset_str in ['cora','citeseer','pubmed']:
        data=get_citation_data(dataset_str)
        #print(data)
        #print(data)
        return data
    elif dataset_str=='reddit':
        raise IOError('Not implemented')
    



def countable(a):
    try:
        b=a+1
        return True
    except:
        return False


#to record exp statistics
class statistic_recorder():
    
    def __init__(self):
        
        self.data={}
        self.statistic={}
        
    def insert(self,thing):
        #'thing' must be a dict with countable values
        for key in thing:
            if countable(thing[key]):
                if key not in self.data:
                    self.data[key]=[]
                self.data[key].append(thing[key])    
        self.update()
        
    def update(self):
        
        for key in self.data:
            self.statistic[key]={}
            self.statistic[key]['Times']=len(self.data[key])
            self.statistic[key]['Sum']=sum(self.data[key])
            self.statistic[key]['Mean']=self.statistic[key]['Sum']/self.statistic[key]['Times']
            self.statistic[key]['Deviation']=np.sqrt(sum([(x-self.statistic[key]['Mean'])**2 for x in 
                          self.data[key]]))
        
        return self.statistic
    
def get_citation_data(dataset_str):
    
    
    
    data_name='./data/{}.npz'.format(dataset_str)
    #data_name='./data/'+dataset_str+'.npz'
    '''data_graph=gnnio.load_npz_to_sparse_graph(data_name)
    data_graph.adj_matrix = preprocess.eliminate_self_loops(data_graph.adj_matrix)
    data_graph =data_graph.to_undirected()
    data_graph =data_graph.to_unweighted()
    
    if dataset_str == 'cora_full':
        # cora_full has some classes that have very few instances. We have to remove these in order for
        # split generation not to fail
        data_graph = preprocess.remove_underrepresented_classes(data_graph,
                                                        30, 30)
        data_graph= data_graph.standardize()
    
    
    labels=data_graph.labels
    adj=data_graph.adj_matrix
    features=data_graph.attr_matrix
    
    try:
        class_number=len(data_graph.class_names)
    except:
        class_number=max(labels)-min(labels)+1
    #将labels变成onehot的
    labels_onehot=[]
    for i in labels:
        temp_vector=[]
        for j in range(class_number):
            if j==i:
                temp_vector.append(1)
            else:
                temp_vector.append(0)
        labels_onehot.append(temp_vector)
    labels=np.array(labels_onehot)'''
    
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    #receive these data
    data_dict={}
    
    #print(features.dtype)
    
    data_dict['feature']=features.astype(np.float)
    data_dict['label']=labels
    data_dict['adjacent matrix']=adj
    return data_dict

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
    
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)    
    
    

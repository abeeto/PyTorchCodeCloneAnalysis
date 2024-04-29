
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gnnio,preprocess
import os,time
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from data_process import *

class base_model():
    
    def __init__(self,model_input):
        
        self.input=model_input
        self.output={}   
        self.trainable_parameter=[]
    
    def run(self):
        raise IOError('Not implemented!')
    
    
    
class GCN_kipf(base_model):

    def __init__(self,model_input):
        super(GCN_kipf,self).__init__(model_input)
        #define the data and parameters
        
        #A must be sparse
        self.A=model_input['adjacent matrix']
        
        '''if self.A.is_sparse:
            pass
        else:
            self.A=self.A.to_sparse()'''
        #X must also be sparse
        self.X=model_input['feature']
        '''if self.X.is_sparse:
            pass
        else:
            self.X=self.X.to_sparse()'''
        self.Y=model_input['label']
        self.settings={'hidden1':16,'learning_rate':0.01 ,'weight_decay':5e-4 ,
                       'training_epoch':200,'dataset':'cora','early_stopping':10,
                       'dropout':0.5,
                       'all_test':False}
        self.settings.update(model_input['settings'])
        #print(self.settings)
        self.output['settings']=model_input['settings']
        
    
    
    def inference(self,laplacian,features,dropout=True):
        
        
        #print(laplacian.shape,features.shape,self.W1.shape)
        hidden_layer_1 = F.relu(torch.sparse.mm(laplacian,torch.sparse.mm(features,self.W1))) 
        if dropout:
            hidden_layer_1 =F.dropout(hidden_layer_1,p=self.settings['dropout'])/self.settings['dropout']
            output = torch.sparse.mm(laplacian, hidden_layer_1.mm(self.W2) )
            output=F.dropout(output,p=self.settings['dropout'])/self.settings['dropout']
        else:
            output = torch.sparse.mm(laplacian, hidden_layer_1.mm(self.W2) )
        return output
        
    def run(self):
        print('starting GCN kipf version on standard split, settings are {}'.format(self.settings))
        A_bar= self.preprocess_edge()
        FloatTensor = torch.FloatTensor
        # Turn the input and output into FloatTensors for the Neural Network
        sparse_x=self.preprocess_feature()
        #print(type(sparse_x.indices[0]))
        x = Variable(torch.sparse.FloatTensor(torch.LongTensor([sparse_x.row.astype(int),
                                                                sparse_x.col.astype(int)]),
                                              torch.from_numpy(sparse_x.data),
                                              sparse_x.shape), requires_grad=False)
        #print(str(x))
        y = Variable(FloatTensor(self.Y).double(), requires_grad=False)
        
        self.number,self.input_dim=x.shape
        _,self.output_dim=y.shape
        
        
        A_bar = Variable(torch.sparse.FloatTensor(torch.LongTensor([A_bar.row.astype(int),
                                                                A_bar.col.astype(int)]),
                                              torch.from_numpy(A_bar.data),
                                              A_bar.shape), requires_grad=False)
        #print(str(A_bar))
        # Create random tensor weights
        self.W1 = Variable(torch.Tensor(self.input_dim, self.settings['hidden1']).double(), requires_grad=True)
        self.W2 = Variable(torch.Tensor(self.settings['hidden1'], self.output_dim).double(), requires_grad=True)
        
        #print(self.W1.type(),self.W2.type())
        torch.nn.init.xavier_uniform(self.W1)
        torch.nn.init.xavier_uniform(self.W2)
        self.trainable_parameter.append(self.W1)
        self.trainable_parameter.append(self.W2)
        
        optimizer = torch.optim.Adam(self.trainable_parameter, lr=self.settings['learning_rate'],
                                     weight_decay=self.settings['weight_decay'])
        
        #construct mask
        mask_train,mask_valid,mask_test=self.get_mask_standard()
        #construct early stopping list
        val_acc_list=[]
        
        for t in range(self.settings['training_epoch']):
            
            
            output = self.inference(A_bar,x,dropout=True)
            
            #print(output.type(),y.type())
            self.losses = torch.add(torch.neg(torch.sum(output.mul(y),dim=1)),torch.log(torch.sum(torch.exp(output),dim=1)))
            #print(self.losses)
            
            '''print(torch.sum(mask_train))
            print(torch.sum(mask_valid))
            print(torch.sum(mask_test))'''
            #masked
            #print(self.losses.shape,mask_train.shape)
            loss=torch.sum(self.losses.mul(mask_train))
            
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            with torch.no_grad():
                
                output = self.inference(A_bar,x,dropout=False)
                prediction=torch.argmax(output,dim=1)
                true_label=torch.argmax(y,dim=1)
                '''print(prediction,true_label)
                print(torch.eq(prediction,true_label),mask_train.byte())
                print(torch.dot(torch.eq(prediction,true_label),mask_train.byte()))
                print(torch.sum(torch.dot(torch.eq(prediction,true_label),mask_train.byte())))'''
                accuracy_train=np.sum(np.dot(np.equal(prediction,true_label),mask_train))/torch.sum(mask_train).item()
                accuracy_val=np.sum(np.dot(np.equal(prediction,true_label),mask_valid))/torch.sum(mask_valid).item()
                accuracy_test=np.sum(np.dot(np.equal(prediction,true_label),mask_test))/torch.sum(mask_test).item()
                
                if len(val_acc_list)<=self.settings['early_stopping']:
                    val_acc_list.append(accuracy_val)
                #early stopping
                if accuracy_val<np.mean(val_acc_list) and len(val_acc_list)>=self.settings['early_stopping']:
                    print('Early Stopping--------------------------------')
                    break
                #print(torch.sum(torch.eq(prediction,true_label)).float()/self.number)         
                print('Epoch:{} Training Acc:{} Loss:{} Val Acc:{} Test Acc:{}'.format(t,accuracy_train,loss.item(),
                      accuracy_val,accuracy_test))
            # Manually zero the gradients after updating weights
            #W1.grad.data.zero_()
            #W2.grad.data.zero_()
            
            
            
        self.output.update({'final_loss':loss.item(),'final_accuracy':accuracy_test})
        return self.output
        
    def get_mask_standard(self):
        dataset_str=self.settings['dataset']
        
        
        
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
    
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
        mask_train = sample_mask(idx_train, labels.shape[0]).astype(np.float)
        mask_valid = sample_mask(idx_val, labels.shape[0]).astype(np.float)
        mask_test = sample_mask(idx_test, labels.shape[0]).astype(np.float)
        
        #follow kipf split
        '''if 'train_size' in self.settings:
            train_len=self.settings['train_size']
        else:
            if dataset=='citeseer':
                train_len=120
            elif dataset=='cora':
                train_len=140
            elif dataset=='pubmed':
                train_len=60
        idx_train = list(range(train_len))
        idx_valid = list(range(train_len, train_len+500))
        
        if self.settings['all_test']==True:
            #remaining nodes are all in test set
            idx_test = list(set(range(self.number))-set(idx_train)-set(idx_valid))
        else:
            idx_test = list(range(self.number-1000,self.number))
            
        mask_train,mask_valid,mask_test=np.zeros(self.number),np.zeros(self.number),np.zeros(self.number)
        
        for idx,mask in [(idx_train,mask_train),(idx_valid,mask_valid),(idx_test,mask_test)]:
            for i in idx:
                mask[i]=1
            mask=torch.from_numpy(mask)
            mask=mask.float()
            mask=Variable(mask, requires_grad=False)'''
            
        
        
        
        
        return torch.DoubleTensor(mask_train),torch.DoubleTensor(mask_valid),torch.DoubleTensor(mask_test)

    def preprocess_edge(self):
        adj =self.A.tocoo()
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj=adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return adj
        
        
    def preprocess_feature(self):
        rowsum = np.array(self.X.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        normalized_X = r_mat_inv.dot(self.X).tocoo()
        #print(normalized_X.__class__)
        #print('*'+str(dir(normalized_X)))
        return normalized_X
        
        
        























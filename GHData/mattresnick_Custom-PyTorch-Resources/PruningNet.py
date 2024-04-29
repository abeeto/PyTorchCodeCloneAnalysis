'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import copy
import random

from ActivationFunctions import createActivationLayers, Channelout
from DropoutLayers import createDropoutLayers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



class PruningNet(nn.Module):
    '''
    Class for a master network, with any given activation function, 
    as well as a pruning function.
    
    Parameters:
        - num_features: (integer) Number of input features.
        - num_classes: (integer) Number of classes for prediction. 1 for binary
          class prediction or regression.
        - dropout_p: (float) Probability of dropout per node.
        - DO_type: (string) Dropout type. Either 'alpha', 'shift', or standard dropout.
        - function_name: (string) Activation function name.
        - use_bn: (boolean) Determines whether or not to use batch norm layers.
        - pool_size: (integer) Pooling size for maxout and channelout functions.
    '''
    
    def __init__(self,num_features,num_classes,dropout_p=0.5, DO_type='alpha',
                 function_name='selu', use_bn=False, pool_size=0):
        super(PruningNet, self).__init__()
        
        self.use_bn = use_bn
        self.pool_size = pool_size
        self.num_nonzero = []
        
        if function_name=='maxout':
            hid_nodes = [1200,1200]
            # Construct architecture for maxout based on desired input nodes per hidden layer.
            self.layer_params = [[num_features,hid_nodes[0]]]
            for i,h in enumerate(hid_nodes):
                if i==len(hid_nodes)-1:
                    self.layer_params.append([h//self.pool_size, 2])
                else:
                    self.layer_params.append([h//self.pool_size, hid_nodes[i+1]])
        else:
            # Create DNN layers based on given architecture sizes.
            self.layer_params = [[num_features,784],[784,784],[784,num_classes]]
            
        self.layers = nn.ModuleList([nn.Linear(
                i[0], i[1]) \
                for i in self.layer_params])
        
        # Initialize masks for unstructured pruning.
        self.layer_masks = [np.ones(l.weight.shape) for l in self.layers[1:]]  
        
        # Initialize Batchnorm layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(i[1]) \
                for i in self.layer_params[:-1]])
        
        # Initialize dropout layers given type and prob(s)
        self.DO_layers = createDropoutLayers(DO_type,len(self.layer_params)-1,dropout_p)
        
        # Determine layer activations
        self.activations = createActivationLayers(function_name,len(self.layer_params[:-1]),pool_size)
        self.activations.append(nn.Softmax(dim=1))
    
    def allParameterPrune(self, i, mu, N_iter, pct=0.8):
        '''
        Function for creating a mask which will, in effect, prune all model 
        parameters using the FSA feature selection schedule. Then masks for 
        individual layers will be constructed from the master mask.
        
        Parameters:
            - i: (int) Current time step.
            - mu: (float or int) Tuneable scale constant.
            - N_iter: (int) Number of total time steps.
            - pct: (float) Desired compression scale. Number of non-zero weights after
              full pruning process will be pct*(# of parameters). 0<pct<1.
        
        Returns:
            None (Class variable containing the masks will be modified.)
        '''
        
        lengths = []
        parameters = np.array([1]).reshape((1,1))
        for l, layer in enumerate(self.layers[1:]):
            beta = layer.weight
            a,b = beta.shape[0], beta.shape[1]
            
            # Add current layer parameters to stack of all parameters
            parameters = np.vstack((parameters,toNumpy(beta).reshape((a*b,1))))
            
            # Retain location info for layer mask reconstruction.
            reference = 0
            if l!=0:
                prev_length = lengths[l-1]
                reference = prev_length[0]*prev_length[1] + prev_length[2]
            lengths.append([a, b, reference])
        parameters = parameters[1:]
        
        # Create prune mask from all parameters.
        M = parameters.shape[0]
        k = int(pct*M)
        Mi = int(k+(M-k)*max(0,(N_iter-2*i)/(2*i*mu+N_iter)))
        
        sorter = [[np.linalg.norm(p, ord=1), i] for i,p in enumerate(parameters)]
        sorter.sort(key=lambda x: x[0], reverse=True)
        
        sel_inds = np.array(sorter,dtype='int32')[:Mi,1]
        
        mask = np.zeros(parameters.shape)
        mask[sel_inds] = 1
        
        # Reconstruct masks for each layer's original shape.
        for i,l in enumerate(lengths):
            a,b,r = l
            self.layer_masks[i] = mask[r:r+a*b].reshape((a,b))
            
        return 
    
    def prune(self, i, mu, N_iter, pct=0.8):
        '''
        Function for creating a mask which will, in effect, prune model 
        parameters by layer using the FSA feature selection schedule.
        
        Parameters:
            - i: (int) Current time step.
            - mu: (float or int) Tuneable scale constant.
            - N_iter: (int) Number of total time steps.
            - pct: (float) Desired compression scale. Number of non-zero weights after
              full pruning process will be pct*(layer size). 0<pct<1.
        
        Returns:
            None (Class variable containing the masks will be modified.)
        '''
        nonzero_count = 0
        for l, layer in enumerate(self.layers[1:]):
            beta = layer.weight
            
            a,b = beta.shape[0], beta.shape[1]
            M = a*b
            k = int(pct*M)
            
            # Stack all parameters into one vector, and obtain the M_i number
            # of parameters to be kept.
            beta = toNumpy(beta).reshape((M,1))
            Mi = int(k+(M-k)*max(0,(N_iter-2*i)/(2*i*mu+N_iter)))
            
            # Create norm, index pairs so original indices can be recovered after sorting.
            norms = np.linalg.norm(beta, ord=1, axis=1)
            sorter = list(zip(norms,list(range(len(norms)))))
            
            # Sort by norm value, slice out Mi top indices.
            sorter.sort(key=lambda x: x[0], reverse=True)
            sel_inds = np.array(sorter,dtype='int32')[:Mi,1]
            
            # Create Boolean mask using retreived indices.
            mask = np.zeros(beta.shape)
            mask[sel_inds] = 1
            nonzero_count += np.sum(mask)
            mask = mask.reshape((a,b))
            
            self.layer_masks[l] = mask
        
        self.num_nonzero.append(nonzero_count)
        return
    
    
    def channelPrune(self, pool_size):
        '''
        Groups layers into channels of size pool_size, and then zeros all but
        parameter with the highest magnitude.
        
        Parameters:
            - pool_size: (integer) Size of the channels from which parameters
              are pruned. Determines compression factor by the scale (1/pool_size).
        
        Returns:
            None (Class variable containing the masks will be modified.)
        '''
        nonzero_count=0
        for l, layer in enumerate(self.layers[1:]):
            beta = layer.weight
            
            a,b = beta.shape[0], beta.shape[1]
            M = a*b
            
            beta = beta.view(1,M)
            CO = Channelout(pool_size,True)
            masked_channels = toNumpy(CO(beta)).reshape((M,1))
            
            mask = np.where(masked_channels!=0,1,0)
            nonzero_count += np.sum(mask)
            mask = mask.reshape((a,b))
            
            self.layer_masks[l] = mask
            
        self.num_nonzero = nonzero_count
        return
    
    
    def tetrisPrune(self,bin_size=100):
        '''
        Function for creating a mask which will, in effect, prune model 
        parameters by layer via binning and thresholding.
        
        Parameters:
            - i: (int) Current time step.
            - mu: (float or int) Tuneable scale constant.
            - N_iter: (int) Number of total time steps.
            - pct: (float) Desired compression scale. Number of non-zero weights after
        
        Returns:
            None (Class variable containing the masks will be modified.)
        '''
        
        for l, layer in enumerate(self.layers[2:]):
            beta = layer.weight
            
            a,b = beta.shape[0], beta.shape[1]
            M = a*b
            
            beta = toNumpy(beta).reshape((M,1))
            sorter = [[b, i] for i,b in enumerate(beta)]
            sorter.sort(key=lambda x: x[0], reverse=True)
            sorter = np.array(sorter)
            
            print ('\nBegin tetris')
            
            sorted_vals= list(sorter[:,0])
            
            sel_inds = []
            hist = np.histogram(sorted_vals,bin_size) # Create count of weights in each bin
            lowest_count = min(hist[0]) # Obtain minimum count
            ref = 0
            for i,h in enumerate(hist[0]):
                # For every bin, keep only the amount above the minimum, zero the rest
                target_count = h - (lowest_count-1)
                look_in = list(sorter[ref:ref+h])
                look_in.sort(key=lambda x: np.abs(x[0]), reverse=True)
                look_in = np.array(look_in)
                
                for j in range(target_count):
                    sel_inds.append(look_in[:,1][j])
                ref+=h
            
            mask = np.zeros(beta.shape)
            mask[sel_inds] = 1
            self.num_nonzero = sum(mask)
            mask = mask.reshape((a,b))
            
            self.layer_masks[l+1] = mask
            
        return
    
    
    def forward(self, x):
        '''
        Forward pass per layer:
        input --> mask applied to layer --> layer --> activation
        --> batch norm --> dropout 
        '''
        for l, layer in enumerate(self.layers):
            if l>0:
                l_mask = self.layer_masks[l-1]
                layer.weight = nn.Parameter((layer.weight)*torch.FloatTensor(l_mask).to(device))
            x = layer(x)
            x = self.activations[l](x)
            
            if l<len(self.layers)-1:
                if self.use_bn:
                    x = self.bn_layers[l](x)
                x = self.DO_layers[l](x)
            
        return x


def toNumpy(arr):
    return arr.cpu().detach().numpy()


def trainMultiTarget(net,X,Y,Xtest,Ytest,learning_rate,batch_size,num_ep,
                     l2,mu,prune_during_training,whole_pruning,prune_pct):
    train_score_history,test_score_history=[],[]
    input_length=len(X)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,weight_decay=l2)
    
    for epoch in range(num_ep):
        num_batches = int(input_length/batch_size)
        selections = random.sample(range(0,input_length,batch_size), num_batches)
        
        net.train()
        for i,batch_sel in enumerate(selections):
            # The input size may not be divisible by the batch size
            end_ind=batch_sel+batch_size
            if end_ind>input_length:
                end_ind=input_length
            
            # Cut batches out of data
            input_features = X[batch_sel:end_ind].copy()
            target = Y[batch_sel:end_ind].copy()
            
            # Convert to torch tensors
            input_features = torch.FloatTensor(input_features).to(device)
            target = torch.LongTensor(target).flatten().to(device)
            
            optimizer.zero_grad()
            output = net(input_features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        
        if prune_during_training and whole_pruning:
            net.allParameterPrune(epoch, mu, num_ep,prune_pct)
        elif prune_during_training:
            net.prune(epoch, mu, num_ep,prune_pct)
        
        net.eval()
        # Get scores for current epoch
        train_score = criterion(net(torch.FloatTensor(X).to(device)), torch.LongTensor(Y).flatten().to(device))
        train_score = toNumpy(train_score)
        test_score = criterion(net(torch.FloatTensor(Xtest).to(device)), torch.LongTensor(Ytest).flatten().to(device))
        test_score = toNumpy(test_score)
        
        train_score_history.append(train_score)
        test_score_history.append(test_score)
        
        # Save the best model by test score.
        if test_score==min(test_score_history):
            best_net = copy.deepcopy(net)
        
        print('\rEpoch #'+str(epoch+1)+' error: ' + \
              str(np.abs(round(train_score*100,2)))+'%    ', end='')
        
    score_history = np.hstack((np.array(train_score_history).reshape((num_ep,1)),
                               np.array(test_score_history).reshape((num_ep,1))))
    
    return net, score_history, best_net

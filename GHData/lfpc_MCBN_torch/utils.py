import torch
import numpy as np

def get_MCP(y):
    ''' Returns the Maximum Class/Softmax Probability of a predicted output.
    Returns the value of the probability of the class with more probability'''
    if not is_probabilities(y): #if y is not a probabilities tensor
        y = torch.nn.functional.softmax(y,dim=-1) #apply softmax

    return torch.max(y,-1).values

def MCP_unc(y):
    '''Returns the Maximum Class/Softmax Probability of a predicted output
     as an uncertainty estimation, since MCP is a certainty quantification.
    '''
    return 1-get_MCP(y)

def is_probabilities(y, tol = 1e-5, dim = -1):
    '''Check if tensor y can be considered as a probabilite tensor, i.e., 
    if it sums to 1 (with float tol) and have all values greater than 0'''

    is_prob = torch.logical_and(torch.all(torch.abs((torch.sum(y,dim=dim) - 1)) < tol),
    torch.all(y>0)) 

    return is_prob

def indexing_2D(ar,idx):
    ''' Index a 2D tensor by a 1D tensor along dimension 1.'''
    return ar[np.arange(len(ar)), idx]

def indexing_3D(ar,idx):
    ''' Index a 2D tensor by a 1D tensor along dimension 1.'''
    return ar[:,np.arange(ar.shape[1]),idx]

def entropy(y, reduction = 'none'):
    '''Returns the entropy of a probabilities tensor.'''
    
    if not is_probabilities(y): #if y is not a probabilities tensor
        y = torch.nn.functional.softmax(y,dim=-1) #apply softmax
    
    entropy = torch.special.entr(y) #entropy element wise
    entropy = torch.sum(entropy,-1)
    
    
    if reduction == 'mean':
        entropy = torch.mean(entropy)
    elif reduction == 'sum':
        entropy = torch.sum(entropy)
        
    return entropy


def MonteCarlo_meanvar(MC_array):
    '''Returns the average variance of a tensor'''
    var = torch.var(MC_array, axis=0) 
    var = torch.mean(var,axis= -1)
    return var

def MonteCarlo_maxvar(MC_array, y = None):
    '''Returns the variance of the MCP of a tensor'''
    if y is None:
        y = torch.argmax(torch.mean(MC_array,dim=0),dim = -1)
    var = torch.var(indexing_3D(MC_array,y), axis=0)
    return var

def mutual_info(pred_array):
    '''Returns de Mutual Information (Gal, 2016) of a probability tensor'''
    ent = entropy(torch.mean(pred_array, axis=0))
    MI = ent - torch.mean(entropy(pred_array), axis=0) 
    return MI
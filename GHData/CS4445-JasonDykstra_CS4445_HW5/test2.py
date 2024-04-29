from problem2 import *
import sys
import math
import numpy as np
from model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (5 points) compute_z'''
    x = th.tensor([[1.,6.], # the first sample in the mini-batch
                   [2.,5.], # the second sample in the mini-batch
                   [3.,4.]])# the third sample in the mini-batch
    m = LogisticRegression(2)# create a logistic regression object
    m.layer.weight.data = th.tensor([[-0.1, 0.1]])
    m.layer.bias.data = th.tensor([0.1]) 
    z = compute_z(x,m)
    assert type(z) == th.Tensor 
    z_true = [ [0.6], # linear logit for the first sample in the mini-batch
               [0.4], # linear logit for the second sample in the mini-batch
               [0.2]] # linear logit for the third sample in the mini-batch
    assert np.allclose(z.data,z_true, atol = 1e-2)
    assert z.requires_grad
    L = th.sum(z) # compute the sum of all elements in z
    L.backward() # back propagate gradient to w and b
    assert np.allclose(m.layer.weight.grad,[[6,15]], atol=0.1)
    assert np.allclose(m.layer.bias.grad,[3], atol=0.1)
    n = np.random.randint(2,5) # batch_size 
    p = np.random.randint(2,5) # the number of input features 
    x  = th.randn(n,p)
    m = LogisticRegression(p)# create a logistic regression object
    z = compute_z(x,m) 
    assert np.allclose(z.size(),(n,1))
#---------------------------------------------------
def test_compute_L():
    ''' (5 points) compute_L'''
    z = th.tensor([[ 0.0], # linear logits for the first sample in the mini-batch
                   [ 0.0], # linear logits for the second sample in the mini-batch
                   [ 0.0], # linear logits for the third sample in the mini-batch
                   [ 0.0]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.tensor([[0.],[0.],[0.],[0.]])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.data,0.6931,atol=1e-3) 
    L.backward() # back propagate 
    assert np.allclose(z.grad,[[0.125],[0.125],[0.125],[0.125]], atol=0.1)
    z = th.tensor([[ -1000.0], # linear logits for the first sample in the mini-batch
                   [ -1000.0], # linear logits for the second sample in the mini-batch
                   [ -1000.0], # linear logits for the third sample in the mini-batch
                   [ -1000.0]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.tensor([[0.], [0.], [0.], [0.]])
    L = compute_L(z,y)
    assert np.allclose(L.data,0,atol=1e-3) 
    L.backward() # back propagate 
    assert np.allclose(z.grad,[[0],[0],[0],[0]], atol=0.1)
    z = th.tensor([[ 1000.0], # linear logits for the first sample in the mini-batch
                   [ 1000.0], # linear logits for the second sample in the mini-batch
                   [ 1000.0], # linear logits for the third sample in the mini-batch
                   [ 1000.0]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.tensor([[1.], [1.], [1.], [1.]])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.data,0,atol=1e-3) 
    L.backward() # back propagate 
    dL_dz_true = [ [0],[0],[0],[0]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.1)
    z = th.tensor([[ 1000.0], # linear logits for the first sample in the mini-batch
                   [ 1000.0], # linear logits for the second sample in the mini-batch
                   [ 1000.0], # linear logits for the third sample in the mini-batch
                   [ 1000.0]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.tensor([[0.], [0.], [0.], [0.]])
    L = compute_L(z,y)
    assert np.allclose(L.data,1000,atol=1e-3) 
    L.backward() # back propagate 
    dL_dz_true = [ [0.25],[0.25],[0.25],[0.25]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.1)
    z = th.tensor([[ 1000.0], # linear logits for the first sample in the mini-batch
                   [ 2000.0], # linear logits for the second sample in the mini-batch
                   [ 4000.0], # linear logits for the third sample in the mini-batch
                   [ 5000.0]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.tensor([[0.], [0.], [0.], [0.]])
    L = compute_L(z,y)
    assert np.allclose(L.data,3000,atol=1e-3) 
    L.backward() # back propagate 
    dL_dz_true = [ [0.25],[0.25],[0.25],[0.25]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.1)
#---------------------------------------------------
def test_update_parameters():
    ''' (5 points) update_parameters'''
    m = LogisticRegression(3)# create a logistic regression object
    m.layer.weight.data = th.tensor([[ 0.5, 0.1,-0.2]])
    m.layer.bias.data = th.tensor([0.2]) 
    # create a toy loss function: the sum of all elements in w and b
    L = m.layer.weight.sum() + m.layer.bias.sum()
    # create an optimizer for w and b with learning rate = 0.1
    optimizer = th.optim.SGD(m.parameters(), lr=0.1)
    # (step 1) back propagation to compute the gradients
    L.backward()
    assert np.allclose(m.layer.weight.grad,np.ones((3,1)),atol=1e-2) 
    assert np.allclose(m.layer.bias.grad,1,atol=1e-2) 
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # lets check the new values of the w and b
    assert np.allclose(m.layer.weight.data,[[ 0.4,  0., -0.3]],atol=1e-2) 
    assert np.allclose(m.layer.bias.data,[0.1],atol=1e-2) 
    # (step 2) back propagation again to compute the gradients
    L.backward()
    update_parameters(optimizer)
    assert np.allclose(m.layer.weight.data,[[ 0.3,  -0.1, -0.4]],atol=1e-2) 
    assert np.allclose(m.layer.bias.data,[0.],atol=1e-2) 
    # (step 3) back propagation again to compute the gradients
    L.backward()
    update_parameters(optimizer)
    assert np.allclose(m.layer.weight.data,[[ 0.2,  -0.2, -0.5]],atol=1e-2) 
    assert np.allclose(m.layer.bias.data,[-0.1],atol=1e-2)
#---------------------------------------------------
def test_train():
    ''' (5 points) train'''
    # create a toy dataset (p = 2, batch_size = 2)
    class toy1(Dataset):
        def __init__(self):
            self.X  = th.tensor([[0., 0.], 
                                 [1., 1.]])
            self.Y = th.tensor([[0.], [1.]])
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    # create a toy dataset
    d = toy1()
    # create a dataset loader
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    # train a logistic regression model
    m = train(data_loader, p=2, alpha=0.1,n_epoch = 100)
    # test the data
    x=th.tensor([[0., 0.], [1., 1.]])
    z = compute_z(x,m)
    assert z[0,0]<0 # the class label for the first sample should be 0
    assert z[1,0]>0 # the class label for the second sample should be 1
    # create another toy dataset 
    # p = 2, batch_size = 2
    class toy2(Dataset):
        def __init__(self):
            self.X  = th.tensor([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]])
            self.Y = th.tensor([[0.], [1.], [1.], [1.]])
    
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy2()
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    m = train(data_loader, p=2, alpha=0.1,n_epoch = 100)
    # test the data
    x=th.tensor([[-1., -1.],
                 [2., 0.],
                 [0., 2.],
                 [2., 2.]])
    z = compute_z(x,m)
    assert th.max(z[0,0])<0
    assert th.max(z[1,0])>0
    assert th.max(z[2,0])>0
    assert th.max(z[3,0])>0


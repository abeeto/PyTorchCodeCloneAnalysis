from problem3 import *
import sys
import math
import numpy as np
from model import CNN
from torch.utils.data import Dataset, DataLoader
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (5 points) compute_z'''
    x = th.zeros(2,1,64,64) # a mini-batch of 2 gray-scale images (1 channel) of size 64 X 64
    m = CNN()
    m.conv1.bias.data = th.zeros(10)
    m.conv2.bias.data = th.zeros(20)
    m.conv3.bias.data = th.zeros(30)
    m.fc.bias.data = th.zeros(1)
    z=m(x)
    assert np.allclose(z.data,np.zeros((2,1)))
    x = th.ones(2,1,64,64)
    z=m(x)
    assert not np.allclose(z.data,np.zeros((2,1)))
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
    x = th.zeros(2,1,64,64) # a mini-batch of 2 gray-scale images (1 channel) of size 64 X 64
    y = th.zeros(2,1)
    m = CNN()
    m.conv1.bias.data = th.zeros(10)
    m.conv2.bias.data = th.zeros(20)
    m.conv3.bias.data = th.zeros(30)
    m.fc.bias.data = th.zeros(1)
    optimizer = th.optim.SGD(m.parameters(), lr=0.1)
    z=m(x)
    L = compute_L(z,y)
    assert np.allclose(L.data,0.6931,atol=0.01)
    L.backward()
    update_parameters(optimizer)
    assert np.allclose(m.fc.bias.data,[-0.05],atol=0.01)
    assert np.allclose(m.conv3.bias.data,np.zeros(30),atol=0.01)
    x = th.ones(4,1,64,64)
    y = th.tensor([[1.],[0.],[1.],[0.]])
    m.conv1.bias.data = th.zeros(10)
    m.conv2.bias.data = th.zeros(20)
    m.conv3.bias.data = th.zeros(30)
    m.fc.bias.data = th.zeros(1)
    optimizer = th.optim.SGD(m.parameters(), lr=1.)
    z=m(x)
    L = compute_L(z,y)
    L.backward()
    update_parameters(optimizer)
    assert not np.allclose(m.conv3.bias.data,np.zeros(30))
    th.save(m,"cnn.pt") # save the CNN for demo
#---------------------------------------------------
def test_train():
    ''' (5 points) train'''
    dataset = th.load("face_dataset.pt")# load face image dataset
    class face_dataset(Dataset):
        def __init__(self):
            self.X  = dataset["X"] 
            self.Y = dataset["y"]
        def __len__(self):
            return 20 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d=face_dataset()
    data_loader = DataLoader(d, batch_size=2, shuffle=True, num_workers=0)
    m = train(data_loader, alpha=0.01, n_epoch=150) # train the CNN model
    z=m(dataset["X"])
    y_predict = (z>0).float()
    y = th.tensor(np.concatenate((np.array([1.]*10),np.array([0.]*10))).reshape(20,1))
    correct = y.eq(y_predict).float()
    assert correct.sum().data >14
    th.save(m.state_dict(),"cnn.pt")


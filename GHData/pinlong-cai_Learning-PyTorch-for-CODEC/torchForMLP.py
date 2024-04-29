'''
key points in this file:
-- use the Pytorch to achieve a MLP(multi-layer perception) neural network.
-- the train and test data is the classical benchmark of MNIST.
-- it can be run by CPU or GPU .
'''

import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
import time
 
print("start")
# number of training
EPOCH=50
# batch size
BATCH_SIZE=20 
# learn rate that should not be too large when using cross-entropy loss function 
LR=0.03 
# use MNIST for training and testing, and if it had been download, DOWNLOAD_MNIST is False
DOWNLOAD_MNIST= True

# GPU can be used for computation to speed up the running process
cuda_available=torch.cuda.is_available() 
# if there is no GPU, the CPU can also be used 
cuda_available=False 

# set a set to display data, and transform it into tensor size
# normilize the data with normal distribution with the parameter of 0.5 and 0.5 -- N(0.5,0.5)
trans=torchvision.transforms.Compose(
    [
        # ToTensor method change [0,255] into [0,1]
        torchvision.transforms.ToTensor(), 
        # represent the mean and standard deviation, respectively
        torchvision.transforms.Normalize([0.5],[0.5]) 
    ]
)
 
print("load data")
train_data=torchvision.datasets.MNIST(
    root="./data/mnist", # the site of the data
    train=True, # for training 
    transform=trans, # for transforming
    download=DOWNLOAD_MNIST 
)

# the second parameter is the batch size, and the third parameter is to shuffle the data
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
 
test_data=torchvision.datasets.MNIST(
    root="./data/mnist",
    train=False, # for testing, so here is False
    transform=trans,
    download=DOWNLOAD_MNIST
)
test_loader=DataLoader(test_data,batch_size=len(test_data),shuffle=False)
print("net creating")
 
# form a neural network with three layer 784 nodes/30 nodes/10 nodes
net=torch.nn.Sequential(
    nn.Linear(28*28,30),
    nn.Tanh(), # activation function
    nn.Linear(30,10)# 10 categories for numbers
)
 
if cuda_available:
    net.cuda() # select GPU or CPU
 
# define the cross entropy as loss function 
loss_function=nn.CrossEntropyLoss()
# SGD: stochastic gradient decent
optimizer=torch.optim.SGD(net.parameters(),lr=LR) 
 
print("start training")
for ep in range(EPOCH):
    # record the start time to check the time consumption of each epoch
    startTick = time.time()
    # select each patch in training data
    for data in train_loader:
        img, label=data
        # change the size of image into a column vector
        img = img.view(img.size(0), -1)
 
        if cuda_available:
            img=img.cuda()
            label=label.cuda()
            
        # get the output from the net
        out=net(img)

        # get the loss
        loss=loss_function(out,label)
        # clean the previous gradients
        optimizer.zero_grad()
        # backward the loss to update the gradients of parameter, which is called Autograd
        loss.backward()
        # update gradients
        optimizer.step()
 
    # calculate the number of successful classification samples
    num_correct = 0
    # as the size of test samples is equal to batch size, so the loop runs only once
    for data in test_loader:
        img, label=data
        img = img.view(img.size(0), -1)
 
        if cuda_available:
            img=img.cuda()
            label=label.cuda()

        # get the output from the net
        out=net(img)
 
        # torch.max() return two results
        # the first is the maximum, and the second is the corresponding index
        # 0 represents the index of maximum in column
        # 1 represents the index of maximum in row
        _,prediction=torch.max(out,1)
        print(prediction)

        # count the number of correct classification samples
        num_correct+=(prediction==label).sum()
    
    # get the accuracy, if the GPU is used, then the num_correct should be changed by .cpu()
    accuracy=num_correct.cpu().numpy()/len(test_data)
    timeSpan = time.time()-startTick
    print("Iteration time: %d, Accuracy: %f, running time: %ds"%(ep+1,accuracy,timeSpan))
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:28:58 2017

@author: Rensu Theart
"""

from __future__ import print_function

# standard imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

# for the MNIST Dataset
from torchvision import datasets, transforms

# gives a slight performance boost on GPU
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

# to plot loss
import matplotlib.pyplot as plt

# for timing
import timeit

##############################################################################
# Define variables for CNN
##############################################################################
batch_size = 100
kernel_sz = 5
epochs = 10

# for plots
loss_array = []
epoch_loss_array = []

##############################################################################
# load MNIST dataset
##############################################################################
train_dataset = datasets.MNIST(root='./data/',train=True, transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data/',train=False, transform=transforms.ToTensor(),download=True)

# batch the data for the training and test datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True)

print(train_loader.__len__()*train_loader.batch_size, 'train samples')
print(test_loader.__len__()*test_loader.batch_size, 'test samples\n')

##############################################################################
# CNN model definition (this is the standard approach to define the model, as a class)
##############################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define all the components that will be used in the NN (these can be reused)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_sz) # 1 input feature, 10 output filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_sz)  # 10 input filters, 20 output filters
        self.mp = nn.MaxPool2d(2, padding=0)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.fc1 = nn.Linear(320,100)  #the 320 is dimensional result after max pooling and applying the kernel, 10 outputs
        self.fc2 = nn.Linear(100,10)
        
    def forward(self, x):
        # define the acutal network
        in_size = x.size(0) # this is the batch size
        # you can chain function together to form the layers
        x = F.relu(self.mp(self.conv1(x))) 
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(in_size, -1) # flatten data, -1 is inferred from the other dimensions (which is 320 here)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


##############################################################################
# initialize model and optimizer
##############################################################################    
net = Net()
net.cuda()    # this is the only line necessary to make your model run on GPU

# standard gradient decent (defining the learning rate and momentum)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


##############################################################################
# Define functions
##############################################################################
# the training function, looping over the batches
def train(epoch):
    net.train() # set the model in "training mode"

    for batch_idx, (data, target) in enumerate(train_loader):
        # data.cuda() loads the data on the GPU, which increases performance
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        optimizer.zero_grad() # necessary for new sum of gradients
        output = net(data)  # call the forward() function (forward pass of network)
        loss = F.nll_loss(output, target) # use negative log likelihood to determine loss
        loss.backward() # backward pass of network (calculate sum of gradients for graph)
        optimizer.step() # perform model perameter update (update weights)
        
        # for graphing puposes
        loss_array.append(loss.data[0])
        
        # print the current status of training
        if(batch_idx % 100 == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


# the test function
def test(epoch):
    net.eval()  # set the model in "testing mode" (won't update parameters)
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda()) # volatile=True, since the test data should not be used to train... cancel backpropagation
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] #fsize_average=False to sum, instead of average losses
        pred = output.data.max(1, keepdim=True)[1]
        correct+= pred.eq(target.data.view_as(pred)).cpu().sum() # to operate on variables they need to be on the CPU again
        
    
    test_dat_len = len(test_loader.dataset)
    test_loss /= test_dat_len
    
    # print the test accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_dat_len, 100. * correct / test_dat_len))


##############################################################################
# Main function
##############################################################################
if __name__ == '__main__':
    for epoch in range(1,epochs):
        start_time = timeit.default_timer()
    
        train(epoch)
        test(epoch)
    
        elapsed = timeit.default_timer() - start_time
        print("Epoch time is", elapsed, "s\n")


    #plot loss
    plt.plot(loss_array)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.show()

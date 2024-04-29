
#--------------Getting Started

# Tensor

from __future__ import print_function
import torch

x=torch.Tensor(5,3)
print(x)
x=torch.rand(5,3)
print(x)
print(x.size())

# Operations
y=torch.rand(5,3)
print(x*y)
print(torch.add(x,y))
# systax2
result=torch.Tensor(5,3)
torch.add(x,y,out=result)
print(result)
# systax3
y.add_(x)
print(y)
print(x[:,1])

x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)  # the size -1 is inferred from other dimensions
print(x.size(),y.size(),z.size())

# Numpy Bridge

#converting a torch tensor to a numpy array
a=torch.ones(5)
print(a)
b=a.numpy(1)
print(b)
a.add_(1)
print(a)
print(b)

# converting numpy arrary to torch tensor 

import numpy as np
a = np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

# CUDA Tensors

if torch.cuda.is_avaliable():
    x=x.cuda()
    y=y.cuda()
    x+y

 
#-----------------Autograd:automatic differentiation
    
# variable

import torch
from torch.autograd import Variable
x=Variable(torch.ones(2,2),requires_grads=True)
print(x)
y=x+2
print(y)
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z,out)

# Gradients

out.backward()
print(x.grad)

x=torch.randn(3)
x=Variable(x,requires_grad=True)

y=x*2
while y.data.norm()<1000:
    y=y*2
print(y)

gradients=torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)

# ------------ Neural Networks

'''nput -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss'''


# Define the network

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)# 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x)) # the size -1 is inferred from other dimensions
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    def num_flat_features(self,x):
        size=x.size()[1:]  # all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features *= s
        return num_features
net=Net()
print(net)

params=list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
        
input=Variable(torch.randn(1,1,32,32))  #resize nSamples x nChannels x Height x Width
out=net(input)
print(out)

net.zero_grad()  # Zero the gradient buffers of all parameters and 
out.backward(torch.randn(1,10)) # backprops with random gradients

# Loss Function

output=net(input)
target=Variable(torch.arange(1,11)) # a dummy target,for example
criterion=nn.MSELoss()
loss=criterion(output,target)

print(loss)
print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0][0])    # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   #Relu

# Backprop

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weights

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)
    
import torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output=net(input)
loss=criterion(output,target)
loss.backward()
optimizer.step()

# --------------Trainning a classifier


# Loading and normalizing CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms

transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset=torchvision.datasets.CIFAR10(root='/data',train=True,
                                      download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
                                        shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='/data',train=False,
                                     download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,
                                       shuffle=False,num_workers=2)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# show some of the training images, for fun.
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):    # function to show an image
    img=img/2+0.5   #unnormalize
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
#get some random training images
dataiter=iter(trainloader)
images,labels=dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
#print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define a Convolution Neural Network

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) # the size -1 is inferred from other dimensions
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
net=Net()

# Define a Loss function and optimizer

import torch.optim as optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# Train the network

for epoch in range(2): # loop over the dataset multiple times
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data  # get the inputs
        inputs,labels=Variable(inputs),Variable(labels) #wrap them in variable
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.data[0]
        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.3f' % (epoch + 1,i+1,running_loss/2000))
            running_loss=0.0
print('Finished Training')

# test the network on the test data

dataiter=iter(testloader)
images,labels=dataiter.next()

imshow(torchvision.utils.make_grid(images)) #print images
print('GroundTruth: ',' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))
correct=0
total=0
for data in testloader:
    images,labels = data
    outputs=net(Variable(images))
    _,predicted=torch.max(outputs.data,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum().item()
print('accuracy of the network on the 10000 test images:%d %%' % (100*correct/total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('accuracy of %5s : %2d %%' % (classes[i],100*class_correct[i]/class_total[i]))




net.cuda()
inputs,labels = Variable(inputs.cuda()),Variables(labels.cuda())

# Data Parallelism

model.gpu()
mytensor = my_tensor.gpu()
model=nn.DataParallel(model)

#imports and parameters

import torch
import torch.nn as nn
from torch.autograd import Variable
from troch.utils.data import Dataset,DataLoader

input_size=5
output_size=2
batch_size=30
data_dize=100

#Dummy Dataset

class RandomDataset(Dataset):
    def __init__(self,size,length):
        self.len=length
        self.data=torch.randn(length,size)
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return self.len
rand_loader = DataLoader(Dataset=RandomDataset(input_size,100),batch_size=batch_size,shuffle=True)

# simple model

class Model(nn.Moudle):
    def __init__(self,input_size,output_size):
        super(Model,self).__init__()
        self.fc=nn.Linear(input_size,output_size)
    def forward(self,input):
        output=self.fc(input)
        print('In model: input size',input.size(),'output size',output.size())
        return output

#--------Create Model and DataParallel

model = Model(input_size,output_size)
if torch.cuda.device_count()>1:
    print("let's use",torch.cuda.device_count(),'GPUs!')
    model=nn.DataParallel(model)
if torch.cuda.is_avaliable():
    model.cuda()

# Run the model

for data in rand_loader:
    if torch.cuda.is_avaliable():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output=model(input_var)
    print("outside : input size",input_var.size(),'output_size',output.size())

# Results

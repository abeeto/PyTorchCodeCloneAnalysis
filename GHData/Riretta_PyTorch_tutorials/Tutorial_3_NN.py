#Neural Network

import torch
import torch.nn as nn
import torch.nn.functional as F

#define the network
#this is an example of a basic NN model
#basically all the networks need an Init function (to define the structure of the model)
#the forward() function that defines how navigate through layers

class Net(nn.Module):

    def __init__(self):
        #constructor
        super(Net, self).__init__()
        #1 input, 6 output, 5x5 square conv
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #y = Wx+b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        #max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        #if the size is a square it is possible to specify only one number
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #view is a resize
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#learnable parameters
params = list(net.parameters())
print(len(params))
print(params[0].size())

#random input
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

print("LOSS FUNCTION")
#computing loss function
output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

#my net is structured as:
#input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view ->
# linear -> relu -> linear - >relu -> linear -> MSEloss-> loss

print(loss.grad_fn) #MSEloss
print(loss.grad_fn.next_functions[0][0]) #Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #relu

print("BACKPROP ")
#backprop
net.zero_grad() #zeroes the gradient buffers of all parameters

print('Conv1.bias.grad before backward()')
print(net.conv1.bias.grad)

loss.backward()
print('Conv1.bias.grad after backward()')
print(net.conv1.bias.grad)

print("UPDATE WEIGHTS")
#stochastic gradient descent
#W = W - LR*Grad

import torch.optim as optim
import matplotlib.pyplot as plt

epochs = 200

plt.axis([0, epochs, 0, 1])

#create optimizer
optimizer = optim.SGD(net.parameters(),lr = 0.01)

#training loop <-------------------
for i in range(1,epochs,1):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output,target)
    print(loss)
    plt.scatter(i,loss.item())
    plt.pause(0.05)
    loss.backward()
    optimizer.step()


plt.show()

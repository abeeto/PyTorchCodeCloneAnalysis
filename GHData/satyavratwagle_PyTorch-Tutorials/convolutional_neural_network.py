import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional Layers
        
        #nn.Conv2D( #input channels, #output channels, square kernel)
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        # Affine operations 
        
        #nn.Linear(#Inputs,#Outputs)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        
        # Max Pooling
        
        # F.max_pool2d( What to maxpool, maxpool kernel size)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        
        # Flatten output of convolutional layer
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features
    
net = Net()
print(net)

# Forward Prop

input_to_network = torch.randn(1, 1, 32, 32)

# Computing Loss

output = net(input_to_network)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# Get the backward path of loss
#print(loss.grad_fn)                                             # MSELoss
#print(loss.grad_fn.next_functions[0][0])                        # Linear
#print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# update weights

learning_rate = 0.01

# Explicitly

def update(net):
    for f in net.parameters():
        f.data.sub_(f.grad.data*learning_rate)
        
# Using package
        
optimizer = optim.SGD(net.parameters(),learning_rate)

iterations = range(500)
for i in iterations:
    optimizer.zero_grad()
    output = net(input_to_network)
    loss = criterion(output, target)
    if(i%50)==0:
        #print(loss.detach().numpy())
        print(net.conv1)
    loss.backward()
    optimizer.step()

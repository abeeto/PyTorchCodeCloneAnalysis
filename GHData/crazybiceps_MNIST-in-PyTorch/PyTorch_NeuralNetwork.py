from __future__ import print_function

import math
import torch
import numpy as np
from torch.autograd import Variable as V
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim  # Imports the optimization function such as adam , rmsprop ,SGD, Nesterov-SGD
import torch.utils.data

random_seed = 3.14
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):
    
    def __init__(self):
        super(Net , self).__init__()
        
        self.conv1 = nn.Conv2d(1 , 8 , 3)
        self.conv2 = nn.Conv2d(8 , 16 , 3)
        
        self.fc1   = nn.Linear(16 * 5 * 5 , 64  )
        self.fc2   = nn.Linear(64 , 32)
        self.fc3   = nn.Linear(32 , 10)
        
    def forward(self , x):
        x = F.max_pool2d(F.relu(self.conv1(x)) , 2)
        x = F.max_pool2d(F.relu(self.conv2(x)) , 2)
        
        x = x.view(-1 , self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        
        return x
    
    def num_flat_features(self , x):
        num_features = np.prod(np.array(x.detach().numpy().shape[1:]))
        
        return num_features
    
net = Net()
net

params = list(net.parameters())

len(params)
list(map(len , params))

# Random Input
input = torch.randn( 1 ,  1 , 28 , 28)
out = net(input)
print(out)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters() , lr = .01 , momentum = .5)

# Loading MNIST data
w = pd.read_csv("/home/manpreet/D/Python_Codes/MNIST_Train/train.csv") # Downloaded file location should be changed here 

# View some random images
plt.imshow(x_train[ 0 , :].reshape(28 , 28) , cmap = "gray")
plt.imshow(x_train[ 1 , :].reshape(28 , 28) , cmap = "gray")
plt.imshow(x_train[ 8 , :].reshape(28 , 28) , cmap = "gray")

# Random input in batches
batch_size = 256
train = w.values
train_loader = torch.utils.data.DataLoader(train , 
                                           batch_size = batch_size , 
                                            shuffle = True , 
                                            num_workers = 2)

example = enumerate(train_loader)
batch_id , example_data = next(example)

for epoch in range(100):
    net.train()
    sum_acc = 0
    for idx , data in enumerate(train_loader):
        
        x_train = V(data[ : , 1:].reshape(data.size(0) , 1 , 28 , 28)).float()
        y_train = V(torch.FloatTensor(data[ : , 0].float())).long()
        
        optimizer.zero_grad()
        
        outputs = net(x_train)
        loss = criterion(outputs.exp() , y_train)
        acc = sum(np.reshape(np.array(list(map(lambda x : np.where(x == max(x)) , outputs))) , 
                             data.size(0)) == np.array(y_train))    
        sum_acc += acc
        loss.backward()
        optimizer.step()
        
    print(epoch , loss.item() , (sum_acc / w.values.shape[0]))
    


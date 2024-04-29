import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models as models
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(225*225,400)
        self.fc2 = nn.Linear(400,2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#region データセット
train_data = torchvision.datasets.ImageFolder(root='Resources/train/', transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(225),
    transforms.ToTensor(),
]))
train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)    

test_data = torchvision.datasets.ImageFolder(root='./train', transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(225),
    transforms.ToTensor(),
]))
test_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4)
#endregion
"""
#reigon 学習(パラメータの更新)
net = Net()
optimizer = optim.Adam(net.parameters(),lr=0.01)
criterion = nn.MSELoss()


#endregion
"""


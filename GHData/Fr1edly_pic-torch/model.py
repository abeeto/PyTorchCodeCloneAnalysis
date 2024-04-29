import torch
import torch.nn as nn
import torch.nn.functional as F


class imgNet(nn.Module):
    def __init__(self):
        super(imgNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 3)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12, 36, 5)
        self.conv2_bn = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 108, 3)
        self.conv3_bn = nn.BatchNorm2d(108)
        self.fc1 = nn.Linear(972, 2430)
        self.fc1_bn = nn.BatchNorm1d(2430)
        self.fc2 = nn.Linear(2430, 320)
        self.fc2_bn = nn.BatchNorm1d(320)
        self.fc3 = nn.Linear(320,67)
        self.fc3_bn = nn.BatchNorm1d(67)
        self.dropout = nn.Dropout()
        """
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,1) 
        self.conv3_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*3*3,256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 =  nn.Linear(256,64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,10)
        self.dropout = nn.Dropout(p=0.5)
        """
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
       # print(x.size(0))
       #x = x.view(x.size(0),36*3*3)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1,x.size(0))
        x = F.relu(self.fc1(x))
        #x = self.dropout(self.fc1_bn(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(self.fc2_bn(x))
        x = self.fc3(x)
        """
        return x

        
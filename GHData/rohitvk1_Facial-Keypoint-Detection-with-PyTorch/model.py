# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)

        # Max-Pool layer
        self.pool = nn.MaxPool2d(2,2)
    	
    	# Linear layers
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(500, 136)
        
        # Dropout
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.25)
        self.drop4 = nn.Dropout(p = 0.3)
        self.drop5 = nn.Dropout(p = 0.4)
        self.drop6 = nn.Dropout(p = 0.5)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=128, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=1000, eps=1e-05)
        self.bn5 = nn.BatchNorm2d(num_features=1000, eps=1e-05)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)
        x = self.drop4(x)
        
        #Flatten
        x = x.view(x.size(0),-1)

        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.drop5(x)

        x = F.relu(self.fc2(x))
        x = self.bn5(x)
        x = self.drop6(x)
        
        x = self.fc3(x)

        return x
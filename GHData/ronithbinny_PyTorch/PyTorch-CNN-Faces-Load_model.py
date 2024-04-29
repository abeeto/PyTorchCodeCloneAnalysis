import os
import cv2
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

X = torch.tensor([i[0] for i in training_data]).view(-1, 28, 28)
X = X/255.0
y = torch.tensor([i[1] for i in training_data])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 28x28x1 => 28x28x8
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x8 => 14x14x8
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0) # (2(14-1) - 28 + 2) = 0                                       
        # 14x14x8 => 14x14x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1) # (1(14-1) - 14 + 3) / 2 = 1                 
        # 14x14x16 => 7x7x16                             
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                         stride=(2, 2),
                                         padding=0) # (2(7-1) - 14 + 2) = 0

        self.linear_1 = torch.nn.Linear(7*7*16, 64)
        
        self.linear_2 = torch.nn.Linear(64, 128)
        
        self.linear_3 = torch.nn.Linear(128, 128)
        
        self.linear_4 = torch.nn.Linear(128, 2)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)
        
        out = self.linear_1(out.view(-1, 7*7*16))
        out = self.linear_2(out)
        out = self.linear_3(out)
        out = self.linear_4(out)
        
        return F.softmax(out, dim=1)


net = Net()

PATH = r'C:\Users\Ronith\Desktop\Lincode Labs\PyTorch\Faces\face'
net = torch.load(PATH)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(0,len(X))):
        net_out = net(X[i].view(-1, 1, 28, 28))
        predicted_class = torch.argmax(net_out)

        if predicted_class == y[i]:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))
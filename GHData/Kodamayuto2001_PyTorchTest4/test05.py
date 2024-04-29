import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#network definition
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(4,10)
        self.fc2 = nn.Linear(10,8)
        self.fc3 = nn.Linear(8,3)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

iris = datasets.load_iris()
y = np.zeros((len(iris.target),1+iris.target.max()),dtype=int)
y[np.arange(len(iris.target)),iris.target] = 1
X_train,X_test,y_train,y_test = train_test_split(iris.data,y,test_size=0.25)

input = torch.randn(1,1,32,32,requires_grad=True)
print(input)
print(input.shape)
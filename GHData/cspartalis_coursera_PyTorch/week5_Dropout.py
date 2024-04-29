import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net,self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)
    def forward(self,x):
        x = torch.relu(self.linear1(x))
        x = self.drop(x)
        x = torch.relu(self.linear2(x))
        x = self.drop(x)
        x = self.linear3(x)
        return x

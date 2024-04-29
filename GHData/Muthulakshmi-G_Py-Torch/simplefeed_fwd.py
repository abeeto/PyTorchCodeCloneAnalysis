import torch
from torch.autograd import grad
from torch.nn import functional as F

from torch import nn

input_size=3
num_classes=2
batch_size=5
hidden_size=4

torch.manual_seed(123)
input=autograd.Variable(torch.rand(batch_size,input_size))

print('input',input)


class Model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.h1=nn.Linear(input_size,hidden_size)
        self.h2=nn.Linear(input_size,num_classes)


 


    def forward(self,x):
        x=self.h1(x)
        x=self.h2(x)
        x=F.tanh(x)
        return x




model=Model(input_size=input_size,hidden_size=hidden_size,num_clsses=num_classes)

out=model(input)

print('out',out)



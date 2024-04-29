import torch
import torch.nn as nn
from torch.nn import Linear

class LR(nn.Module):
    def __init__(self,in_size,output_size):
        # super constructor allows to create objects from
        # the package nn.Module w/o initializing explicitly
        super(LR,self).__init__()
        self.linear=nn.Linear(in_size,output_size)

    def forward(self,x):
        out=self.linear(x)
        return out

model=LR(1,1)
# use state_dict to initialize the weight and bias of the model
model.state_dict()['linear.weight'].data[0]=torch.tensor([0.5153])
model.state_dict()['linear.bias'].data[0]=torch.tensor([-0.4414])
print(model.state_dict())
print(list(model.parameters()))
x=torch.tensor([[1.0],[2.0]])
yhat=model(x)
print('yhat is: ',yhat)

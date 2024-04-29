import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,Layers):
        super(Net,self).__init__()
        self.hidden = nn.ModuleList()
        for input_size,output_size in zip(Layers,Layers[1:]):
            self.hidden.append(nn.Linear(input_size,output_size))

        def forward(self,activation):
            L = len(self.hidden)
            for (l,linear_transform) in zip(range(L),self.hidden):
                if l < L-1:
                    activation = torch.relu(linear_transform(activation))
                else:
                    activation = linear_transform(activation)
            return activation

Layers = [2,3,4,3]
model = Net(Layers)
yhat = model(x)
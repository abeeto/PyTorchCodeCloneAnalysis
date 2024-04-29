
import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_units, out_units):
        super().__init__()
        self.layer = nn.Linear(in_units, out_units)
        nn.init.normal_(self.layer.weight, 0, 1e-4)
        nn.init.constant_(self.layer.bias, 0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x): return self.dropout(self.layer(x))

class Maxout(nn.Module):

    def __init__(self, in_units, out_units, pieces):
        super().__init__()
        self.out_units = out_units
        self.pieces = pieces
        self.linear = Linear(in_units, out_units * pieces)
    
    def forward(self, x):
        return self.linear(x).view(-1, self.out_units, self.pieces).max(-1)[0]

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_z = nn.Sequential(Linear(100, 200), nn.ReLU())
        self.layer_y = nn.Sequential(Linear(10, 1000), nn.ReLU())
        self.layer_join = nn.Sequential(Linear(1200, 1200), nn.ReLU())
        self.layer_out = nn.Sequential(Linear(1200, 784), nn.Sigmoid())

    def forward(self, z, y):
        joint = torch.cat((self.layer_z(z), self.layer_y(y)), 1)
        return self.layer_out(self.layer_join(joint))

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_x = Maxout(784, 240, 5)
        self.layer_y = Maxout(10, 50, 5)
        self.layer_join = Maxout(290, 240, 4)
        self.layer_out = nn.Sequential(nn.Linear(240, 1), nn.Dropout(0.5), nn.Sigmoid())

    def forward(self, x, y):
        joint = torch.cat((self.layer_x(x), self.layer_y(y)), 1)
        return self.layer_out(self.layer_join(joint))

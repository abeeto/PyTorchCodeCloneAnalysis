import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Decoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))
    
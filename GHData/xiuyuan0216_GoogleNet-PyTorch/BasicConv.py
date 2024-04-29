import torch 
import torch.nn as nn 


class BasicConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.nets = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.nets(x)
        return out
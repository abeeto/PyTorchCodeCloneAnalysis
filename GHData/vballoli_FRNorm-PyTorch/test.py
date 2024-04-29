from frn import FRNorm
import torch

frn_layer = FRNorm((3, 3, 32, 32))
t = torch.randn(3, 3, 32, 32)
print(frn_layer(t))
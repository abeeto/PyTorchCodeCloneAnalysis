import torch 
import torch.nn as nn 
import torch.optim as optim 




m1 = nn.Sequential(nn.Linear(1,3), nn.Tanh(), nn.Linear(3,1))
m2 = nn.Sequential(nn.Linear(1,3), nn.Tanh(), nn.Linear(3,1))

# putting modules into a module list allows to get easily their parameters 
adam = optim.Adam(nn.ModuleList([m1, m2]).parameters(), lr = 1e-3)

params = adam.param_groups
print(len(params[0]['params']))

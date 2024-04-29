import torch 
import torch.nn as nn 
import torch.nn.functional as F 



t1 = nn.Sequential(nn.Linear(1,3), nn.Tanh(), nn.Linear(3,1))
t2 = nn.Sequential(nn.Linear(1,3), nn.Tanh(), nn.Linear(3,1))



it_m1 = t1.modules()
it_m2 = t2.modules() 

for m1, m2 in zip(it_m1, it_m2): 
	if isinstance(m1, nn.Linear): 
		m1.weight.data.copy_(m2.weight.data)


for m1, m2 in zip(t1.modules(), t2.modules()): 
	if isinstance(m1, nn.Linear): 
		print('M1 values: {}\nBiais: {}\n\nM2 {}\nBiais {}\n\n\n'.format(m1.weight, m1.bias, m2.weight, m2.bias))




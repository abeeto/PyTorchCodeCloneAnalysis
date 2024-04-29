import torch 
import torch.nn as nn 


model = nn.Sequential(nn.Linear(1,3), nn.Linear(3,1))

def weight_init(m):
	
	if isinstance(m, nn.Linear): 
		print(m)
		m.weight.data.normal_(0.,0.02)
		m.bias.data.zero_()

def xavier_init(m): 

	if isinstance(m, nn.Linear): 
		nn.init.xavier_normal_(m.weight)
		m.bias.data.zero_()

for l in model: 
	print(l)
	for p in l.parameters():
		print(p)

model.apply(xavier_init)

for l in model: 
	print(l)
	for p in l.parameters():
		print(p)

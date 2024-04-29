import torch 
import torch.nn as nn 
import copy 

import torch.optim as optim
import torch.nn.functional as F 

def print_model(model): 

	for p in model.parameters():
		print(p)

class Model(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(1,5)

	def forward(self, x): 

		return self.l1(x)


model_a = Model()
model_b = Model()

model_b.load_state_dict(model_a.state_dict())

print_model(model_a)
print('\n' + '='*20)
print_model(model_b)

print('\n'*3)
# They do have the same parameters 

# Sharing ? 

adam = optim.Adam(model_a.parameters(),1.)
out = model_a(torch.rand(1,1))
loss = F.mse_loss(out, torch.ones_like(out))

loss.backward()
adam.step()

print_model(model_a)
print('\n' + '='*20)
print_model(model_b)

print('Models are not sharing ! ')


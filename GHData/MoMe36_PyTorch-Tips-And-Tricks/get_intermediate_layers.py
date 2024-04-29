import torch 
import torch.nn as nn 
import torch.functional as F 
import torch.optim as optim

class MyModel(nn.Module):

	def __init__(self): 

		super().__init__()

		self.register = nn.Sequential(nn.Linear(1,3), nn.ReLU(),nn.Linear(3,5), nn.ReLU(), nn.Linear(5,3))

	def forward(self, x): 

		results = []
		for i,l in enumerate(self.register): 

			x = l(x)
			if i in [0, 2, 4]: 
				results.append(x)
		return x, results


model = MyModel()
print(model)


x = torch.rand(1,1)

out = model(x)

loss = out[1][1].mean()

print(loss)

loss.backward()

for i, layer in enumerate(model.register): 

	if isinstance(layer, nn.Linear): 
		print('Layer {} has grads: {}'.format(i, layer.weight.grad))
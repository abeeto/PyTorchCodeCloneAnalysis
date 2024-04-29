import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim



# ================================================
# ================================================
# ================================================
#
#			BASIC MODEL CLASS
#
# ================================================
# ================================================
# ================================================
class Model(nn.Module):

	def __init__(self): 

		super().__init__()


		self.l1 = nn.Linear(2,3, bias = False)
		self.l2 = nn.Linear(3,3, bias = False)
		self.l3 = nn.Linear(3,2, bias = False)

	def forward(self, x): 

		for l in [self.l1,self.l2, self.l3]:
			x = torch.sigmoid(l(x))

		return x 

# Seed for reproductibility 
torch.manual_seed(0)

# Instiantiate model 
model = Model()

# Here we select parameters to optimize 
parameters_to_optimize = []
for c in model.named_children():    # this returns tuples with (layer name, layer) eg: ('l1', Linear(in_features....))
	if not c[0] == 'l2':			# Apply some kind of filter. Here we want to exclude l2 weight and bias from the optimization process 
		for p in c[1].parameters(): # Add the parameters you're interested in into a list that we'll pass to the optimizer
			parameters_to_optimize.append(p)

adam = optim.Adam(parameters_to_optimize, lr = 1.) # Pass the list 
print(adam.param_groups) # Check whether we do have the wanted parameters 


# ============================================================
# UNIT TEST 
# ============================================================

x = torch.rand(5,2)*5
out = model(x)

print('Prediction before')
print(out)
loss = torch.mean(torch.pow(out,2))
print('Loss: ', loss)

print('parameters before')
for p in model.parameters():
	print(p)

adam.zero_grad()
loss.backward()
adam.step()

print('\n'*3)
print('parameters after')
for p in model.parameters():
	print(p)

out = model(x)
print('\n\n\nPrediction after\n', out)
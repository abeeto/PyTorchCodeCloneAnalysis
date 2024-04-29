import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyper parameters

step_size = 0.01 # How much should the network weights change in case of a wrong prediction?

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # Construct network graph according to given parameters
		
        #nn.Affine(#Inputs,#Outputs)
		
        self.fc1 = nn.Linear(input_size,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        
        # Forward pass through network
		
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
# Create Network object to pass data to
	
net = Net()

input_to_network = torch.randn(1, 32)

# Get target values

target = torch.randn(10)  		# a dummy target, for example
target = target.view(1, -1)  	# make it the same shape as output

# Network paramaters for gradient computation

criterion = nn.MSELoss() 						# Define loss criteron
optimizer = torch.optim.SGD(net.parameters(), 
							step_size)			# Optimizer selection

for i in range(iterations):
	'''
	Steps:
		- Set gradient of all layers to be 0
		- Compute prediction on given inputs by forward pass through the network
		- Calculate loss between predictions and truth values
		- Calculate gradients for all layers with backpropagation
		- Change values of weights in layers according to optimizer and step size
	'''
	net.zero_grad()
	output = net(input_to_network)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()

# END OF CODE -----------------------------------------------------------------


# Ways to update weights :

# Explicitly

def update(net):
    for f in net.parameters():
        f.data.sub_(f.grad.data*step_size)
        
# Using package
        
optimizer = optim.SGD(net.parameters(),step_size)

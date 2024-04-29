# Computational graphs and autograd are still too low-level for complex neural networks because they require manual definitions for each stage, although not for each derivate.
# For TensorFlow framework, packages such as Keras, TensorFlow-Slim and TFLearn provide higher level abstraction.
# For PyTorch, the nn package serves a similar purpose.
# In nn, we work with 'modules' which are roughly the same as neural network 'layers'. A module receives input from tensors and computes output tensors, but may also hold internal state such as tensors containing learnable parameters.
# The nn package also have predefined loss functions that are commonly used when training neural networks
# In this example, nn package is demonstrated for the two-layer neural network.

import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10


# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out),)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.

loss_fn = torch.nn.MSELoss(reduction='sum')

# Set learning rate

learning_rate = 1e-4

# Iterate

for t in range(500):
	# Forward pass: Here we pass out input to the nn model 
	y_pred = model(x)
	
	# Compute and print loss: We pass the prediction and truth tensors to the loss function
	loss = loss_fn(y_pred, y)
	print(t, loss.item())
	
	# Zero the gradients before running the backward pass. We do this because by default pytorch uses additive gradients, which is useful when running RNNs. For straight forward NNs, this has to be manually set to zero to prevent additive gradients.
	model.zero_grad()
	
	# Backward pass: Same as before. This works although we did not explicity set "require_grad = True" for our weights because the nn.Sequential takes care of this for us.
	loss.backward()
	
	# Update parameters:  Access gradients as before using no_grad and calling model.parameters()
	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad


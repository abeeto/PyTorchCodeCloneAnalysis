# Updating weights manually becomes complicated if we have a lot of layers in our neural network. Further, this is easy only while we use straight forward gradient descent (new weight = weight - learning_rate * weight_gradient). Often we use optimization methods for gradient descent such as Adam, RMSProp, AdaGrad etc, which complicate the new weight calculation (they also add new hyperparameters).
# The optim package in PyTorch abstracts the idea of an optimization algorithm to provide implementations of commonly used optimization algorithms. We can also create our own optimization methods if necessary.
# This example combines the use of nn package for the model and optim package for the optimization part.

import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10


# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

# Set learning rate

learning_rate = 1e-4

# Optimization with optim package. Pass the parameters from the model along with the learning rate to the optimizer function.

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Iterate

for t in range(500):
	# Forward pass: use model
	y_pred = model(x)
	
	# Compute and print loss
	loss = loss_fn(y_pred, y)
	print(t, loss.item())

	# Instead of setting model gradients to zero, now we set optimizer gradients to 0. This is because by default, gradients are accumulated in buffers( i.e, not overwritten) whenever .backward() is called. Checkout docs of torch.autograd.backward for more details.
	optimizer.zero_grad()
	
	# Backward pass: same as before
	loss.backward()
	
	# Call the optimization step to update parameters
	optimizer.step()
	

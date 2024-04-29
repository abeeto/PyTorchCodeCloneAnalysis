# A two layer network using torch instead of numpy. Fundamentally torch uses a N-dimensional numpy array casted as a different datatype, a tensor. The operations are very comparable between numpy and torch, however, torch allows for much more efficient utilization of computational resources such as GPUs. Numpy cannot use GPUs.
# This is still a hand coded neural network

import torch

dtype = torch.float # Set datatype, always safe to set it to float
device = torch.device("cpu") # No GPUs available on the current system, we will still use CPUs
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device = device, dtype =dtype) # We use torch randn function instead of the numpy randn function - this allows for the right objects to be defined
y = torch.randn(N, D_out, device = device, dtype = dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device =device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

# Set learning rate

learning_rate = 1e-6

# Iterate 

for t in range(500):
	
	# Forward pass
	h = x.mm(w1) # tensor.mm is matrix multiplication, this replaces nparray.dot (dot product) that we used in example1
	h_relu = h.clamp(min=0) # clamps all input values within a range given a min and a max, here we are only interested in a min by definition of ReLu
	y_pred = h_relu.mm(w2)
	
	# Compute and print loss
	loss = (y_pred -y).pow(2).sum().item() # pow is the power function, sum is the sum function and item is to drop array dimension and return a single value (without all brackets)
	print(t, loss)
	
	# Backprop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	#Update weights using gradient descent
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2

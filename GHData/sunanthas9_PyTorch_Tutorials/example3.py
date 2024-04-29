# In this example we will implement automated differentiation calculation during backprop. We use the autograd package of PyTorch for this. 
# When using autograd, the forward pass will define the computational graph, with each node being a tensor and each edge being a function.
# For each node (tensor) x, we will set attribute "x.requires_grad = True" which would automatically create x.grad tensort holding the gradient of x with respect to some scalar value
# In this example we will implement a two-layer network using autograd


import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10 

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype, requires_grad = False)
y = torch.randn(N, D_out, device=device, dtype=dtype, requires_grad = False)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

# Set learning rate

learning_rate =1e-6

# Iterate

for t in range(500):
	# Forward pass: Essentially the same computation as example2, but we don't save intermediate matrices because we are not implementing backprop by hand
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	
	# Compute and print loss
	loss =(y_pred - y).pow(2).sum()
	print(t, loss.item()) # Note that we moved the '.item()' to this line, because we want to save loss as a tensor instead of scalar for later use. Here it helps printing a scalar.
	
	# Backward pass with autograd
	loss.backward() # This computes gradients for ALL tensors which had requires_grad = True and creates their '.grad' tensors. Note that there are other ways of implementing this step as well.
	
	with torch.no_grad(): # Temporarily sets 'requires_grad = False' for all variables 
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
		
		# Manually zero the gradients after updating weights
		w1.grad.zero_()
		w2.grad.zero_()
	

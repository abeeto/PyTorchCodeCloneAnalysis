'''
Network implemented in tensors.py except using autograd instead of manually computing gradients 
'''

import torch 

dtype = torch.float
device = torch.device("cpu")

batch_size, input_dim, hidden_dim, output_dim = 64, 100, 1000, 10

# Random input and output data
# Setting requires_grad = False means that we do 
# not need to compute gradients for these Tensors during the back pass
x = torch.randn(batch_size, input_dim, requires_grad = False, device = device, dtype = dtype)
y = torch.randn(batch_size, output_dim, requires_grad = False, device = device, dtype = dtype)

# Initialize with random weights
# Make requires_grad = True so that gradients are computed for these Tensors 
# during the back pass
w1 = torch.randn(input_dim, hidden_dim, requires_grad = True, device = device, dtype = dtype)
w2 = torch.randn(hidden_dim, output_dim, requires_grad = True, device = device, dtype = dtype)

learning_rate = 1e-6
epochs = 500
for t in range(epochs):
    # Forward pass: compute predicted y using operations on Tensors
    # No need to keep references to immediate values since we're not
    # manually computing gradients
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)

    # Compute and print loss using operations on Tensors
    # Keeping loss as a Tensor so that autograd can work with it
    loss = (y_pred - y).pow(2).sum()
    print(t + 1, loss.item())

    # Use autograd to compute the backwards pass.
    # This operation computes the gradient of the loss with respect to all Tensors 
    # where requires_grad = True. 
    # Once this call is complete, w1.grad and w2.grad will be holding each of their
    # respective gradients
    loss.backward()

    # Manually update weights using gradient descent
    # These operations are wrapped in "with torch.no_grad()" because we don't 
    # need or want autograd to know about us operating on Tensors that have requires_grad = True
    with torch.no_grad():
        w1 -= learning_rate * w1.grad 
        w2 -= learning_rate * w2.grad 

        # Manually zero-out the gradients after updating the weights
        w1.grad.zero_()
        w2.grad.zero_()
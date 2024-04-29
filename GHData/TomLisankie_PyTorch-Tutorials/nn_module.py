'''
An implementation of the same simple two-layer network from tensors.py and autograd.py but using the 
torch.nn module instead.
'''

import torch 

batch_size, input_dim, hidden_dim, output_dim = 64, 100, 1000, 10

# Random input and output data
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)

# Using the nn package to define our model instead of a pure computational graph. 
# nn.Sequential is a Module that contains other Modules and applies them in sequence to produce output. 
# Linear Modules compute the output using a linear function and hold internal Tensors that contain 
# the weights and biases.
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim)
)

# The nn package also contains some popular loss functions
loss_fn = torch.nn.MSELoss(size_average = False)

learning_rate = 1e-4
epochs = 500
for t in range(epochs):
    # Compute predicted y. Module objects override the __call__ operator so you can call them like functions
    y_pred = model(x)

    # Compute and print the loss
    loss = loss_fn(y_pred, y)
    print(t + 1, loss.item())

    # Zero the gradients before running the backward pass
    model.zero_grad()

    # Do backward pass
    loss.backward()

    # Update the weights using gradient descent
    # Each parameter is a Tensor
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
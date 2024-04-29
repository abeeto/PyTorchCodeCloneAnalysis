'''
Implementation of the same two-layer network as tensors.py, autograd.py, and nn_module.py except using 
an optimizer built in to PyTorch instead of optimizing by hand.
'''

import torch 

batch_size, input_dim, hidden_dim, output_dim = 64, 1000, 100, 10

# Create random input and output Tensors
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, output_dim)

# Use the nn package to define our model and loss function
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim)
)
loss_fn = torch.nn.MSELoss(size_average = False)

# Use the optim package to define an Optimizer to update our weights for us. In this case, Adam.
# The first argument to an Optimizer's constructor tells it which Tensors to update
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 500
for t in range(epochs):
    # Forward pass: compute predicted y
    y_pred = model(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(t + 1, loss.item())

    # Before the backward pass, use the Optimizer to zero out the stored gradients for 
    # the variables it's supposed to update.
    optimizer.zero_grad()

    # Backward pass: compute gradient of loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
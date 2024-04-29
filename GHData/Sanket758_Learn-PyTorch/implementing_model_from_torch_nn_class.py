"""
Description: Implementing function for training loop in pytorch to understand how machine learning works 
            under the hood but this time using pytorch for backward pass

There are 3 major steps in learning:
    1. Forward pass - Here inputs and weights are sent through the network to get a prediction. prediction can be incorrect at the start but 
                    it will improve eventually as network learns.
    2. Calculate local gradients - Here network will calculate gradient at each node of the network according to its input and output
    3. Backward pass - Here we will use 'chain rule' to calculate overall loss for the whole network from local gradients
    4. Update weights - Optimizer step - We take one step of the optimizer in order to optimize the loss

We will be using torch.nn.Linear layer to achieve this.
"""

import torch
import torch.optim
import torch.nn

# Notice something different here? Well for pytorch's Linear layer to work, we need the input
# in the shape of n_samples*n_features. Here we have 4 samples with 1 feature. 
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Define our model
n_samples, n_features = X.shape  # (4, 1)
Xtest = torch.tensor([5], dtype=torch.float32)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # Define out layers
        self.linear_model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_model(x)

model = LinearRegression(n_features, 1) # As we are only predicting one class

# Hyper parameters
epochs = 20
learning_rate = 0.01

# Define out optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = torch.nn.MSELoss()

print(f"Prediction before training: f(5) = {model.forward(Xtest).item()}")

for epoch in range(epochs):
    # Prediction
    y_hat = model.forward(X)

    # Calculate loss
    l = loss(y_hat, y)

    #Backward pass
    l.backward()

    #Take optimizer step
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch+1}: w = {w[0][0].item()} loss = {l.item()}")

print(f"Prediction before training: f(5) = {model.forward(Xtest).item()}")

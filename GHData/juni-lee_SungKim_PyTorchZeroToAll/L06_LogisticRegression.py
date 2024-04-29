"""
1. Design your model using class
2. Construct loss and optimizer (select from PyTorch API)
3. Training cycle (forward, backward, update)
"""

import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0.], [0.], [1.], [1.]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()

# Creates a criterion that measures the Binary Cross Entropy between the target and the output
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = torch.Tensor([1.0])
print("predict 1 hour ", 1.0, model(hour_var).item() > 0.5)
hour_var = torch.Tensor([7.0])
print("predict 7 hour ", 7.0, model(hour_var).item() > 0.5)
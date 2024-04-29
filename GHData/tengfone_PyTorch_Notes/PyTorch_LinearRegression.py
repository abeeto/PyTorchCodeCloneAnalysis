# PyTorch Steps
# 1) Design model using class with VARIABLES
# 2) Construct loss and optimizer (PyTorch API)
# 3) Training Cycle (Forward, Backward, Update)

import torch
from torch.autograd import Variable

# Data definition (3x1)
x_data = Variable(torch.Tensor([[1],[2],[3]]))
y_data = Variable(torch.Tensor([[2],[4],[6]]))


# Step 1
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1) # Input size 1 (1 by 1), Output size 1 (1 y value)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

# Step 2
criterion = torch.nn.MSELoss(size_average = False)
# SGD = Stochastic Gradient descent algorithm, LR = learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Step 3
for i in range (500):
    # Forward
    y_pred = model(x_data)

    # Compute loss
    loss = criterion(y_pred, y_data)
    print(i,loss.data.item())

    # Backward pass and reset gradient and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[4]]))
print("After training: ", 4, model.forward(hour_var).data[0][0])

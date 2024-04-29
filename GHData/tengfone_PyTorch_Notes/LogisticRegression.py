from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

# Training data, y data uses binary logistic regression
x_data = tensor([[1], [2], [3], [4]])
y_data = tensor([[0], [0], [1], [1]])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 1 input 1 output
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred

# Model
model = Model()

criterion = nn.BCELoss(reduction= 'mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# Training loop
for i in range (1000):
    # Forward
    y_pred = model(x_data)

    # Compute
    loss = criterion(y_pred,y_data)

    # Zero gradient and backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
print('Predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')

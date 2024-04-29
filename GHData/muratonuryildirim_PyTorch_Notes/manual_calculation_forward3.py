import torch
import torch.nn as nn

# function: f=2x
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


model = LinearRegression(input_dim=1, output_dim=1)

print(f'Prediction before training for f(5) {model(x_test).item()}')

learning_rate = 0.01
n_iters = 50

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for epoch in range(n_iters):
    [w, b] = model.parameters()
    # forward pass
    y_pred = model(x)
    # loss
    l = loss(y, y_pred)
    # backward pass
    l.backward()
    # update weight(s)
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch {epoch} : w = {w}, loss= {l}')
    print(f'Prediction after training for f(5) {model(x_test).item()}')

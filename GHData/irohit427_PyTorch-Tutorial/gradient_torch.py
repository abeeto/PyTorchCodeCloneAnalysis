import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([1, 4, 6, 8], dtype=torch.float32)
X = X.view(4, 1)
y = y.view(4, 1)
X_test = torch.tensor([5], dtype=torch.float32)

n_sample, n_features = X.shape
output = 1

class LinearRegression(nn.Module):
  
  def __init__(self, input_dim, output_dim):
    super(LinearRegression, self).__init__()
    
    self.linear = nn.Linear(input_dim, output_dim)
  
  def forward(self, x):
    return self.linear(x)

learning_rate = 1e-2
iterations = 40
loss = nn.MSELoss()
model = LinearRegression(n_features, output);
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(iterations):
  y_pred = model(X)
  
  l = loss(y, y_pred)
  
  l.backward()
  
  optimizer.step()
  optimizer.zero_grad()
  
  if epoch % 8 == 0:
    print(f'epoch {epoch+1}: loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
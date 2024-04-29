########## PyTorch funcions#################

# 1. Design model (input, output, forward pass)
# 2. Constract loss function
# 3. Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights
import  torch
import  torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)


n_sample, n_features = X.shape

input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
n_iters = 1000
learning_rate = 0.001
loss = nn.MSELoss()

class LinearRegression(nn.Module):
    def __init__(self, input_dim, optput_dim):
        super(LinearRegression,self).__init__()
        self.lin = nn.Linear(input_dim, optput_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #Gradients
    l.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch+1}: w= {w[0][0].item():.3f}, loss= {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
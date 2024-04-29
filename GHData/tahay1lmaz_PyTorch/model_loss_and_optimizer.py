#%% Training Pipeline: Model, Loss and Optimizer
# We implemented logistic regression from scratch in this section. Calculations were made as follows ;
# -Prediction: PyTorch Model
# -Gradients Computation: Autograd
# -Loss Computation: PyTorch Loss
# -Parameter Updates: PyTorch Optimizer
import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# Other way to calculate the Linear Regression
# class LinearRegression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         #define the layers
#         self.lin = nn.Linear(input_dim, output_dim)
        
#     def forward(self, x):
#         return self.lin(x)
    
# model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters = 10 

loss = nn.MSELoss() # Mean Square Error is a callable function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step() # Will do optimization step automatically
        
    # zero gradients
    optimizer.zero_grad() # before the next iteration, we want to make sure our gradients are zero again
    
    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

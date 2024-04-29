#%% Training Pipeline: Model, Loss and Optimizer
# We implemented logistic regression from scratch in this section. Calculations were made as follows ;
# -Prediction: Manually
# -Gradients Computation: Autograd
# -Loss Computation: PyTorch Loss
# -Parameter Updates: PyTorch Optimizer
import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 10 

loss = nn.MSELoss() # Mean Square Error is a callable function
optimizer = torch.optim.SGD([w], lr=learning_rate) 

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    optimizer.step() # Will do optimization step automatically
        
    # zero gradients
    optimizer.zero_grad() # before the next iteration, we want to make sure our gradients are zero again
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')

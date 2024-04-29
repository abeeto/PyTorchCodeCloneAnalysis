#%% Gradient Descent with Autograd and Backpropagation with pytorch
# We implemented logistic regression from scratch in this section. Calculations were made as follows;
# -Prediction: Manually
# -Gradients Computation: Autograd
# -Loss Computation: Manually
# -Parameter Updates: Manually
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 10 

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw
    
    # update weights
    with torch.no_grad(): # we used this statement,
        w -= learning_rate * w.grad # because this operation should not be part of our gradient tracking graph 
        
    # zero gradients
    w.grad.zero_() # before the next iteration, we want to make sure our gradients are zero again
    
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')

#%% Gradient Calculation with Autograd
import torch

x = torch.randn(3, requires_grad=True) # requires_grad will be give grad_fn
print(x)

y = x + 2 # grad_fn=<AddBackward0>
z = y*y*2 # grad_fn=<MulBackward0>
#z = z.mean() # grad_fn=<MeanBackward0>

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32) # if we couldn't give an argument, we must give it the vector
z.backward(v) # dz/dx , grad can be implicitly created only for scalar outputs
print(x.grad) # gradients of this tensor
#%% 3 Ways to stop creating gradient functions
x = torch.randn(3, requires_grad=True)
print(x)

x.requires_grad_(False) # This modify the variable and require grad attribute doesn't show anymore
y = x.detach() # It creates a new tensor with same values but it doesn't require the gradient
with torch.no_grad(): # 3th way
    y = x+2
    print(y)
#%%
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_() # Before the next operation or the next iteration in our optimization steps, we must empty our gradient. We must call this function
    
#optimizer = torch.optim.SGD(weights, lr=0.01)
#optimizer.step()
#optimizer.zero_grad()

#%% Backpropagation Example
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)
# 1-
# Prediction            : Manually
# Gradients Computation : Manually
# Loss Computation      : Manually
# Parameter Updates     : Manually

# 2-
# Prediction            : Manually
# Gradients Computation : Autograd
# Loss Computation      : Manually
# Parameter Updates     : Manually

# 3-
# Prediction            : Manually
# Gradients Computation : Autograd
# Loss Computation      : Pytorch Loss
# Parameter Updates     : pytorch Optimizer

# 4-
# Prediction            : Pytorch Model
# Gradients Computation : Autograd
# Loss Computation      : Pytorch Loss
# Parameter Updates     : pytorch Optimizer
#######################################################
"""
# Number 1
import numpy as np

#f=w*x
#f=2*x
x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([2,4,6,8],dtype=np.float32)
w=0.0

#model prediction
def forward(x): 
    return w*x
#loss
def loss (y,y_predicted):
    return ((y_predicted-y)**2).mean()
#gradient
# MSE=1/N*(w*x-y)**2
#dJ/dw=1/N*2x*(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f"Prediction before training: f(5)= {forward(5):.3f}")

#training
learning_rate=0.01
n_iters=20
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    
    #Loss
    l=loss(y,y_pred)
    
    #gradients
    dw=gradient(x,y,y_pred)
    
    #update weights
    w-=learning_rate*dw
    
    if epoch%1==0:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={l:.8f}")
        
print(f"Prediction after training f(5)={forward(5):.3f}")
"""
#######################################################
""""
# Number 2
import torch

#f=w*x
#f=2*x
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

#model prediction
def forward(x): 
    return w*x
#loss
def loss (y,y_predicted):
    return ((y_predicted-y)**2).mean()

print(f"Prediction before training: f(5)= {forward(5):.3f}")

#training
learning_rate=0.01
n_iters=70
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    
    #Loss
    l=loss(y,y_pred)
    
    #gradients=backward pass
    l.backward() #dl/dw
    
    #update weights
    with torch.no_grad():
        w-=learning_rate*w.grad
    #zero Gradients
    w.grad.zero_()
    
    if epoch%2==0:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={l:.8f}")
        
print(f"Prediction after training f(5)={forward(5):.3f}")
"""
#######################################################
"""""
# Number 3
import torch
import torch.nn as nn
#f=w*x
#f=2*x
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

#model prediction
def forward(x): 
    return w*x

print(f"Prediction before training: f(5)= {forward(5):.3f}")

#training and loss
learning_rate=0.01
n_iters=70

loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=learning_rate)

for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(x)
    
    #Loss
    l=loss(y,y_pred)
    
    #gradients=backward pass
    l.backward() #dl/dw
    
    #update weights
    optimizer.step()
    #zero Gradients
    optimizer.zero_grad()
    
    if epoch%2==0:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={l:.8f}")
        
print(f"Prediction after training f(5)={forward(5):.3f}")
"""""
#######################################################
# Number 4
import torch
import torch.nn as nn
#f=w*x
#f=2*x
x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

x_test=torch.tensor([5],dtype=torch.float32)
n_spamples,n_features=x.shape
print(n_spamples,n_features)

#model prediction
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)

class LinearRegresison(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegresison,self).__init__()
        #define layers
        self.lin=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.lin(x)
model=LinearRegresison(input_size,output_size)
        

print(f"Prediction before training: f(5)= {model(x_test).item():.3f}")

#training and loss
learning_rate=0.01
n_iters=100 

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(x)
    
    #Loss
    l=loss(y,y_pred)
    
    #gradients=backward pass
    l.backward() #dl/dw
    
    #update weights
    optimizer.step()
    #zero Gradients
    optimizer.zero_grad()
    
    if epoch%10==0:
        [w,b]=model.parameters()
        print(f"epoch {epoch+1}: w={w[0][0].item():.3f}, loss={l:.8f}")
        
print(f"Prediction after training f(5)={model(x_test).item():.3f}")

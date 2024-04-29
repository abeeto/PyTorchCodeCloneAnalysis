#1) Design model (input,out size,forward)
#3) Training loop
# -forward pass: compute prediction
# -backward pass:gradients
# -update weights
from pickletools import optimize
import torch
import torch.nn as nn

#f=w*x
#f=2*x
X= torch.tensor([1,2,3,4],dtype=torch.float32)
Y= torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# model prediction
def forward(x):
    return w*x

#Training
learning_rate=0.01
n_iters=100

loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=learning_rate)


print(f'Prediction before training: f(5)={forward(5):.3f}')



for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=forward(X)
    
    #loss
    l=loss(Y,y_pred)
    
    #gradinets=backprop
    l.backward()#dl/dw
    
    #update weights
    optimizer.step()
    #zero grdients
    optimizer.zero_grad()
    
    
    if epoch%10==0:
        print(f'epoch{epoch+1}:w={w:.3f},loss={l:.8f}')
        
print(f'Prediction after training:f(5)={forward(5):.3f}')

    
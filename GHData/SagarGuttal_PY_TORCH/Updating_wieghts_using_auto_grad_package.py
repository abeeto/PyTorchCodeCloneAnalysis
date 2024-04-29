import torch

# f = w * x
# f = 2 * x

X = torch.tensor([1,2,3,4,5], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model_prediction
def forward(X):
    return w * X

# Loss
def loss(y, y_prediction):
    return ((y_prediction-y)**2).mean()

#training
learning_rate = 0.01
n_iteration = 10

for epoch in range(n_iteration):
    
    # prediction
    y_pred = forward(X)
    #LOss
    l = loss(Y, y_pred)
    #gradient -- backpropogation
    l.backward() #this function calculates the gradients ---> dl/dw
    # update wieghts
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()
    if epoch % 2 == 0:
       print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.3f}")

print(f"Prediction after training : f(5) = {forward(5):.3f}")
print(f"Prediction after training : f(19)= {forward(19):.3f}")
    

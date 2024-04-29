import torch
import numpy as np
from matplotlib import pyplot as plt


D1 = 30
D2 = 20
N = 4000
ITER = 500
SWAP = False
NOISE = True
NORMALIZE = True
VAL_PERC = 0.3
learning_rate = 1e0


W_gt = torch.randn(D1, D2)
W_gt[:D2, :D2] += 10 * torch.eye(D2)
b_gt = 0.01

x = torch.randn(N, D1)
y = x @ W_gt + b_gt 

if NOISE:
    y += 0.1 * torch.randn(N, D2)

if SWAP:
    x_copy = x
    x = y
    y = x_copy

x_norm = 1
y_norm = 1
if NORMALIZE:
    x_norm = x.max()
    y_norm = y.max()
    x = x / x_norm
    y = y / y_norm

class LinearRegression(torch.nn.Module): 
    def __init__(self):
        super(LinearRegression, self).__init__() 
        if SWAP:
            self.linear = torch.nn.Linear(D2, D1, bias = True) # bias is default True
            self.linear2 = torch.nn.Linear(D1, D1 * 1, bias = True) # bias is default True
            self.linear3 = torch.nn.Linear(D1 * 1, D1, bias = True) # bias is default True
        else:
            self.linear = torch.nn.Linear(D1, D2, bias = True) # bias is default True
            self.linear2 = torch.nn.Linear(D2, D2 * 1, bias = True) # bias is default True
            self.linear3 = torch.nn.Linear(D2 * 1, D2, bias = True) # bias is default True

        self.activation1 = torch.nn.Tanh()
        self.activation2 = torch.nn.Tanh()
        self.activation3 = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation1(x)
        x = self.linear2(x)
        #x = self.activation2(x)
        #x = self.linear3(x)
        #x = self.activation2(x)
        return x

    def my_loss(self, output, target):
        loss1 = torch.mean((output - target)**2)
        #loss2 = torch.mean((self.linear.weight[0][0] - W_gt[0][0])**2)
        loss2 = 0#torch.mean((self.linear.weight[0][0])**2)
        loss2 = 0.1 * 1 / torch.trace(self.linear.weight)
        #loss2 = torch.norm(self.linear.weight)
        loss = loss1 + loss2
        return loss

x_train = x[:int(N*(1-VAL_PERC)),None]
y_train = y[:int(N*(1-VAL_PERC)),None]
x_val = x[:int(N*VAL_PERC),None]
y_val = y[:int(N*VAL_PERC),None]

model = LinearRegression()

criterion = torch.nn.MSELoss()
criterion = model.my_loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_arr = []
val_arr = []
def trainBuildIn(model, x, y, iter):
    for i in range(iter):
        model.train()
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
        
        # get output from the model, given the inputs
        y_pred = model(x_train)
        

        # get loss for the predicted output
        loss = criterion(y_pred, y_train)
        print(loss)
        loss_arr.append(loss.item())
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('Iter {} / {}, loss {}'.format(i, iter, loss.item()))

        model.eval()
        y_pred = model(x_val)
        loss = criterion(y_pred, y_val)
        val_arr.append(loss.item())

trainBuildIn(model, x_train, y_train, ITER)

y_pred_bi = model(x_val).data.numpy()

print("----- ----- ----- ----- -----")
print("Prediction:")
print(f"train loss: {round(loss_arr[-1], 4)}, val loss: {round(val_arr[-1], 4)}")

plt.figure()
plt.title("Validation")
plt.clf()
plt.plot(y[0], label='True data', alpha=0.5)
plt.plot(y_pred_bi[0][0], label='Predictions', alpha=0.5)
plt.legend(loc='best')

plt.figure()
plt.plot(loss_arr)
plt.plot(val_arr)

plt.figure()
plt.subplot(211)
if SWAP:
    plt.title("PINV-GT")
    plt.imshow(np.linalg.pinv(W_gt).T)
else:
    plt.title("GT")
    plt.imshow(W_gt.T)

plt.subplot(212)
plt.title("Model")
plt.imshow(model.linear.weight.detach())
plt.show()


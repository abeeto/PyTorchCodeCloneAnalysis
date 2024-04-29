# =============================================================================
# Import required libraries
# =============================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Generate data 
# =============================================================================
# y = a(x) + b + noise => a=2, b=7
np.random.seed(10)
x = np.random.rand(100, 1)
y = 2*x + 7 + 0.2*np.random.rand(100, 1)

# shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# uses first 80 random indices for train
trian_idx = idx[:80]
# uses the remaining indices for test
test_idx = idx[80:]

# generates train and test sets
x_train, y_train = x[trian_idx], y[trian_idx]
x_test, y_test = x[test_idx], y[test_idx]

# converts numpy arrays to tensors
x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()

# =============================================================================
# Plot data 
# =============================================================================
plt.scatter(x_train, y_train, c='b')
plt.title('Train data')
plt.grid()
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

plt.scatter(x_test, y_test, c='r')
plt.title('Test data')
plt.grid()
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.show()

# =============================================================================
# Define model
# =============================================================================
class LinearRegression(torch.nn.Module):
    def __init__(self): 
        # np.random.seed(20)
        # self.a = torch.randn(1, requires_grad=True)
        # self.b = torch.randn(1, requires_grad=True)
        super(LinearRegression, self).__init__()
        self.neuron = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.neuron(x)
        return x
        
    
net = LinearRegression()
print(net)

# =============================================================================
# Train model
# =============================================================================
epochs = 1000
lr = 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

train_losses = []

net.train()
for epoch in range(epochs):
    
    # zero the gradients parameter
    optimizer.zero_grad()
    
    # forward pass: compute predicted outputs by passing inputs to the model
    y_pred = net(x_train)
    
    # calculate the loss
    # error = y_train - y_pred
    # loss = (error ** 2).mean()
    loss = criterion(y_train, y_pred)
    
    # backward pass: compute gradient of the loss with respect to model parameters
    # a_grad = -2 * (error * x_train).mean()
    # b_grad = -2 * error.mean()
    loss.backward()
    
    # update parameters
    # net.set_a(net.get_a() - lr * a_grad) 
    # net.set_b(net.get_b() - lr * b_grad) 
    optimizer.step()
    
    train_losses.append(loss)
    print('Epoch: {} \t Training Loss: {:.3f}'.format(epoch+1, loss))
       
# print parameters
print(net.state_dict())

# =============================================================================
# Plot model 
# =============================================================================
plt.scatter(x_test, y_test, c='r')
plt.title('Test data')
plt.grid()
plt.xlabel('x_test')
plt.ylabel('y_test')
#
params = net.state_dict()
plt.plot(x_test, params.get('neuron.weight') * x_test + params.get('neuron.bias'), 'b')
plt.show()

# =============================================================================
# Plot train loss
# =============================================================================
plt.plot(range(epochs), train_losses, 'g')
plt.xlabel('epochs')
plt.ylabel('Training loss')
plt.show()
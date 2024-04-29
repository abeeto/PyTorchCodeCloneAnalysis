# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Generate data 
# =============================================================================
# y = a(x) + b + noise => a=2, b=7
np.random.seed(10)
x = np.random.rand(100)
y = 2*x + 7 + 0.2*np.random.rand(100)

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
class LinearRegression():
    def __init__(self):           
        np.random.seed(20)
        self.a = np.random.rand()
        self.b = np.random.rand()

    def forward(self, x):
        y = self.a * x + self.b
        return y
    
    def get_a(self):
        return self.a
    
    def get_b(self):
        return self.b
    
    def set_a(self, in_a):
        self.a = in_a
    
    def set_b(self, in_b):
        self.b = in_b

net = LinearRegression()

# =============================================================================
# Train model
# =============================================================================
epochs = 1000
lr = 0.1

train_losses = []

for epoch in range(epochs):
    
    # forward pass: compute predicted outputs by passing inputs to the model
    y_pred = net.forward(x_train)
    
    # calculate the loss
    error = y_train - y_pred
    loss = (error ** 2).mean()
    
    # backward pass: compute gradient of the loss with respect to model parameters
    a_grad = -2 * (error * x_train).mean()
    b_grad = -2 * error.mean()
    
    # update parameters
    net.set_a(net.get_a() - lr * a_grad) 
    net.set_b(net.get_b() - lr * b_grad) 
    
    train_losses.append(loss)
    print('Epoch: {} \t Training Loss: {:.3f}'.format(epoch+1, loss))
       
print(net.get_a())
print(net.get_b())

# =============================================================================
# Plot model 
# =============================================================================
plt.scatter(x_test, y_test, c='r')
plt.title('Test data')
plt.grid()
plt.xlabel('x_test')
plt.ylabel('y_test')
#
plt.plot(x_test, net.get_a()*x_test+net.get_b(), 'b')
plt.show()

# =============================================================================
# Plot train loss
# =============================================================================
plt.plot(range(epochs), train_losses, 'g')
plt.xlabel('epochs')
plt.ylabel('Training loss')
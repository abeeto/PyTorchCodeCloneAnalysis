# Define the network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Kernel
        self.conv1 = nn.Conv2d(1, 6, 3) # 1 input image channel, 6 output image channels, 3 * 3 square convolution
        self.conv2 = nn.Conv2d(6, 16, 3)

        # Affine operation (y = Wx + b)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling (2, 2) (if size is a square, single number is possible)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features

net = Net()
print(net)

# After define foward(), backward() is automatically defined using autograd
params = list(net.parameters())
print(len(params))
print(params[0].size())

# To use MNIST dataset, need to resize the images from the dataset to 32 * 32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backpropagation with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# Loss function (mean-squared error)
output = net(input)
target = torch.randn(10) # Random dummy target
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# When loss.backward() called, the whole graph is differenciated about loss, every tensor, has requires_grad = True, has accumulated gradient
print(loss.grad_fn) # MSE
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# To error backpropagation, only loss.backward() is needed, but need to clear the existing gradient to prevent accumulating
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weights (weight = weight - learning_rate * gradient)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# If the various update ruels are neede, use the small package named torch.optim
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01) # Create optimizer

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import time
import timeit

import matplotlib.pyplot as plt
#%%
# Define the autoencoder class:
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder:
        self.encL1 = nn.Linear(784, 1000)
        self.encL2 = nn.Linear(1000, 500)
        self.encL3 = nn.Linear(500, 250)
        self.encL4 = nn.Linear(250, 30)
        
        # decoder:
        self.decL1 = nn.Linear(30, 250)
        self.decL2 = nn.Linear(250, 500)
        self.decL3 = nn.Linear(500, 1000)
        self.decL4 = nn.Linear(1000, 784)

    def forward(self, x):
        x = F.relu(self.encL1(x))
        x = F.relu(self.encL2(x))
        x = F.relu(self.encL3(x))
        x = F.relu(self.encL4(x))
        x = F.relu(self.decL1(x))
        x = F.relu(self.decL2(x))
        x = F.relu(self.decL3(x))
        x = F.relu(self.decL4(x))
        return x
#%%
# Define a model and an optimizer:
net = AE()

# Adam optimizer with learning rate 1e-3:
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Mean-squared error loss as an objective function:
criterion = nn.MSELoss()

# Define parameters:
epochs = 20
#%%
# Load the data:
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)

test_data = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=100, shuffle=True, num_workers=0, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=10, shuffle=False, num_workers=0
)
#%%
# Train the network:
start = timeit.default_timer() # Initialize the timer to compute the time
start_cpu = time.process_time() # Initialize the cpu timer to compute time

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = net(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    
print('Finished Training')

stop = timeit.default_timer() # stop the time counting
stop_cpu = time.process_time()  # stop the cpu time counting
print ('Total time, sec: '+str(stop-start))
print ('Time cpu, sec: '+str(stop_cpu-start_cpu))
#%%
# Visualization of the results:
test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784)
        reconstruction = net(test_examples)
        break
    
with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
#%%
# Training on GPU:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# move all modules to CUDA tensors:
net.to(device)
#%%
# Train the network on the GPU:
start = timeit.default_timer() # Initialize the timer to compute the time
start_cpu = time.process_time() # Initialize the cpu timer to compute time

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = net(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    
print('Finished Training')

stop = timeit.default_timer() # stop the time counting
stop_cpu = time.process_time()  # stop the cpu time counting
print ('Total time, sec: '+str(stop-start))
print ('Time cpu, sec: '+str(stop_cpu-start_cpu))
#%%
# Visualization of the results:
test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.view(-1, 784).to(device)
        reconstruction = net(test_examples)
        break

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
#%%
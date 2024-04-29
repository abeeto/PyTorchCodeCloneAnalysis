####################################################################
# Activation Functions: Sigmoid, Tanh and ReLU
####################################################################
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
torch.manual_seed(2)


# Create a tensor
z = torch.arange(-10, 10, 0.1).view(-1, 1)

# Create a sigmoid object
sig = nn.Sigmoid()
# Create a tanh object
tanh = nn.Tanh()
# Create a relu object
relu = nn.ReLU()

# Make a prediction of each function
y_sig = sig(z)
y_tanh = tanh(z)
y_relu = relu(z)


plt.figure()
plt.plot(z.detach().numpy(), y_sig.numpy(), label='Sigmoid')
plt.plot(z.detach().numpy(), y_tanh.numpy(), label='Tanh')
plt.plot(z.detach().numpy(), y_relu.numpy(), label='ReLU')
plt.legend()
plt.xlabel('z')
plt.ylabel('Prediction')
plt.show()

# Make a prediction using build-in function
x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.figure()
plt.plot(x.detach().numpy(), torch.sigmoid(x).numpy(), label='Sigmoid')
plt.plot(x.detach().numpy(), torch.tanh(x).numpy(), label='Tanh')
plt.plot(x.detach().numpy(), torch.relu(x).numpy(), label='ReLU')
plt.legend()
plt.xlabel('x')
plt.ylabel('Prediction')
plt.show()


####################################################################
# Test Sigmoid, Tanh and ReLU Activation Functions
####################################################################
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np


# Build the model with sigmoid function
class Net_sig(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_sig, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.sigmoid(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Build the model with Tanh function
class Net_tanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_tanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.tanh(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Build the model with ReLU function
class Net_relu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net_relu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y1 = torch.relu(self.linear1(x))
        yout = self.linear2(y1)
        return yout


# Define the function for training the model
def train(model, criterion, optimizer, train_loader,
          validation_loader, epochs=100):
    i = 0
    results = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        # Training
        for i, (_x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = model(_x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            results['training_loss'].append(loss.item())

        # Validation
        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28*28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * correct / len(validation_loader)
        results['validation_accuracy'].append(accuracy)
        return results


# Create training dataset
train_dataset = dsets.MNIST(root='./data', train=True,
                            download=True, transform=transforms.ToTensor())
# Create validation dataset
validation_dataset = dsets.MNIST(root='./data', train=False,
                                 download=True, transform=transforms.ToTensor())

# Create criterion function
criterion = nn.CrossEntropyLoss()
# Create the training data loader and validation data loader object
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=5000, shuffle=False)

# Create the network structure
input_dim = 28 * 28
hidden_dim = 10
output_dim = 10
learning_rate = 0.01

# Train model with sigmoid function
model_sig = Net_sig(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model_sig.parameters(), lr=learning_rate)
training_results_sig = train(model_sig, criterion, train_loader,
                             validation_loader, optimizer, epochs=30)

# Train model with Tanh function
model_tanh = Net_tanh(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
training_results_tanh = train(model_tanh, criterion, train_loader,
                              validation_loader, optimizer, epochs=30)

# Train model with ReLU function
model_relu = Net_relu(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
training_results_relu = train(model_relu, criterion, train_loader,
                              validation_loader, optimizer, epochs=30)

# Compare the training loss
plt.plot(training_results_tanh['training_loss'], label='tanh')
plt.plot(training_results_sig['training_loss'], label='sigmoid')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()

# Compare the validation loss
plt.plot(training_results_tanh['validation_accuracy'], label='tanh')
plt.plot(training_results_sig['validation_accuracy'], label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label='relu')
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')
plt.legend()
plt.show()

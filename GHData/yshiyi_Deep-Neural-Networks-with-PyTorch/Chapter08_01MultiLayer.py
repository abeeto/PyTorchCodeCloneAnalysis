####################################################################
# Multiple Hidden Layer Deep Network
# In this example, there are two hidden layers
####################################################################
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)


# Create the model using Sigmoid
class Net_sig(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net_sig, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        y1 = torch.sigmoid(self.linear1(x))
        y2 = torch.sigmoid(self.linear2(y1))
        yout = self.linear3(y2)
        return yout


# Create the model using Tanh
class Net_tanh(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net_tanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        y1 = torch.tanh(self.linear1(x))
        y2 = torch.tanh(self.linear2(y1))
        yout = self.linear3(y2)
        return yout


# Create the model using ReLU
class Net_relu(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net_relu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        y1 = torch.relu(self.linear1(x))
        y2 = torch.relu(self.linear2(y1))
        yout = self.linear3(y2)
        return yout


# Train the model
def train(model, criterion, optimizer, train_loader,
          validation_loader, epochs=100):
    i = 0
    results = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            results['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()

        accuracy = 100 * (correct / len(validation_dataset))
        results['validation_accuracy'].append(accuracy)

    return results


# Create the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())
# Create the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True,
                                 transform=transforms.ToTensor())
# Create the criterion function
criterion = nn.CrossEntropyLoss()
# Create the training data loader and validation data loader object
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=5000, shuffle=False)

# Set the parameters for create the model
input_dim = 28 * 28
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10
# Set the number of iterations
cust_epochs = 10


# Train the model with sigmoid function
learning_rate = 0.01
model_sig = Net_sig(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer_sig = torch.optim.SGD(model_sig.parameters(), lr=learning_rate)
results_sig = train(model_sig, criterion, train_loader, validation_loader,
                    optimizer_sig, epochs=cust_epochs)

# Train the model with tanh function
learning_rate = 0.01
model_tanh = Net_tanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
results_tanh = train(model_tanh, criterion, train_loader, validation_loader,
                     optimizer_tanh, epochs=cust_epochs)

# Train the model with ReLU function
learning_rate = 0.01
model_relu = Net_relu(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
results_relu = train(model_relu, criterion, train_loader, validation_loader,
                     optimizer_relu, epochs=cust_epochs)

# Compare the training loss
plt.figure()
plt.plot(results_tanh['training_loss'], label='tanh')
plt.plot(results_sig['training_loss'], label='sigmoid')
plt.plot(results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

# Compare the validation loss
plt.figure()
plt.plot(results_tanh['validation_accuracy'], label = 'tanh')
plt.plot(results_sig['validation_accuracy'], label = 'sigmoid')
plt.plot(results_relu['validation_accuracy'], label = 'relu')
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')
plt.legend()

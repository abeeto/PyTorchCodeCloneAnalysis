import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook'])
plt.rcParams['axes.linewidth'] = 2
plt.rc('axes', edgecolor='black')

class NeuralNetwork(nn.Module):
    """Network Structure"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        a = self.layer1(x)
        a = self.relu(a)
        a = self.layer2(a)
        return a

X = np.loadtxt("./data/iris.data", delimiter=",", usecols=[0,1,2,3], dtype=np.float32) # samples

y = np.concatenate((np.zeros(50), np.ones(50), np.full(50, 2))) # classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # Split up data

X_train, X_test, y_train, y_test = torch.from_numpy(X_train), torch.from_numpy(X_test), \
                                   torch.from_numpy(y_train), torch.from_numpy(y_test) # Convert to tensors

input_size, hidden_size, num_classes = 4, 10, 3
lr, num_epochs = 0.1, 1000

model = NeuralNetwork(input_size, hidden_size, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_values = []

# Training
for epoch in range(num_epochs):
    # Forward pass and loss 
    output = model(X_train)
    loss = loss_fn(output, y_train.long())

    # Backpass and updating weights    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        loss_values.append(loss.item())
        print(f'Epoch {epoch+1}/{num_epochs} completed, loss = {loss.item():.4f}')

# Plot of loss
x = np.arange(5, num_epochs+1, 5)
plt.figure(figsize=(8.88,5), dpi=160)
plt.semilogy(x, loss_values, linewidth=3, color='k')
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.title("Loss on Training Data", fontsize=24)
plt.xlim(min(x), max(x))
plt.grid(linestyle='dashed', linewidth=1.5)
plt.show()

# Testing and accuracy
num_correct = 0
num_samples = len(y_test)
output = model(X_test)
value, index = torch.max(output, 1)
num_correct = (index == y_test).sum().item() 
acc = 100 * num_correct / num_samples
print(f'\nAccuracy on test data: {acc:.2f}%')


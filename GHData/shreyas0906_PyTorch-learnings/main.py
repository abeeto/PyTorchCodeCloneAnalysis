import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
    """
    Feed forward network
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64) # input , output
        self.fc2 = nn.Linear(64, 64) # input from previous layer, output of the current layer
        self.fc3 = nn.Linear(64, 64) # input from previous layer, output of the current layer
        self.fc4 = nn.Linear(64, 10) # 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1) # dim=0 corresponds to batches, dim=1 corresponds to the actual output



net = Net()
print(net)
X = torch.rand((28,28))
X = X.view(-1, 28*28) # unknown shape input
output = net(X)
optimizer = optim.Adam(net.parameters(), lr=0.001) # The parameters of the network to be adjusted.
EPOCHS = 3
# Epochs is the number of times the entire train data will be passed to the model.
for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad(): # This is for testing, we are telling the model to not to calculate any kind of gradients and just use the trained model for inference
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i  in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print(f" accuracy: {correct / total}")

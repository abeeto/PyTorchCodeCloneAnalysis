import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# set parameters
learning_rate = 0.001
batch_size = 100

# get dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
# set data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# build model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0), -1) # out.size(0) == batch_size
        out = self.fc(out)
        return out

model = CNN().to(device)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# learning
nb_epochs = 15
batches = len(data_loader)
for epoch in range(nb_epochs):
    avg_cost = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        # hypothesis
        hypothesis = model(X)
        # cost
        cost = F.cross_entropy(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / batches
    print(f'epoch: {epoch+1:2d}/{nb_epochs} Cost: {avg_cost:.6f}')
print('Learning completed')

# accuracy
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # hypothesis
    hypothesis = model(X_test)
    # accuracy
    isCorrect = torch.argmax(hypothesis, dim=-1) == Y_test
    accuracy = torch.mean(isCorrect.float())
    print(f'Accuracy: {accuracy*100:.3f}%')
# 98.650%로 높은 정확도를 보임.
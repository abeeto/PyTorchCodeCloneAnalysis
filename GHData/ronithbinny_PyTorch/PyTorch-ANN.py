import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
                       
test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
                       
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28,28))
plt.show()

import torch.nn as nn
import torch.functional as F
#torch.nn.functional.relu

class Net(nn.Module) :

  def __init__(self) :
    super().__init__()
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)

  def forward(self,x) :
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = torch.nn.functional.relu(self.fc3(x))
    x = self.fc4(x)

    return torch.nn.functional.log_softmax(x, dim = 1)
    
net = Net()
print(net)

import torch.optim as optim

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(2):
  for data in trainset :
    X, y = data
    net.zero_grad()
    output = net(X.view(-1,28*28))
    loss = loss_function(output, y)
    #loss = torch.nn.functional.nll_loss(output, y)
    loss.backward()
    optimizer.step()
  print(loss)
  
  correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 4))

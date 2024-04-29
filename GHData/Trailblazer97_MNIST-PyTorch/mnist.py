import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
                                
train_data = datasets.MNIST(root = "./data", transform = transform, train = True, download = True)
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size = 64)


model = nn.Sequential(nn.Linear(784, 512),
                      nn.ReLU(),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim = 1))

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.025)
epochs = 10

for i in range(epochs):
  running_loss = 0
  for step, (images, labels) in enumerate(trainloader):

    optimizer.zero_grad()
    inputs = images.view(images.shape[0], -1)
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    
    running_loss += loss.item()
    
    loss.backward()
    optimizer.step()
  else:
    print(running_loss/len(trainloader))


test_data = torchvision.datasets.MNIST(root = "./data", train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size = 64)


total = 0
count = 0
for steps, (images, labels) in enumerate(testloader):
  inputs = images.view(images.shape[0], -1)
  output = model.forward(inputs)
  val, index = output.max(1)
  total += abs(torch.sum(labels-index))
  # print(total)
  count += 1

print("Accuracy - ", 1-total.numpy()/(count*64))

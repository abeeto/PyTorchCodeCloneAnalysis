import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms

transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )
# transforms.ToTensor(): 將[0-255]的image轉換成(C,H,W)


trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

# Module
class NeuralNetworkModule(nn.Module):
    def __init__(self):
        super(NeuralNetworkModule, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        return self.main(input)

net=NeuralNetworkModule()
print(net)

# Hyper Parameter
total_epoch = 3 # 訓練的迭代次數
learningRate = 0.002 # Learning Rate
criterion = nn.CrossEntropyLoss() # Loss Function
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9) # 優化器

# Train
for epoch in range(total_epoch):
    runningLoss=0.0
    for times, data in enumerate(trainLoader):
        input = data[0]
        label = data[1]

        input = input.view(input.shape[0], -1) # 壓縮維度到784

        optimizer.zero_grad()

        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        runningLoss = runningLoss + loss.item()
        if times % 100 == 99 or times + 1 == len(trainLoader):
            print('[%d %d, %d %d] loss:%.3f' % (epoch+1, total_epoch, times+1, len(trainLoader), runningLoss/1000))

print("Training Finished!!")

# Test
correct=0
total=0

with torch.no_grad():
    for data in testLoader:
        input, label = data
        input=input.view(input.shape[0], -1)

        output = net(input)
        _, predicted = torch.max(output.data, 1)
        total = total+label.size(0)
        correct = correct + (predicted == label).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

class_correct = [0]*10
class_total = [0]*10

with torch.no_grad():
    for data in testLoader:
        input, labels=data[0], data[1]
        input=input.view(input.shape[0], -1)

        output=net(input)

        _, predicted = torch.max(output, 1)
        c=(predicted==labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] = class_correct[label]+c[i].item()
            class_total[label] = class_total[label] + 1
            print(class_correct)
            print(class_total)

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))

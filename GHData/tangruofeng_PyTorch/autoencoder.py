import torch
import torch.nn as nn
from torch.nn.modules import Module
from torchvision import transforms
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#trainset和testset两两一组，上方一组是MNIST数据集，下方一组是FashionMNIST数据集，使用时注释掉另一组即可
#trainset = datasets.MNIST(root='MNIST',train=True, download=True,transform=transforms.ToTensor())
#testset = datasets.MNIST(root='MNIST',train=False,download=True,transform=transforms.ToTensor())
trainset = datasets.FashionMNIST(root='MNIST',train=True, download=True,transform=transforms.ToTensor())
testset = datasets.FashionMNIST(root='MNIST',train=False,download=True,transform=transforms.ToTensor())
#将原始的训练集按照7：3的比例划分维训练集和验证集
trainsetloader, validsetloader = random_split(dataset=trainset, lengths=[42000, 18000], generator=torch.Generator().manual_seed(0))
trainloader = DataLoader(trainsetloader,batch_size=100,shuffle=True)
validloader = DataLoader(validsetloader,batch_size=100,shuffle=True)
testloader = DataLoader(testset,batch_size=100,shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())
cost = torch.nn.CrossEntropyLoss()
epoches = 10

lossList = []
accuList = []
timeList = []
validList = []

for epoch in range(epoches):
    sum_loss = 0.0
    train_correct = 0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = Variable(inputs.view(-1, 28*28)).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        train_correct += torch.sum(id == labels.data)
    lossList.append(sum_loss)
    accuList.append(100 * train_correct / len(trainsetloader))
    timeList.append(epoch)
    print('[%d, %d] loss:%.03f' % (epoch + 1, epoch, sum_loss / len(trainloader)))
    print("correct:%.3f%%" % (100 * train_correct / len(trainsetloader)))

    validLoss = 0.0
    for data in validloader:
        loss = cost(outputs, labels)
        validLoss += loss.data
    validList.append(validLoss)

model.eval()
test_correct = 0
for data in testloader:
    inputs, labels = data
    inputs, labels = Variable(inputs.view(-1, 28*28)).cuda(), Variable(labels).cuda()
    outputs = model(inputs)
    _, id = torch.max(outputs.data, 1)
    test_correct += torch.sum(id == labels.data)
print("\n\n")
print("TestSet correct:%.3f%%" % (100 * test_correct / len(testset)))
print("\n\n")

plt.figure()
plt.subplot(1,3,1)
plt.plot(timeList, torch.Tensor(accuList).cpu())
plt.title('TrainSet accuracy')
plt.legend()
plt.subplot(1,3,2)
plt.plot(timeList, torch.Tensor(lossList).cpu())
plt.title('TrainSet loss Function')
plt.legend()
plt.subplot(1,3,3)
plt.plot(timeList, torch.Tensor(validList).cpu())
plt.title('validSet loss Function')
plt.legend()
plt.show()
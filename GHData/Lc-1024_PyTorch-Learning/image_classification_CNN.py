# 图像分类
# 用torchvision.datasets中的MNIST/FashionMNIST作为数据集
# 用tud的dataloader处理数据，转换为迭代器
# 模型是最简单的CNN模型，用两层卷积层和两层线性回归层
# 由SGD优化，用F.nll_loss计算损失


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# 固定初始化种子
SEED = 24
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(SEED)


BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.5
NUM_EPOCHS = 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 channel -> 20 channels
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # 28 * 28 -> (28+1-5) = 24 * 24
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        # x: batch_size * 1 * 28 * 28
        x = F.relu(self.conv1(x)) # batch_size * 20 * 24 * 24
        x = F.max_pool2d(x,2,2) # batch_size * 20 * 12 * 12
        x = F.relu(self.conv2(x)) # batch_size * 50 * 8 * 8
        x = F.max_pool2d(x,2,2) # batch_size * 50 * 4 *4 
        x = x.view(-1, 4*4*50) # batch_size * (50*4*4) 
        x = F.relu(self.fc1(x))
        x= self.fc2(x)
        # return x
        return F.log_softmax(x, dim=1) # log probability
        

# MNIST/FashionMNIST有60,000张图片组成训练集
# 图片形状为1*28*28，对应着0-9的数字/10种服装服饰
# 另有10,000张组成的测试集
train_data = datasets.MNIST(".data", train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))
# 获取训练集的平均值和标准差
data = [d[0].data.cpu().numpy() for d in train_data]
train_mean = np.mean(data)
train_std = np.std(data)

# 用tud.dataloader来处理数据
# 测试集的normalize要和训练集一样
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(".data", train=True, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               # Normalize输入为两个tuple，output=(input-mean)/std
               transforms.Normalize((train_mean,), (train_std,)) # (x,)输出为一维tuple
           ])),
    batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=0, pin_memory=False
)
test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(".data", train=False, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               # Normalize输入为两个tuple，output=(input-mean)/std
               transforms.Normalize((train_mean,), (train_std,)) # (x,)输出为一维tuple
           ])),
    batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=0, pin_memory=False
)
print("Dataloader successfully loaded.")

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data) # batch_size * 10
        loss = F.nll_loss(pred, target) 

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, i, loss.item()))

def test(model, device, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data) # batch_size * 10
            total_loss += F.nll_loss(output, target, reduction="sum").item() 
            pred = output.argmax(dim=1) # batch_size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct/len(test_loader.dataset) * 100.
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))


# 进行模型的训练，由SGD优化，用F.nll_loss计算损失
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
for epoch in range(NUM_EPOCHS):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)
    

# torch.save(model.state_dict(), "image_classification_CNN_MNIST.pt")


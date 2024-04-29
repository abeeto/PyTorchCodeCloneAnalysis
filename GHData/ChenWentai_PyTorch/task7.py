import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from time import time
#定义batch size
batch_size = 64

#下载MNIST数据集
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(), 
                               download=True)
test_dataset = datasets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

#将下载的MNIST数据导入到dataloader中
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

#数据可视化
plt.figure()
plt.imshow(train_loader.dataset.train_data[0].numpy())
plt.show()

#搭建神经网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.tanh(self.conv1(x)), (2,2))
#         x = F.dropout(x, p=0.3, training = self.training)
        x = F.max_pool2d(F.tanh(self.conv2(x)), (2,2))
#         x = F.dropout(x, p=0.3, training = self.training)
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.fc1(x))
#         x = F.dropout(x, p=0.3, training = self.training)
        x = F.tanh(self.fc2(x))
#         x = F.dropout(x, p=0.3, training = self.training)
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
model = LeNet5()
print(model)

#训练神经网络
def train(model, num_epoch):
    Loss = []
    model.train(True)
    for i in range(num_epoch):
        running_loss = 0
        running_corrects = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            running_corrects += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            running_loss += loss.data.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.data.item() / len(train_dataset)
        print("Epoch:", i,"running_loss:", running_loss, "Loss:", epoch_loss)
        print("Epoch:", i,"running_corrects:", running_corrects.data.item(), "acc:", epoch_acc)
        Loss.append(epoch_loss)
    return Loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()
a = time()
LOSS = train(model, 30)    
b = time() - a
print("training time:{}s".format(b))

#可视化训练结果
plt.plot(LOSS)
plt.xlabel('epoch')
plt.ylabel('loss')

##测试神经网络

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #将每个batch的loss加和
        test_loss += criterion(output, target).data.numpy()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print("test_loss:",test_loss)
    acc = correct.item()/len(test_dataset)
    print("test accuracy:", acc)
test()
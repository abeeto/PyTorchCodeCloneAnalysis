from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
from model import  MobileNetV1 , MobileNetV1_075, MobileNetV1_050, MobileNetV1_025
from DataLoader import DogvsCatDataSet
from torchvision import transforms
import torch.utils.data
import matplotlib.pyplot as plt

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 50  # 训练次数
batch_size = 16  # 批处理大小
num_workers = 4  # 多线程的数目
use_gpu = torch.cuda.is_available()

#test_dir="./archive/test_set/test_set"
#train_dir="./archive/training_set/training_set"

test_dir = "./dc_2000/test"
train_dir = "./dc_2000/train"
# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
])


train_dataset = DogvsCatDataSet(train_dir, transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dataset = DogvsCatDataSet(test_dir, transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# 加载resnet18 模型，
net = MobileNetV1(num_classes= 2)

# 加载resnet18 模型，
#net = models.resnet18(pretrained=False)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 2)  # 更新resnet18模型的fc模型，

if use_gpu:
    net = net.cuda()
print(net)

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# 开始训练
net.train()
for epoch in range(epochs):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(train_labels)
        # inputs, labels = Variable(inputs), Variable(train_labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, train_predicted = torch.max(outputs.data, 1)
        # import pdb
        # pdb.set_trace()
        train_correct += (train_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("epoch: ",  epoch, " loss: ", loss.item())
        train_total += train_labels.size(0)
    print('train %d epoch loss: %.3f  acc: %.3f traintotal %d' % (
    epoch + 1, running_loss / train_total * batch_size, 100 * train_correct / train_total, train_total))

    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
    #loss_plot.append(test_loss / test_total)
    #acc_plot.append(100 * correct / test_total)
    print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))
    #print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, correct))
torch.save(net, "mobilenetv1_1.00_1.pth")

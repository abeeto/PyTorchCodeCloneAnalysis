import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import LeNet5
from torchvision import datasets, transforms
from glob import glob
import numpy as np
from PIL import Image
import os


# 定义一些超参数
batch_size = 100
learning_rate = 0.001
weight_decay = 0
momentum = 0.78
epochs = 10


# 搬运数据到GPU
device = None
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

after_dir = '.png'

# 调用提供的api加载数据集
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
#
# test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)


# 自定义加载数据集
# 加载train集
train_data_length = 60000
train_data_label = [-1] * train_data_length
train_prev_dir = './mnist_png/mnist_png/training/'
for id in range(10):
    id_string = str(id)
    for filename in glob(train_prev_dir + id_string +'/' + '*.png'):
        position = filename.replace(train_prev_dir+id_string, '')
        position = position.replace(after_dir, '')
        train_data_label[int(position[1:])] = id

# 加载test集
test_data_length = 10000
test_data_label = [-1] * test_data_length
test_prev_dir = './mnist_png/mnist_png/testing/'
for id in range(10):
    id_string = str(id)
    for filename in glob(test_prev_dir + id_string +'/*.png'):
        position = filename.replace(test_prev_dir+id_string, '')
        position = position.replace(after_dir, '')
        test_data_label[int(position[1:])] = id

# 继承自Dataset
class MNISTDataset(Dataset):
    def __init__(self, img_dir, data_label):
        self.img_dir = img_dir
        self.ids = data_label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = self.img_dir+ str(idx) + '/' + str(i) + '.png'
        img = Image.open(img_file).convert('L')
        img = np.array(img)
        img = img.reshape(1,28,28)
        if img.max() > 1:
            img = img / 255
        return [torch.from_numpy(img), torch.tensor(idx)]


train_set = MNISTDataset(train_prev_dir, train_data_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = MNISTDataset(test_prev_dir, test_data_label)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


# 加载神经网络模型
net = LeNet5.LeNet5().to(device)
if os.path.exists('./MNIST_model.ph'):
    net.load_state_dict(torch.load('./MNIST_model.ph'))
    print('模型加载成功！')

# 定义损失函数和优化器
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch, last_epoch=-1)
criterion = nn.CrossEntropyLoss().to(device)


# 训练网络
for epoch in range(epochs):
    print("epoch {}, 学习率lr:".format(epoch), optimizer.param_groups[0]['lr'])
    # train
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        pred = net(data)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # 更改学习率
    scheduler.step()

    # test
    if epoch % 2 == 1:
        # net.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            pred = net(data)
            test_loss += criterion(pred, target).item()
            pred = pred.max(1)[1]
            correct += pred.eq(target).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # net.train()

# 模型保存
torch.save(net.state_dict(), './MNIST_model.ph')
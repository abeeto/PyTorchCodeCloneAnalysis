import torch
from torch.utils.data import DataLoader  # 我们要加载数据集的
from torchvision import transforms  # 数据的原始处理
from torchvision import datasets  # pytorch十分贴心的为我们直接准备了这个数据集
import torch.nn.functional as F  # 激活函数
import torch.optim as optim

batch_size = 64
# 我们拿到的图片是pillow,我们要把他转换成模型里能训练的tensor也就是张量的格式
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集，pytorch十分贴心的为我们直接准备了这个数据集，注意，即使你没有下载这个数据集
# 在函数中输入download=True，他在运行到这里的时候发现你给的路径没有，就自动下载
train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
# 同样的方式加载一下测试集
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)


# 接下来我们看一下模型是怎么做的
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义了我们第一个要用到的卷积层，因为图片输入通道为1，第一个参数就是1
        # 输出的通道为10，kernel_size是卷积核的大小，这里定义的是5x5的
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 看懂了上面的定义，下面这个你肯定也能看懂
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 再定义一个池化层
        self.pooling = torch.nn.MaxPool2d(2)
        # 最后是我们做分类用的线性层
        self.fc = torch.nn.Linear(320, 10)

    # 下面就是计算的过程
    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)  # 这里面的0是x大小第1个参数，自动获取batch大小
        # 输入x经过一个卷积层，之后经历一个池化层，最后用relu做激活
        x = F.relu(self.pooling(self.conv1(x)))
        # 再经历上面的过程
        x = F.relu(self.pooling(self.conv2(x)))
        # 为了给我们最后一个全连接的线性层用
        # 我们要把一个二维的图片（实际上这里已经是处理过的）20x4x4张量变成一维的
        x = x.view(batch_size, -1)  # flatten
        # 经过线性层，确定他是0~9每一个数的概率
        x = self.fc(x)
        return x


model = Net()  # 实例化模型
# 把计算迁移到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义一个损失函数，来计算我们模型输出的值和标准值的差距
criterion = torch.nn.CrossEntropyLoss()
# 定义一个优化器，训练模型咋训练的，就靠这个，他会反向的更改相应层的权重
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)  # lr为学习率


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 每次取一个样本
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 优化器清零
        optimizer.zero_grad()
        # 正向计算一下
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)
        # 反向求梯度
        loss.backward()
        # 更新权重
        optimizer.step()
        # 把损失加起来
        running_loss += loss.item()
        # 每300次输出一下数据
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不用算梯度
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            # 我们取概率最大的那个数作为输出
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            # 计算正确率
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()
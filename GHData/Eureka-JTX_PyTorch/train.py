import torchvision.datasets
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
eu = Eureka()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
opt = torch.optim.SGD(eu.parameters(), lr=learning_rate)

# 设置参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 激活tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    # 开始训练
    print("第{}轮训练开始".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        outputs = eu(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器调优
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 开始测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = eu(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
    print("整体测试集上的loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    total_test_step += 1

    torch.save(eu, "eu{}.pth".format(i))
    print("模型已保存")
writer.close()

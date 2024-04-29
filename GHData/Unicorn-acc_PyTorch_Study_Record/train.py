import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch import nn
from torch.utils.data import DataLoader

writer = SummaryWriter("logs_train")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
# 训练数据集的长度为：50000
# 测试数据集的长度为：10000

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

total_train_step = 0
total_test_step = 0
total_test_loss = 0
model = MyModule()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("运行设备：{}".format(device))

loss_fn = nn.CrossEntropyLoss()

lr = 1e-2 # 1e-2 = 0.01
epoch = 10

optim = torch.optim.SGD(model.parameters(), lr)
for i in range(10):
    print("--------第{}轮训练开始--------".format(i))
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_accuracy = 0
    for imgs, targets in test_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (output.argmax(1) == targets).sum() # 求输出正确的个数
        total_accuracy += accuracy
    print("整体验证集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    #每轮迭代保存一下模型，防止训练一半炸了
    torch.save(model,"model_{}".format(i))
    print("模型已保存")

writer.close()
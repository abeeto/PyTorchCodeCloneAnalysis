import os
import sys
import torch
import torch.nn as nn
# 导入优化器
import torch.optim as optim
# 导入学习率迭代器
from torch.optim.lr_scheduler import StepLR
# 把我们写的模型导入进来
from models import vgg, resnet, googlenet
# 把我们的数据导入进来
from data_process import data_process

# 获取当前路径，查看当前路径下是否存在weight目录，如果不存在则创建，用以保存模型参数
path = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]
if not os.path.exists(path+'/weight'):
    os.mkdir(path+'/weight')

# 选择要训练的模型
name = 'vgg'
if name == 'vgg':
    net = vgg()
if name == 'resnet':
    net = resnet()
if name == 'googlenet':
    net = googlenet()

# (1) 判断CPU或GPU计算资源是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# (2) 把模型放到计算资源上
net = net.to(device)
# (3) 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# (3) 定义随机梯度下降优化器，初始学习率为0.1，动量0.9，l2_norm为5e-4
optimizer = optim.SGD(net.parameters(), lr='code1', momentum=0.9, weight_decay=5e-4)
# (3)定义学习率衰减策略，每150个epoch，学习率下降为原来的0.1倍
scheduler = StepLR(optimizer, step_size='code2', gamma=0.1)

# 记录accuracy的最优值
best_acc = 0
# 读取我们之前处理的数据
trainloader, testloader = data_process()


def train(epoch):

    # 将模型设置为训练模式,即允许进行梯度的反向传播等操作
    net.train()
    # 记录每个epoch的loss总和
    train_loss = 0
    # 记录每个epoch的预测正确的样本数
    correct = 0
    # 记录每个epoch一共预测了多少样本
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # (2) 把数据放到计算资源上
        inputs, targets = inputs.to(device), targets.to(device)
        # (4) 在每次迭代前，模型参数的梯度清空，避免上一次迭代的梯度扰乱这一次迭代计算
        optimizer.zero_grad()
        # (5) 模型预测得到预测值
        outputs = net(inputs)
        # (6) 通过预测值和真实值的对比得到loss, 预测值32×10, 真实值32×1,
        loss = criterion(outputs, targets)
        # (7) Loss的反向传播
        loss.backward()
        # (8) 梯度更新,优化器进行优化 w = w – lr*w’
        optimizer.step()
        # 所有样本的loss累加
        train_loss += loss.item()
        # (9) 得到预测值中最大概率所在位置索引，即所属类别，0，1，2，3，4，5，6，7，8，9
        _, predicted = outputs.max(1)
        # (9) 计算一共多少个样本
        total += targets.size(0)
        # (9) 真实值与预测值相比较，记录正确的样本数目
        correct += predicted.eq(targets).sum().item()
        # (9) 求准确率：预测正确的数目/总样本数
        accuracy = 100. * correct / total
        # 求损失的均值
        losses = train_loss / (batch_idx + 1)
        # 打印模型训练信息,因为平台的运行时间限制，只有300s,且使用cpu进行计算，时间花费较长，所以我们只打印前3个batch的信息
        if batch_idx < 3:
           print('Training | Epoch: %d | Best_acc: %.2f | Batches: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.5f'
                 % (epoch, best_acc, batch_idx, len(trainloader), losses, accuracy, correct, total, optimizer.param_groups[0]['lr']))
        else:
            # 超过3个batch就结束训练
            break
    # (10) 学习率迭代，是在每个epoch后进行，注意与优化器在每个batch进行做区分
    scheduler.step()


def val(epoch):
    global best_acc
    # 将模型设置为验证模式，不允许反向传播等操作，节省空间
    net.eval()
    # 记录每个epoch的loss总和
    test_loss = 0
    # 记录每个epoch的预测正确的样本数
    correct = 0
    # 记录每个epoch一共预测了多少样本
    total = 0
    # 显式的说明，这里的迭代不需要计算梯度，告诉框架不要浪费空间资源
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # (2) 把数据放到计算资源上
            inputs, targets = inputs.to(device), targets.to(device)
            # (5) 模型预测得到预测值，这里我们是不需要采取对优化器的操作的，因为我们验证阶段不涉及参数更新和梯度计算
            outputs = net(inputs)
            # (6) 通过预测值和真实值的对比得到loss, 预测值32×10, 真实值32×1,这里我们求得loss之后也不需要进行反向传播，只是用以观察在验证阶段loss的下降情况
            loss = criterion(outputs, targets)
            # 所有样本的loss累加
            test_loss += loss.item()
            # (9) 得到预测值中最大概率所在位置索引，即所属类别，0，1，2，3，4，5，6，7，8，9
            _, predicted = outputs.max(1)
            # (9) 计算一共多少个样本
            total += targets.size(0)
            # (9) 真实值与预测值相比较，记录正确的样本数目
            correct += predicted.eq(targets).sum().item()
            # (9) 求准确率：预测正确的数目/总样本数
            accuracy = 100. * correct / total
            # 求损失的均值
            losses = test_loss / (batch_idx + 1)
            # 打印模型验证信息,因为平台的运行时间限制，只有300s,且使用cpu进行计算，时间花费较长，所以我们只打印前3个batch的信息
            if batch_idx < 3:
               print(
                   'Testing | Epoch: %d | Best_acc: %.2f | Batches: %d/%d | Loss: %.3f | Acc: %.3f%% (%d/%d) '
                   % (epoch, best_acc, batch_idx, len(testloader), losses, accuracy, correct, total))
            else:
                # 超过3个batch就结束验证
                break

    acc = 100.*correct/total
    # 判断最优值，如果之前的最优值小于当前epoch的acc，则进行替代，并保存模型参数
    if best_acc < acc:
        best_acc = acc
        torch.save(net.state_dict(), path+'/weight/'+name+'.pkl')  # 保存模型参数


# -*- coding: utf-8 -*-
# @Author: Lishi
# @Date:   2017-11-09 18:41:13
# @Last Modified by:   Lishi
# @Last Modified time: 2017-11-14 20:00:48
'''
    CIFAR-10 数据集分类：
        1、使用 torchvision 加载并预处理 CIFAR-10数据集
        2、定义网络
        3、定义损失函数和优化器
        4、训练网络并更新网络参数
        5、测试网络
    数据集介绍：
        CIFAR-101是一个常用的彩色图片数据集，它有10个类别:
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'；
        每张图片都是$3*32*32，也即3-通道彩色图片，分辨率为$32*32
'''
# at the top of the module
from lenet_cifar import *
import pdb

import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()  # 可以把Tensor转化为Image，方便可视化


def testCode():
    # 第一次运行程序torchvision会自动下载CIFAR-10数据集，
    # 大约100M，需花费一定的时间，
    # 如果已经下载有CIFAR-10，可通过root参数指定

    # 定义对数据的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转化为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 训练集
    trainset = tv.datasets.CIFAR10(  # Dataset对象
        root='./cifar10/',
        train=True,
        download=True,
        transform=transform)

    trainloader = t.utils.data.DataLoader(  #Dataloader 对象
        trainset, batch_size=4, shuffle=True,
        num_workers=2)  # dataset,batch_size,是否反转，多线程;

    # 测试集
    testset = tv.datasets.CIFAR10(
        './cifar10/', train=False, download=True, transform=transform)

    testloader = t.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    # Dataset对象是一个数据集，可以按照下标访问，返回形如 （data，label）的数据
    (data, label) = trainset[100]
    print(classes[label])
    # (data + 1) / 2是为了还原被归一化的数据
    show((data + 1) / 2).resize((100, 100))

    # Dataloader是一个可迭代的对象，它将dataset返回的每一条数据拼接成一个batch，并提供多线程加速优化和数据打乱等操作
    # 对dataset的所有数据遍历完一遍之后，相应的对Dataloader也完成了一次迭代
    dataiter = iter(trainloader)
    images, labels = dataiter.next()  # 返回四张图片及标签,batch_size = 4
    print(' '.join('%11s' % classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100))

    ## 定义网络
    net = Net()
    print(net)

    # 定义损失函数 和 优化器 （loss 和 optimizer）
    from torch import optim
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    ## 训练网络过程
    # 输入数据 => 前向传播+反向传播 => 更新参数
    t.set_num_threads(8)
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # 输入数据
            inputs, labels = data
            #inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # 把 Tensor 转化到 GPU上
            if t.cuda.is_available():
                net.cuda()
                images = inputs.cuda()  # 转化为 GPU Tensor
                labels = labels.cuda()  # 转化为 GPU Tensor
                output = net(Variable(images))  # 转化为 GPU Tensor
                loss = criterion(
                    output, Variable(labels))  # criterion 参数需要为 Variable变量
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            # 更新参数
            optimizer.step()

            # 打印 log 信息
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # 每 2000个batch打印一个训练状态
                print('[%d, %5d] loss: %.3f' \
                % (epoch + 1,i+1,running_loss/2000))
                running_loss = 0.0

    print('Finished Training')

    ## 测试：在测试集上进行图像类别的预测
    # 实际labels
    dataiter = iter(testloader)
    images, labels = dataiter.next()  # 一个batch返回4张图像
    print('实际的label: ', ' '.join(\
            '%08s'%classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100))

    # 测试
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))


def main():
    testCode()


if __name__ == '__main__':  # 模块名称
    main()
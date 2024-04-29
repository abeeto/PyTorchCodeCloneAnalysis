"""
    目前为止，我们已经看到了如何定义网络，计算损失，并更新网络的权重。所以你现在可能会想,
    数据应该怎么办呢？
    通常来说，当必须处理图像、文本、音频或视频数据时，可以使用python标准库将数据加载到numpy数组里。然后将这个数组转化成torch.*Tensor。
        对于图片，有Pillow，OpenCV等包可以使用
        对于音频，有scipy和librosa等包可以使用
        对于文本，不管是原生python的或者是基于Cython的文本，可以使用NLTK和SpaCy
    特别对于视觉方面，我们创建了一个包，名字叫torchvision，其中包含了针对Imagenet、CIFAR10、MNIST等常用数据集的数据加载器(data loaders
    还有对图像数据转换的操作，即torchvision.datasets和torch.utils.data.DataLoader
    这提供了极大的便利，可以避免编写样板代码
    在这个教程中，我们将使用CIFAR10数据集，它有如下的分类：“飞机”，“汽车”，“鸟”，“猫”，“鹿”，“狗”，“青蛙”，“马”，“船”，“卡车”等。
    在CIFAR-10里面的图片数据大小是3x32x32，即：三通道彩色图像，图像大小是32x32像素。
    训练一个图片分类器
    我们将按顺序做以下步骤：
        1.通过torchvision加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
        2.定义卷积神经网络
        3.定义损失函数
        4.利用训练数据训练网络
        5.利用测试数据测试网络
"""
# 使用torchvision加载CIFAR10超级简单
import torch
import torchvision
import torchvision.transforms as transforms
# torchvision数据集加载完后的输出是范围在[0, 1]之间的PILImage。
# 我们将其标准化为范围在[-1, 1]之间的张量。

def trainTest():
    # 会生成 ./data目录，并有一些文件
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import matplotlib.pyplot as plt
    import numpy as np

    # 输出图像的函数
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.ion()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
        plt.pause(2)  # 显示秒数
        plt.close()

    # 随机获取训练图片
    dataiter = iter(trainloader)
    # while 1:
    images, labels = dataiter.next()
    # 打印图片标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # 显示图片
    imshow(torchvision.utils.make_grid(images))

    # 3.定义损失函数和优化器
    # ****************还没学完，后面继续
    # https://pytorch.apachecn.org/docs/1.4/blitz/cifar10_tutorial.html

if __name__=="__main__":
    trainTest()


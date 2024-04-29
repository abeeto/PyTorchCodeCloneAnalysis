import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from VGG import VGG16


def originalImgShow(img):
    """绘制原始图片"""
    img = img.squeeze()
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.savefig('./picture/feature/origin_img.png')
    plt.show()


def convShow(img, num):
    """卷积层输出绘制"""
    img = img.squeeze()

    # 设置行列数
    # 输出为16列，行数由通道数确定
    length = len(img)
    col = 16
    row = length // col

    # 子图的位置
    flag = 1

    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, flag)
            plt.axis("off")
            plt.imshow(img[flag - 1, :, :], cmap='gray')
            flag += 1

    plt.savefig('./picture/feature/feature_map{}.png'.format(num))
    plt.show()


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 随机取图将shuffle换为True
    train_dataset = datasets.CIFAR10(root='./', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    img_iter = iter(train_loader)
    images, _ = img_iter.next()

    # 绘制原始图片
    originalImgShow(images)

    # 实例模型
    model = VGG16().eval()
    model.load_state_dict(torch.load("./VGG16.pth"))

    # 存放特征图的列表
    feature_map = []

    def forward_hook(module, input, output):
        feature_map.append(output.detach())

    # 卷积层的Sequential
    features = list(model.children())[0]

    # 14层卷积层
    conv1 = features[0]
    conv2 = features[3]
    conv3 = features[7]
    conv4 = features[10]
    conv5 = features[14]
    conv6 = features[17]
    conv7 = features[20]
    conv8 = features[24]
    conv9 = features[27]
    conv10 = features[30]
    conv11 = features[34]
    conv12 = features[37]
    conv13 = features[40]
    conv14 = features[43]

    conv = [conv1, conv2, conv3, conv4, conv5,
            conv6, conv7, conv8, conv9, conv10,
            conv11, conv12, conv13, conv14]

    # 注册14层卷积的hook
    for c in conv:
        c.register_forward_hook(forward_hook)

    # 前向传播
    with torch.no_grad():
        model(images)

    # 打印每一层的特征图
    for i in range(len(feature_map)):
        convShow(feature_map[i], i + 1)


if __name__ == '__main__':
    main()

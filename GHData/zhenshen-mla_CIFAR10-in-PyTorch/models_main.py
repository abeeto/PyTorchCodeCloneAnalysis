import torch
from models import vgg, resnet, googlenet

if __name__ == '__main__':
    net1 = vgg()
    net2 = resnet()
    net3 = googlenet()

    x = torch.randn(1, 3, 32, 32)
    y1 = net1(x)
    y2 = net2(x)
    y3 = net3(x)

    if y1.size(0) == y2.size(0) == y3.size(0) == 1 and y1.size(1) == y2.size(1) == y3.size(1) == 10:
        print('模型创建成功！')
        print(1)
    else:
        print('模型创建失败！')
        print(2)
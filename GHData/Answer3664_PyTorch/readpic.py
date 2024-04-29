from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def readImage(path='./3.jpg', size=28):
    mode = Image.open(path).convert('L')  # 转换成灰度图
    transform1 = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),  # 切割
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    mode = mode.view(mode.size(0), -1)
    return mode


def showTorchImage(image):
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    plt.show()
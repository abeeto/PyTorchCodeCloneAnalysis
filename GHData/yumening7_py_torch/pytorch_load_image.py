'''
@author: Dzh
@date: 2020/1/9 15:03
@file: pytorch_load_image.py
'''

from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
train_set = tv.datasets.ImageFolder(root='./', transform=transform)
data_loader = DataLoader(dataset=train_set)

to_pil_image = transforms.ToPILImage()

for image, label in data_loader:

    # 方法1：Image.show()
    # transforms.ToPILImage()中有一句
    # npimg = np.transpose(pic.numpy(), (1, 2, 0))
    # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
    img = to_pil_image(image[0])
    img.show()

    # 方法2：plt.imshow(ndarray)
    img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    img = img.numpy()  # FloatTensor转为ndarray
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后

    # 显示图片
    plt.imshow(img)
    plt.show()

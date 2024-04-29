# -*- coding: utf-8 -*-
# weibifan 2022-10-12
#  TorchVision：图像数据集及处理工具。
# 图像数据的案例：图像的显示

"""
图像数据集的预处理：torchvision.transforms

0）将narray数据转换为tensor，使用ToTensor()
1）归一化，Normalize()
3）图像数据的平坦化，Flat()

*）多个预步骤需要使用Compose，lambda，或者转换函数。

The FashionMNIST features are in PIL Image format, and the labels are integers.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 第1步和第2步：构建内存数据集
# 先看本地是否有数据集，如果有则加载。如果没有则先下载到本地，然后加载
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 补充标签的名称。
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 第2步的子步骤：浏览图像数据
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):   #为什么要+1？最后一个stop值，此时退出循环。
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # 数据集的随机访问
    img, label = training_data[sample_idx]  #每次只能访问一个instance，难以支持minibatch

    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

#
from torch.utils.data import DataLoader

# 第3步：构建数据集的访问指针。分为随机访问，序列访问。
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

"""
ToTensor()

Lambda Transforms
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

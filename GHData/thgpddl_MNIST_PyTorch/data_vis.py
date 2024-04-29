# -*- encoding: utf-8 -*-
"""
@File    :   data_vis.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/29 19:51   thgpddl      1.0         None
"""
from data import getLoader
import matplotlib.pyplot as plt

data_train_loader, data_test_loader = getLoader()

figure = plt.figure()
num_of_images = 60

for imgs, targets in data_train_loader:
    break

for index in range(num_of_images):
    plt.subplot(6, 10, index + 1)
    plt.axis("off")  # 关闭坐标轴
    img = imgs[index]
    # numpy()：将tensor进行array化
    # squeeze()：降维
    plt.imshow(img.numpy().squeeze(), cmap="gray_r")
plt.show()

# -*- coding: utf-8 -*-
"""
@Author ：Shimon-Cheung
@Date   ：2022/4/24 15:41
@Desc   ：transforms工具箱，一些预处理方法
"""

"""
python的用法 -》 tensor数据类型
通过transforms.ToTensor去看这两个问题
1、 通过transforms该如何使用
2、 为什么我们需要tensor数据类型
"""
import cv2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 1、 通过transforms该如何使用
img = cv2.imread("dataset/hymenoptera_data/train/ants/0013035.jpg")
tensor_obj = transforms.ToTensor()  # 初始化tensor转换对象，类似工厂方法，返回一个对象，下面的 tensor_obj()就是调用__call__方法
img_tensor = tensor_obj(img)  # 这里传入的是PIL Image 或者 numpy.ndarray
print(img_tensor)

# 2、为什么我们需要tensor数据类型，方便使用
writer = SummaryWriter(log_dir="logs")  # 传入log文件目录初始化对象
writer.add_image("to_tensor", img_tensor)  # 把图片tensor传入进去
writer.close()

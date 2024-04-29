# -*- coding: utf-8 -*-
"""
@Author ：Shimon-Cheung
@Date   ：2022/4/24 16:17
@Desc   ：transforms的常用的一些案例
"""
from torchvision import transforms
# import cv2
from PIL import Image

# 先获取一个图片对象，方便下面的操作
# img = cv2.imread("dataset/hymenoptera_data/train/bees/17209602_fe5a5a746f.jpg")
img = Image.open("dataset/hymenoptera_data/train/bees/17209602_fe5a5a746f.jpg")

# 1、totensor的使用
totensor_obj = transforms.ToTensor()
img_tensor = totensor_obj(img)
print(type(img_tensor))

# 2、Normalize归一化处理
print(img_tensor[0][0][0])  # 归一化之前的第一个数值
normalize_obj = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 传入均值和标准差
img_norm = normalize_obj(img_tensor)
print(img_norm[0][0][0])  # 归一化之后的第一个数值

# 3、compose的使用
resize = transforms.Resize((512, 512))  # 先初始化一个重置图片size的对象
tran_compose = transforms.Compose([
    resize,  # 先执行裁剪为512的操作
    totensor_obj  # 再执行转换成tensor，这里的输入就是上一步的输出
])
img_compose = tran_compose(img)
print(img_compose)

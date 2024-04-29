# -*- coding: utf-8 -*-
"""
@Author ：Shimon-Cheung
@Date   ：2022/4/24 10:34
@Desc   ：tensorboard中图片的写入操作
"""
import cv2  # 导入库
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs")
img = cv2.imread("dataset/hymenoptera_data/train/ants/0013035.jpg")  # 读取图片
print(img.shape)
writer.add_image("test", img, 1, dataformats="HWC")
writer.close()
# tensorboard --logdir=logs 运行web端命令

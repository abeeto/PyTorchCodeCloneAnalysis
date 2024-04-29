#!/usr/bin/env python
# encoding: utf-8
'''
@author:muczcy
@file: Step3-Tensorboard.py
@time: 2022/2/7 17:38
@contact: 21400179@muc.edu.cn
'''

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer = SummaryWriter("logs")
image_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(type(img_array))

print(img_array.shape)


writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()
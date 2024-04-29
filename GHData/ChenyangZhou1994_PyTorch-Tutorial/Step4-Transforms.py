#!/usr/bin/env python
# encoding: utf-8
'''
@author:muczcy
@file: Step4-Transforms.py
@time: 2022/2/7 17:42
@contact: 21400179@muc.edu.cn
'''

from PIL import Image
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)
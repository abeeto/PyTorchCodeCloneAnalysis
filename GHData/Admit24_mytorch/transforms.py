import numpy as np
from torchvision.transforms import transforms
from PIL import Image

# CenterCrop(size)	中心裁剪
# FiveCrop(size)	4个角+中心裁剪 = 5， 返回多张图像
# Grayscale(num_output_channels = 1)	灰度化
# Pad(padding, fill=o,padding_mode='constant)	图像边沿加pad
# RandomAffine(degrees,translate,scale,shear,resample,fillcolor)	随进放射变换
# RandomApply(..)	对图像随机应用变换
# RandomCrop(..)	随机位置裁剪
# RandomGrayscale(..)
# Resize(size)	对图像进行尺寸缩放

# 准备好实验的图像，一个彩色32bit图像
IMG_PATH = './lena.png'
img = Image.open(IMG_PATH)

# -----------------类型转换---------------------------------------
# transforms1 = transforms.Compose([transforms.ToTensor()])
# img1 = transforms1(img)
# print('img1 = ', img1)

# ---------------Tensor上的操作---------------------------------
# transforms2 = transforms.Compose([transforms.Normalize(mean=(0.5, 0.5, #0.5), std=(0.5, 0.5, 0.5))])
# img2 = transforms2(img1)
# print('img2 = ', img2)

# ---------------PIL.Image上的操作---------------------------------
transforms3 = transforms.Compose([transforms.Resize(256)])
img3 = transforms3(img)
print('img3 = ', img3)
img3.show()

transforms4 = transforms.Compose([transforms.CenterCrop(256)])
img4 = transforms4(img)
print('img4 = ', img4)
img4.show()

transforms5 = transforms.Compose([transforms.RandomCrop(224, padding=0)])
img5 = transforms5(img)
print('img5 = ', img5)
img5.show()

transforms6 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
img6 = transforms6(img)
print('img6 = ', img6)
img6.show()

transforms7 = transforms.Compose([transforms.ColorJitter()])
img7 = transforms7(img)
img7.show()

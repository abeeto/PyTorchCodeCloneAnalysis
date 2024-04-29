import torch
import numpy as np
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image

psnr = PeakSignalNoiseRatio()
'''
读取照片 -> 转换为ndarray -> 使用torchmetrices对tensor进行运算
SSIM 对数据的要求是四维的 batch canal weigh height

'''

preds = Image.open(r"C:\MyDataset\test_dataset\ITS\gt\1400.png")
target = Image.open(r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")
preds = np.array(preds)
target = np.array(target)
preds = torch.tensor(preds)
target = torch.tensor(target)
# mypsnr = psnr(target,preds)
# print(mypsnr)

for i in range(10):
	mypsnr = psnr(target, preds)
	print("psnr is {} in {}".format(mypsnr,i))



metric = psnr.compute()
print(metric)



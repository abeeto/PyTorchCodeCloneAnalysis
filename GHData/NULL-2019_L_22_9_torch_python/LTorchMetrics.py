import torch
import torchvision
# print(torchvision.__version__)

import torchmetrics.image.ssim

# from torchmetrics import StructuralSimilarityIndexMeasure
# from torchmetrics import PeakSignalNoiseRatio
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# psnr = PeakSignalNoiseRatio()
# ssim = StructuralSimilarityIndexMeasure()
# import torch
# preds = Image.open(r"C:\MyDataset\test_dataset\ITS\gt\1400.png")
# target = Image.open(r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")
# preds = np.array(preds)
# target = np.array(target)
# preds = torch.tensor(preds)
#
# target = torch.tensor(target)
#
# mypsnr = psnr(preds, target)
# # print(preds.shape())
# print(preds.size())
#
# print(mypsnr)

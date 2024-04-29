import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import torchvision.transforms as transforms
from Models.UNet.unet_model import UNet
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    weights_file = "C:/Users/PC/PycharmProjects/Pytorch-SegmentationModels/output/epoch_70.pth"
    image_file = "C:/Users/PC/PycharmProjects/Pytorch-SegmentationModels/7_A2780.png"


    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet(1,1).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    y = cv2.imread(image_file,0)
    y=np.array(y).astype('float64')

    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    image = y.to(device=device,dtype=torch.float)
    with torch.no_grad():
        preds = model(image)


    print(preds)
    preds = preds.cpu().data.numpy()

    print(preds.shape)

    preds = np.array(preds).reshape((2448,3264))

    norm_image = cv2.normalize(preds, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    ret3, threshold = cv2.threshold(norm_image,150, 255, cv2.THRESH_BINARY)

    cv2.imwrite("segout.png",threshold)

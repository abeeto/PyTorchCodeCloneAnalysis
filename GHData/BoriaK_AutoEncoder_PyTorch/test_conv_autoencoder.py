import os, sys, glob, argparse
import re
import torch.utils.data
import torch
from datasets import DataSet1
# import argparse
import torchvision.transforms as transforms
from models import AutoEncoderTest
import torchvision.utils as tv_utils
from PIL import Image
from add_Functions import PSNR
from add_Functions import JPEGCompression
from add_Functions import JPEG2000Compression
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test_chkps', type=str, default='c_1', help='checkpoint for testing')
parser.add_argument('--data', type=str, default=r'./dataSet/TestData', help='root directory of the dataset')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--root-chkps', type=str, default='./checkpoints', help='checkpoints directory')
parser.add_argument('--Image_size', type=int, default=128, help='size of the input images')
args = parser.parse_args()
args.device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
args.dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor
args.dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() and args.cuda else torch.LongTensor

# get the model using our helper function
# move model to the right device
model = AutoEncoderTest(args.Image_size).to(args.device)

f = os.path.join(args.root_chkps, args.test_chkps+'.pth')
chkp = torch.load(f)
model.load_state_dict(chkp['model_state'])
model.to(args.device)
model.eval()

transforms_test = transforms.Compose([
    # transforms.RandomCrop((args.Image_size, args.Image_size)),
    transforms.Resize((args.Image_size, args.Image_size)),
    transforms.ToTensor()
])

dataset_test = DataSet1(args.data, mode='Testing', transforms_=transforms_test)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0)

psnr_Arr = np.zeros((3, len(dataset_test)))
Orig_Image_Arr = torch.empty(0)
CAE_Image_Arr = torch.empty(0)
JPEG_Image_Arr = torch.empty(0)
JPEG2K_Image_Arr = torch.empty(0)
B = 12
with torch.no_grad():
    for i, data in enumerate(data_loader_test):
        data = data.to(args.device).type(args.dtype_float)
        output = model(data)
        Num_of_Pixels = output[1].shape[2] * output[1].shape[3]
        Num_of_Symbols = output[0].shape[1] * output[0].shape[2]
        Num_of_Bits = Num_of_Symbols * B
        # Code_Rate = len(output[0])/(data[i].shape()[0]*data[i].shape()[1])  # Bits per pixel for the coded image
        Code_Rate = Num_of_Bits / Num_of_Symbols

        print(output[0].shape)
        print(output[1].shape)

        psnr_AE = PSNR(data, output[1])
        print(psnr_AE)
        psnr_Arr[0][i] = psnr_AE

        data_JPEG = JPEGCompression(data.squeeze(0))
        psnr_JPEG = PSNR(data, data_JPEG)
        print(psnr_JPEG)
        psnr_Arr[1][i] = psnr_JPEG

        data_JPEG2K = JPEG2000Compression(data.squeeze(0), B)
        psnr_JPEG2K = PSNR(data, data_JPEG2K)
        print(psnr_JPEG2K)
        psnr_Arr[2][i] = psnr_JPEG2K

        # stack 5 Original, CAE, JPEG and JPEG2K Images side by side
        if i % 2 == 0:
            Orig_Image_Arr = torch.cat((Orig_Image_Arr, data.squeeze(0)), dim=-1)
            CAE_Image_Arr = torch.cat((CAE_Image_Arr, output[1].squeeze(0)), dim=-1)
            JPEG_Image_Arr = torch.cat((JPEG_Image_Arr, data_JPEG.squeeze(0)), dim=-1)
            JPEG2K_Image_Arr = torch.cat((JPEG2K_Image_Arr, data_JPEG2K.squeeze(0)), dim=-1)

    # Save the original, CAE Reconstructed, JPEG and JPEG2K Images
    Orig_Image_Arr_img = transforms.functional.to_pil_image(Orig_Image_Arr)
    CAE_Image_Arr_img = transforms.functional.to_pil_image(CAE_Image_Arr)
    JPEG_Image_Arr_img = transforms.functional.to_pil_image(JPEG_Image_Arr)
    JPEG2K_Image_Arr_img = transforms.functional.to_pil_image(JPEG2K_Image_Arr)
    # Orig_Image_Arr_img.show()
    Orig_Image_Arr_img.save('./Test_Results/Original_Images.png')
    CAE_Image_Arr_img.save('./Test_Results/CAE_Images.png')
    JPEG_Image_Arr_img.save('./Test_Results/JPEG_Images.png')
    JPEG2K_Image_Arr_img.save('./Test_Results/JPEG2000_Images.png')

    plt.figure()
    plt.plot(range(len(dataset_test)), psnr_Arr[0][:], range(len(dataset_test)), psnr_Arr[1][:], range(len(dataset_test)
                                                                                                       ),
             psnr_Arr[2][:])
    plt.title("PSNR over test samples, for CAE and JPEG2K R=12bpp, for JPEG Q=75")
    plt.xlabel("tested Data Samples")
    plt.ylabel("PSNR [dB]")
    plt.grid()
    plt.legend(['CAE PSNR', 'JPEG PSNR', 'JPEG2000 PSNR'])

    plt.savefig('./Test_Results/PSNR_over_Test_samples.png')
    plt.show()

    psnr_AE_AVG = np.mean(psnr_Arr[0])
    psnr_JPEG_AVG = np.mean(psnr_Arr[1])
    psnr_JPEG2K_AVG = np.mean(psnr_Arr[2])
    print('Average CAE PSNR = ' + str(psnr_AE_AVG))
    print('Average JPEG PSNR = ' + str(psnr_JPEG_AVG))
    print('Average JPEG2000 PSNR = ' + str(psnr_JPEG2K_AVG))

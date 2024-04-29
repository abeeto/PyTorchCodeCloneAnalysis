import torch
import os
import csv
import cv2
import torchvision
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from Data.Dataset import SegmentationDataset, Detection
import time
import segmentation_models_pytorch as smp
from config import *
from utils import *
from utils.csv_file import CSV
import sys, getopt

from utils.utils import get_device, visualize

weigths_path= ""
image_path= ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:w:",["img=","weigths="])
except getopt.GetoptError:
    print('test.py -i <inputfile> -o <outputfile>')
    sys.exit(2)

for opt, arg in opts:
      if opt == '-h':
         print('unet_detection.py -img <image> -w <path to weigths>')
         sys.exit()
      elif opt in ("-i", "--img"):
         image_path = arg
      elif opt in ("-w", "--weigths"):
         weigths_path = arg
if weigths_path is "" or image_path is "":
    print('unet_detection.py -img <image> -w <path to weigths>')
    sys.exit()
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                 ])

test_dataset = Detection([image_path], transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=NUM_CLASSES,  # model output channels (number of classes in your dataset)
)
model.load_state_dict(torch.load(weigths_path, map_location=torch.device('cpu')))
DEVICE = get_device()
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    for (i, (x, y)) in enumerate(test_loader):
        # send the input to the device
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        # make the predictions and calculate the validation loss
        pred = model(x)
        pred = torch.sigmoid(pred)

        filename = "predictions_{}.png".format(i)
        visualize(filename, Image=x[0].cpu().data.numpy(),
                        Prediction=pred.cpu().data.numpy()[0].round())
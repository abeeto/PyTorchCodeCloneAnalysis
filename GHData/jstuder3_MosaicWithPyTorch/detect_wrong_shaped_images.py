from torchvision.io import read_image, write_png, write_jpeg
from torchvision.transforms import Resize
import torch.nn as nn
import torch
import sys
import time
import glob
import random
from PIL import Image

from matplotlib import pyplot

device = "cuda" if torch.cuda.is_available() else "cpu"

images_folder = "C:/Users/justi/Desktop/MosaicWithPyTorch/AlleBilder/"

filenames = glob.glob(images_folder+"*.JPG")

for file in filenames:
    im = Image.open(file)
    if round(im.size[0]/im.size[1], 2)!=1.5:
        print(f"{file} - {im.size} : {round(im.size[0]/im.size[1], 2)}")


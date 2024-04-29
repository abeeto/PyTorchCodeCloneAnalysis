from __future__ import print_function
from pathlib import Path
from torchvision.utils import save_image

import torch
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
import numpy as np
from dataset import UpsampleDataset
from model import Model
from tqdm import tqdm
from skimage import io, transform
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from model2 import Model2



if __name__ == "__main__":
    Path('./generated').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('-m', action='store', dest='model')
    parser.add_argument('-i', action='store', dest='input')

    args = parser.parse_args()

    print(f'Using version {args.model}')
    print(f'Upscaling file {args.input}')


    model = Model2(10)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    # resize = T.Resize((256, 256))
    # img = resize(Image.open(args.input).convert('RGB'))
    img = Image.open(args.input).convert('RGB')
    img = T.ToTensor()(img)

    img = torch.unsqueeze(img, 0)
    T.ToPILImage()(img[0]).save('./generated/in.jpg')
    
    out = model(img)

    save_image(out[0], f'./generated/out.jpg')
    # T.ToPILImage()(out[0]).save(f'./generated/out.jpg')
    
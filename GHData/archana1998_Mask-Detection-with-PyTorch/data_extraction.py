import argparse
import json
import torch 
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Resize, ToPILImage, ToTensor, Normalize
from torch import long, tensor
from torch.utils.data.dataset import Dataset

classes = ['with_mask', 'without_mask']
parser = argparse.ArgumentParser()
parser.add_argument('-rootdir', type=str, default='root', help='Directory of training data')
args = parser.parse_args()

root_dir = Path(args.rootdir)
mask_path = root_dir/'with_mask'
nonmask_path = root_dir/'without_mask'
print(nonmask_path)
maskdata_df = pd.DataFrame()

if not os.path.isfile('data/mask_df.pickle'):
    for subject in tqdm(list(nonmask_path.iterdir()), desc='non mask photos'):
        image = cv2.imread(str(subject))
        maskdata_df = maskdata_df.append({
        'image': image,
        'mask': 0
    }, ignore_index=True)
    for subject in tqdm(list(mask_path.iterdir()), desc='mask photos'):
        image = cv2.imread(str(subject))
        maskdata_df = maskdata_df.append({
            'image': image,
            'mask': 1
    }, ignore_index=True)
    pickle_path = Path('data')
    if not os.path.isdir(pickle_path):
        os.mkdir(pickle_path)
    maskdata_df.to_pickle('data/mask_df.pickle')

else: 
    print('Dataframe exists')





     
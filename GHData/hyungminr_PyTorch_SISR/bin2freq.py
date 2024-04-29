import torch

import torchvision.transforms as T

import glob
from PIL import Image
from tqdm import tqdm
import os
from utils import high_pass_filter_hard_kernel

transform = []
transform.append(T.ToTensor())
t = T.Compose(transform)



data_dir = []
data_dir.append('./data/DIV2K/bin/DIV2K_train_HR/*pt')
data_dir.append('./data/DIV2K/bin/DIV2K_valid_HR/*pt')
data_dir.append('./data/DIV2K/bin/DIV2K_train_LR_bicubic/X2/*pt')
data_dir.append('./data/DIV2K/bin/DIV2K_valid_LR_bicubic/X2/*pt')
data_dir.append('./data/DIV2K/bin/DIV2K_train_LR_bicubic/X4/*pt')
data_dir.append('./data/DIV2K/bin/DIV2K_valid_LR_bicubic/X4/*pt')

for d in data_dir:
    images = glob.glob(d)
    for iname in tqdm(images):
        rname = iname.replace('/DIV2K/bin/', '/DIV2K/bin/hfreq/')
        if os.path.exists(rname): continue
        tensor = torch.load(iname)
        tensor = high_pass_filter_hard_kernel(tensor.unsqueeze(0))
        tensor = tensor.squeeze(0)
        os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
        torch.save(tensor, rname)

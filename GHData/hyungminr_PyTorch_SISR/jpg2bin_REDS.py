import torch

import torchvision.transforms as T

import glob
from PIL import Image
from tqdm import tqdm
import os

transform = []
transform.append(T.ToTensor())
t = T.Compose(transform)

data_dir = []
data_dir.append('./data/benchmark/REDS/train/train_sharp/*/*.png')
data_dir.append('./data/benchmark/REDS/train/train_blur_bicubic/X4/*/*.png')
data_dir.append('./data/benchmark/REDS/val/val_sharp/*/*.png')
data_dir.append('./data/benchmark/REDS/val/val_blur_bicubic/X4/*/*.png')
data_dir.append('./data/benchmark/REDS/test/test_blur_bicubic/X4/*/*.png')


        
iname = './data/benchmark/REDS/train/train_blur_bicubic/X4/069/00000006.png'                                                            
rname = iname.replace('/REDS/', '/REDS/bin/')
rname = rname.replace('.png', '.pt')
img = Image.open(iname)
tensor = t(img)
os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
torch.save(tensor, rname)

"""
for d in data_dir:
    print(d)
    images = glob.glob(d)
    for iname in tqdm(images):
        rname = iname.replace('/REDS/', '/REDS/bin/')
        rname = rname.replace('.png', '.pt')
        if os.path.exists(rname): continue
        img = Image.open(iname)
        tensor = t(img)
        os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
        torch.save(tensor, rname)

"""

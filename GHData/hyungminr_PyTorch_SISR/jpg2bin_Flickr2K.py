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
data_dir.append('./data/benchmark/Flickr2K/train/sharp/*.png')
data_dir.append('./data/benchmark/Flickr2K/train/blur/*.png')
data_dir.append('./data/benchmark/Flickr2K/val/sharp/*.png')
data_dir.append('./data/benchmark/Flickr2K/val/blur/*.png')

for d in data_dir:
    print(d)
    images = glob.glob(d)
    for iname in tqdm(images):
        try:
            rname = iname.replace('/Flickr2K/', '/Flickr2K/bin/')
            rname = rname.replace('.png', '.pt')
            if os.path.exists(rname): continue
            img = Image.open(iname)
            tensor = t(img)
            # os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
            torch.save(tensor, rname)
        except:
            with open('ERROR_LIST.txt', 'a') as f:
                f.write(f'{rname}\n')

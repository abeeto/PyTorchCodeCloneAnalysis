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
data_dir.append('./data/benchmark/REDS/bin/train/train_sharp/*/*.pt')
data_dir.append('./data/benchmark/REDS/bin/train/train_blur_bicubic/X4/*/*.pt')
data_dir.append('./data/benchmark/REDS/bin/val/val_sharp/*/*.pt')
data_dir.append('./data/benchmark/REDS/bin/val/val_blur_bicubic/X4/*/*.pt')
data_dir.append('./data/benchmark/REDS/bin/test/test_blur_bicubic/X4/*/*.pt')

failure_cases = []
for d in data_dir:
    images = glob.glob(d)
    for iname in tqdm(images):
        rname = iname.replace('/bin/', '/bin/hfreq/')
        if os.path.exists(rname): continue
        try:
            tensor = torch.load(iname)
            tensor = high_pass_filter_hard_kernel(tensor.unsqueeze(0))
            tensor = tensor.squeeze(0)
            os.makedirs('/'.join(rname.split('/')[:-1]), exist_ok=True)
            torch.save(tensor, rname)
        except:
            failure_cases.append(iname)
print(failure_cases)

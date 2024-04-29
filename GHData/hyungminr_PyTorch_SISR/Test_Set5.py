import os
import glob
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as T

from models.EDSR import EDSR
from models.common import GMSD_quality

from utils import imshow
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr

torch.manual_seed(0)
scale_factor = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)
model = EDSR(scale=scale_factor).to(device)

weights = {'Baseline': torch.load('./weights/2021.01.15/EDSR_x2_Baseline/epoch_1000.pth'),
           'MSHF': torch.load('./weights/2021.01.15/EDSR_x2_v10_MSHF/epoch_1000.pth'),
           'GMSD': torch.load('./weights/2021.01.15/EDSR_x2_v8_gmsd/epoch_1000.pth')}

def get_tensor(lr):
    trans = T.Compose([T.ToTensor()])
    hr = lr.replace('LR_bicubic/X2', 'HR')
    hr = hr.replace('x2.png', '.png')
    lr = Image.open(lr)
    lr = trans(lr)
    lr = lr.unsqueeze(0).to(device)
    hr = Image.open(hr)
    hr = trans(hr)
    hr = hr.unsqueeze(0).to(device)
    
    return lr, hr

set5 = glob.glob('./data/benchmark/Set5/LR_bicubic/X2/*.png')
results = {s:{} for s in set5}
evals = {}
evals['psnr'] = {s.split('/')[-1]:{} for s in set5}
evals['ssim'] = {s.split('/')[-1]:{} for s in set5}
evals['ms_ssim'] = {s.split('/')[-1]:{} for s in set5}

dataset = 'Set5'
result_dir = f'./results/Benchmark/{dataset}/'
os.makedirs(result_dir, exist_ok=True)

for k in weights.keys():
    model.load_state_dict(weights[k])
    model.eval()
    for image in set5:
        lr, hr = get_tensor(image)
        sr, fea = model(lr)
        results[image][k] = sr
        evals['psnr'][image.split('/')[-1]][k] = get_psnr(sr, hr)
        evals['ssim'][image.split('/')[-1]][k] = get_ssim(sr, hr).item()
        evals['ms_ssim'][image.split('/')[-1]][k] = get_msssim(sr, hr).item()

for image in set5:
    lr, hr = get_tensor(image)
    rname = result_dir + image.split('/')[-1]
    img = [torch.cat((lr, torch.zeros_like(lr)), dim=-2)] + [results[image][k] for k in weights.keys()] + [hr]
    imshow(img, filename=rname, visualize=False)
    
df = pd.DataFrame(evals['psnr'])
df.to_csv(result_dir + 'x2.csv')
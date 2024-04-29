#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

from models.EDSR import EDSR
from models.EDSR_freq_fusion import EDSR_fusion
from models.EDSR_hm import EDSR as EDSR_hm
from models.EDSR_hm_with_att import EDSR as EDSR_hm_with_att
from models.EDSR_with_att2 import EDSR as EDSR_with_att2
from models.EDSR_hm_with_att_v2 import EDSR as EDSR_hm_with_att_v2
from models.EDSR_hm_high_freq import EDSR as EDSR_hm_high_freq
from models.EDSR_with_att2_std import EDSR as EDSR_with_att2_std
from models.EDSR_with_att2_std2 import EDSR as EDSR_with_att2_std2

from models.common import GMSD_quality

from utils import imshow
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr
from utils import pass_filter
from utils import high_pass_filter_hard_kernel

torch.manual_seed(0)
scale_factor = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)
# model = EDSR(scale=scale_factor).to(device)

if scale_factor == 2:
    weights = {'Baseline': torch.load('./weights/Benchmark/EDSR_x2.pth'),
               'GMSD': torch.load('./weights/2021.01.19/EDSR_x2_v8_GMSD/epoch_1000.pth'),
               'GMS': torch.load('./weights/2021.01.19/EDSR_x2_v11_GMS/epoch_1000.pth'),
               'GMS_MSHF': torch.load('./weights/2021.01.19/EDSR_x2_v12_GMS_MSHF/epoch_1000.pth'),
               'Freq_Fusion': torch.load('./weights/2021.01.20/EDSR_x2_v18_freq_fusion/epoch_1000.pth'),
               'EDSR_x2_v39_hm': torch.load('./weights/2021.01.27/EDSR_x2_v39_hm/epoch_1000.pth'),
               'EDSR_x2_v40_hm_att': torch.load('./weights/2021.01.27/EDSR_x2_v40_hm_att/epoch_1000.pth'),
               'EDSR_x2_v41_hm_att_v2': torch.load('./weights/2021.01.27/EDSR_x2_v41_hm_att_v2/epoch_1000.pth'),
               'EDSR_x2_v43_base_att2': torch.load('./weights/2021.01.27/EDSR_x2_v43_base_att2/epoch_1000.pth'),
               'EDSR_x2_v44_base_att2_std': torch.load('./weights/2021.01.27/EDSR_x2_v44_base_att2_std/epoch_1000.pth'),
               'EDSR_x2_v45_base_att2_std2': torch.load('./weights/2021.01.27/EDSR_x2_v45_base_att2_std2/epoch_1000.pth'),
              }

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

set5 = glob.glob(f'./data/benchmark/Set5/LR_bicubic/X{scale_factor}/*.png')
results = {s:{} for s in set5}
evals = {}
evals['psnr'] = {s.split('/')[-1]:{} for s in set5}
evals['ssim'] = {s.split('/')[-1]:{} for s in set5}
evals['ms_ssim'] = {s.split('/')[-1]:{} for s in set5}

dataset = 'Set5'
result_dir = f'./results/Benchmark/{dataset}/'
os.makedirs(result_dir, exist_ok=True)

with torch.no_grad():
    for ki, k in enumerate(weights.keys()):
        print(f'{ki+1} / {len(weights.keys())} || {k}')
        if k == 'Freq_Fusion':
            model = EDSR_fusion(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v39_hm':
            model = EDSR_hm(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v40_hm_att':
            model = EDSR_hm_with_att(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v41_hm_att_v2':
            model = EDSR_hm_with_att_v2(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v43_base_att2':
            model = EDSR_with_att2(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v44_base_att2_std':
            model = EDSR_with_att2_std(scale=scale_factor).to(device)
        elif k == 'EDSR_x2_v45_base_att2_std2':
            model = EDSR_with_att2_std2(scale=scale_factor).to(device)
        else:
            model = EDSR(scale=scale_factor).to(device)
        model.load_state_dict(weights[k])
        model.eval()

        for image in tqdm(set5):
            lr, hr = get_tensor(image)

            if k == 'Freq_Fusion':
                lr_high, lr_low = pass_filter(lr)
                sr, out_img, out_hf, out_lf = model(lr, lr_high, lr_low)
            elif 'hm' in k:
                lr_hf = high_pass_filter_hard_kernel(lr)
                sr, _ = model(lr, lr_hf)
            else:
                sr, fea = model(lr)

            results[image][k] = sr
            evals['psnr'][image.split('/')[-1]][k] = get_psnr(sr, hr)
            evals['ssim'][image.split('/')[-1]][k] = get_ssim(sr, hr).item()
            evals['ms_ssim'][image.split('/')[-1]][k] = get_msssim(sr, hr).item()
        del model
        torch.cuda.empty_cache()
        
for meas in ['psnr', 'ssim', 'ms_ssim']:
    evals[meas]['mean'] = dict()        
    for k in weights.keys():
            array = [evals[meas][image.split('/')[-1]][k] for image in set5]
            evals[meas]['mean'][k] = np.mean(np.array(array))
        
for image in set5:
    lr, hr = get_tensor(image)
    rname = result_dir + image.split('/')[-1]
    img = [torch.cat((lr, torch.zeros_like(lr)), dim=-2)] + [results[image][k] for k in weights.keys()] + [hr]
    imshow(img, filename=rname, visualize=False)
    
for meas in ['psnr', 'ssim', 'ms_ssim']:
    df = pd.DataFrame(evals[meas])
    df.to_csv(f'{result_dir}x{scale_factor}_{meas}.csv')


# In[ ]:


# display(pd.DataFrame(evals['psnr']))
# display(pd.DataFrame(evals['ssim']))
# display(pd.DataFrame(evals['ms_ssim']))
print(pd.DataFrame(evals['psnr']))
print(pd.DataFrame(evals['ssim']))
print(pd.DataFrame(evals['ms_ssim']))


# In[4]:


for meas in ['psnr', 'ssim', 'ms_ssim']:
    df = pd.DataFrame(evals[meas])
    df.to_csv(f'{result_dir}x{scale_factor}_{meas}.csv')


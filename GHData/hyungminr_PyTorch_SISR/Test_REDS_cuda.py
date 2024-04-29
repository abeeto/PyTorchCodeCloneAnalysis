#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

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
from models.RCAN import RCAN

from models.common import GMSD_quality

from utils import imshow
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr
from utils import pass_filter
from utils import high_pass_filter_hard_kernel
from utils import evaluate

torch.manual_seed(0)
scale_factor = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)
# model = EDSR(scale=scale_factor).to(device)

weights = {'EDSR': torch.load('./weights/Benchmark/EDSR_x4.pth'),
           'Freq_Fusion': torch.load('./weights/2021.01.20/EDSR_x4_v18_freq_fusion/epoch_1000.pth'),
           'RCAN': torch.load('./weights/Benchmark/RCAN_x4.pth')
          }

def get_tensor(lr, mode='set5'):
    trans = T.Compose([T.ToTensor()])
    if mode.lower() == 'set5':
        hr = lr.replace('LR_bicubic/X4', 'HR')
        hr = hr.replace('x4.png', '.png')
    elif mode.lower() == 'reds':
        hr = lr.replace('/val_blur_bicubic/X4/', '/val_sharp/')
    lr = Image.open(lr)
    lr = trans(lr)
    lr = lr.unsqueeze(0).to(device)
    hr = Image.open(hr)
    hr = trans(hr)
    hr = hr.unsqueeze(0).to(device)
    
    return lr, hr

# results = {s:{} for s in set5}
evals = {}
evals['psnr'] = {k:{} for k in weights.keys()}
evals['ssim'] = {k:{} for k in weights.keys()}
evals['ms_ssim'] = {k:{} for k in weights.keys()}

# dataset = 'Set5'
# data = glob.glob(f'./data/benchmark/Set5/LR_bicubic/X{scale_factor}/*.png')

dataset = 'Reds'
data = glob.glob(f'./data/benchmark/REDS/val/val_blur_bicubic/X{scale_factor}/*/*.png')

result_dir = f'./results/Benchmark/{dataset}/'
os.makedirs(result_dir, exist_ok=True)

with torch.no_grad():
    for ki, k in enumerate(weights.keys()):
        print(f'{ki+1} / {len(weights.keys())} || {k}')
        if k == 'Freq_Fusion':
            model = EDSR_fusion(scale=scale_factor).to(device)    
        elif k == 'RCAN':
            model = RCAN(scale=scale_factor).to(device)
        else:
            model = EDSR(scale=scale_factor).to(device)
        model.load_state_dict(weights[k])
        model.eval()
        pfix = OrderedDict()
        
        with tqdm(data, desc=f'{ki+1} / {len(weights.keys())} || {k}', position=0, leave=True) as pbar:
            for image in pbar:

                
                lr, hr = get_tensor(image, dataset)

                if k == 'Freq_Fusion':
                    lr_high, lr_low = pass_filter(lr)
                    sr, out_img, out_hf, out_lf = model(lr, lr_high, lr_low)
                elif 'hm' in k:
                    lr_hf = high_pass_filter_hard_kernel(lr)
                    sr, _ = model(lr, lr_hf)
                else:
                    sr, fea = model(lr)

                psnr, ssim, ms_ssim = evaluate(sr, hr)
                pfix['psnr'] = f'{psnr:.4f}'
                pfix['ssim'] = f'{ssim:.4f}'
                pfix['ms_ssim'] = f'{ms_ssim:.4f}'
                    
                # results[image][k] = sr
                evals['psnr'][k][image.split('/')[-1]] = psnr
                evals['ssim'][k][image.split('/')[-1]] = ssim
                evals['ms_ssim'][k][image.split('/')[-1]] = ms_ssim
                pbar.set_postfix(pfix)
        del model
        torch.cuda.empty_cache()
        
for meas in ['psnr', 'ssim', 'ms_ssim']:      
    for k in weights.keys():
            array = [evals[meas][k][image.split('/')[-1]] for image in data]
            evals[meas][k]['mean'] = np.mean(np.array(array))
"""   
for image in set5:
    lr, hr = get_tensor(image)
    rname = result_dir + image.split('/')[-1]
    img = [torch.cat([lr] + [torch.zeros_like(lr) for _ in range(scale_factor-1)], dim=-2)] + [results[image][k] for k in weights.keys()] + [hr]
    imshow(img, filename=rname, visualize=False)
"""

for meas in ['psnr', 'ssim', 'ms_ssim']:
    df = pd.DataFrame(evals[meas])
    df.to_csv(f'{result_dir}{dataset}x{scale_factor}_{meas}.csv')


# In[ ]:


# display(pd.DataFrame(evals['psnr']))
# display(pd.DataFrame(evals['ssim']))
# display(pd.DataFrame(evals['ms_ssim']))
# print(pd.DataFrame(evals['psnr']))
# print(pd.DataFrame(evals['ssim']))
# print(pd.DataFrame(evals['ms_ssim']))


# In[4]:


for meas in ['psnr', 'ssim', 'ms_ssim']:
    df = pd.DataFrame(evals[meas])
    df.to_csv(f'{result_dir}x{scale_factor}_{meas}.csv')


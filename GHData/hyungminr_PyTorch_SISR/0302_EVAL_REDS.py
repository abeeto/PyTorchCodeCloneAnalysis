#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import glob
import os
from utils import evaluate, img2tensor

ddd = 'val_batch_32'

snames = glob.glob(f'./results/REDS/{ddd}/*.png')

psnrs = []
ssims = []
pfix = OrderedDict()
with tqdm(snames) as pbar:
    for sname in pbar:
        res_img = imageio.imread(sname)
        sname = sname.replace('_', '.').replace(f'{ddd}', 'val')
        ssplit = sname.split('/')[-1]

        ref_img = imageio.imread('./data/benchmark/REDS/val/val_sharp_9/' + ssplit.split('.')[0] + '_' + ssplit.split('.')[1] + '.png')

        psnr = peak_signal_noise_ratio(ref_img, res_img)
        ssim = structural_similarity(ref_img, res_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)

        psnrs.append(psnr)
        ssims.append(ssim)
        
        # pfix['file'] = ssplit
        pfix['PSNR'] = f'{psnr:.2f}'
        pfix['SSIM'] = f'{ssim:.4f}'
        mpsnr = np.array(psnrs).mean()
        mssim = np.array(ssims).mean()
        pfix['avg PSNR'] = f'{mpsnr:.2f}'
        pfix['avg SSIM'] = f'{mssim:.4f}'
        
        pbar.set_postfix(pfix)

print(np.array(psnrs).mean())

print(np.array(ssim).mean())


# In[ ]:





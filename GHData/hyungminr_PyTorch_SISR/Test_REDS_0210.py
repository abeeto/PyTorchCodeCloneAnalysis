#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from models.hmnet_heavy import hmnet
from utils.data_loader import get_loader
from utils import imshow
from utils import evaluate
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
scale_factor = 4
batch_size = 1


# In[2]:


device = 'cpu'
device = 'cuda'


# In[3]:


model = hmnet(scale=scale_factor).to(device)
model.load_state_dict(torch.load('./weights/2021.02.07/HMNET_x4_Heavy_REDS_batch_32/epoch_0060.pth'))
model.eval()

# In[4]:


val_loader = get_loader(data='REDS', height=0, width=0, scale_factor=4, mode='test', force_size=True)


# In[ ]:


evals = list()
pfix = OrderedDict()
with torch.no_grad():
    with tqdm(val_loader) as pbar:
        for lr, hr, f in pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            sr, _, _ = model(lr)
            psnr, ssim, msssim = evaluate(sr, hr)
            evals.append([psnr, ssim, msssim])
            
            fsplit = f[0].split('/')
            rname = './results/REDS/val/' + fsplit[-2] + '_' + fsplit[-1].replace('.pt', '.png')
            _ = imshow(sr, filename=rname, visualize=False)
                    
            pfix['PSNR'] = psnr
            pfix['SSIM'] = ssim
            pfix['MS-SSIM'] = msssim
            
            npevals = np.array(evals)
            
            pfix['avg PSNR'] = npevals[:,0].mean()
            pfix['avg SSIM'] = npevals[:,0].mean()
            pfix['avg MS-SSIM'] = npevals[:,0].mean()
            pbar.set_postfix(pfix)


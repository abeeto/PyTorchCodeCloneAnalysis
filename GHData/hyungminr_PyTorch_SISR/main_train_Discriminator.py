import os
import torch
import datetime
import time
import numpy as np
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from collections import OrderedDict
from utils import imsave, sec2time, get_gpu_memory
from utils.eval import ssim as get_ssim
from utils.eval import ms_ssim as get_msssim
from utils.eval import psnr as get_psnr
from utils.data_loader import get_loader

from models.Discriminator import VGG

def evaluate(hr: torch.tensor, sr: torch.tensor):
    batch_size, _, h, w = hr.shape
    psnrs, ssims, msssims = [], [], []
    for b in range(batch_size):
        psnrs.append(get_psnr(hr[b], sr[b]))
        ssims.append(get_ssim(hr[b].unsqueeze(0), sr[b].unsqueeze(0)).item())
        if h > 160 and w > 160:
            msssim = get_msssim(hr[b].unsqueeze(0), sr[b].unsqueeze(0)).item()
        else:
            msssim = 0
        msssims.append(msssim)    
    return np.array(psnrs).mean(), np.array(ssims).mean(), np.array(msssims).mean()

quantize = lambda x: x.mul(255).clamp(0, 255).round().div(255)

torch.manual_seed(0)

train_loader = get_loader(mode='train', batch_size=16, scale_factor=4, augment=True)
test_loader = get_loader(mode='test')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
up = torch.nn.UpsamplingBilinear2d(scale_factor=4).to(device)

def get_sr(lr, hr, alpha=0.1):
    return quantize(up(lr) * (1-alpha) + hr * alpha)
    
model = VGG(pretrained=True).to(device)



weight_dir = f'./weights/Discriminator'
os.makedirs(weight_dir, exist_ok=True)

params = list(model.parameters())
optim = torch.optim.Adam(params, lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma= 0.99)
criterion = torch.nn.L1Loss()

mode = 'Discriminator'
start_time = time.time()
print(f'Training Start || Mode: {mode}')

step = 0
pfix = OrderedDict()
pfix_test = OrderedDict()

alpha = 0.05

alpha_i = 0
not_best_since = 0
torch_seed = 0
torch.manual_seed(torch_seed)

epoch = 0
num_epochs = 1000
while alpha < 0.95 or epoch < num_epochs:

    if epoch == 0:
        if os.path.exists(f'{weight_dir}/model_best.pth'):
            model.load_state_dict(torch.load(f'{weight_dir}/model_best.pth'))
        else:
            torch.save(model.state_dict(), f'{weight_dir}/model_best.pth')
        
    loss_best = 1.0 / alpha
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=0, leave=True) as pbar:
        psnrs = []
        ssims = []
        msssims = []
        losses = []
        for lr, hr, _ in pbar:

            lr = lr.to(device)
            hr = hr.to(device)
            
            sr = get_sr(lr, hr, alpha)
            
            # prediction
            fake = model(sr)
            real = model(hr)

            # training
            loss = criterion(torch.zeros_like(fake, device=device), fake)
            loss += criterion(torch.ones_like(real, device=device), real)
            loss_tot = loss
            optim.zero_grad()
            loss_tot.backward()
            optim.step()
            scheduler.step()

            # training history 
            elapsed_time = time.time() - start_time
            elapsed = sec2time(elapsed_time)            
            pfix['Step'] = f'{step+1}'
            pfix['Not Best Since'] = not_best_since
            pfix['Loss'] = f'{loss.item():.4f}'

            sr = quantize(sr)
            psnr, ssim, msssim = evaluate(hr, sr)

            psnrs.append(psnr)
            ssims.append(ssim)
            msssims.append(msssim)

            psnr_mean = np.array(psnrs).mean()
            ssim_mean = np.array(ssims).mean()
            msssim_mean = np.array(msssims).mean()

            pfix['alpha'] = f'{alpha:.6f}'
            pfix['PSNR'] = f'{psnr:.4f}'
            pfix['SSIM'] = f'{ssim:.4f}'

            free_gpu = get_gpu_memory()[0]

            pfix['free GPU'] = f'{free_gpu}MiB'
            pfix['Elapsed'] = f'{elapsed}'

            pbar.set_postfix(pfix)
            step += 1
            
            if loss_best > loss.item() / alpha:
                loss_best = loss.item() / alpha
                alpha_best = alpha
                torch.save(model.state_dict(), f'{weight_dir}/model_best.pth')
                torch.save(model.state_dict(), f'{weight_dir}/model_{alpha_i}.pth')
            else:
                not_best_since += 1
            
            if loss.item() == 1 or not_best_since > 500:
                not_best_since = 0
                torch_seed += 1
                torch.manual_seed(torch_seed)
                model.load_state_dict(torch.load(f'{weight_dir}/model_best.pth'))
                
            if loss.item() < 0.1:
                alpha_i += 1   
                alpha = 1.0 - 1.1**(-alpha_i)
            
        epoch += 1

#!/usr/bin/env python3

# %%
from utils.utils import calculate_mean_std, AverageMeter
from utils import metrics
import os
from torch.utils.data import DataLoader
import tempfile
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation
import wandb
import torch
from torchvision import datasets, transforms
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

wandb.init(project='test', dir=tempfile.gettempdir())
print(f'Using cuda {torch.version.cuda}, cudnn version {torch.backends.cudnn.version()}, pytorch version {torch.__version__}')
trainset = datasets.CIFAR100('~/data', True, transform=transforms.ToTensor(), download=True)
# %%
wandb.log({'sample images': [wandb.Image(trainset[i][0], caption=trainset[i][1]) for i in range(32)]})
# %%
train_loader = DataLoader(trainset, 1024, num_workers=os.cpu_count())
[mean, std] = calculate_mean_std(train_loader)
print(f'Dataset attributes : {mean}, {std}')
# %%
trainset = datasets.CIFAR100('~/data', True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]), download=True)

train_loader = DataLoader(trainset, 1024, num_workers=os.cpu_count(), shuffle=True)

wandb.log({'augmented training images': [wandb.Image(i) for i in next(iter(train_loader))[0]]})

validset = datasets.CIFAR100('~/data', False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]), download=True)

valid_loader = DataLoader(validset, 1024, num_workers=os.cpu_count())
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from utils.model import SomeNet

criterion = nn.CrossEntropyLoss()
_metrics_to_collect = {'loss':criterion, 'accuracy':metrics.accuracy}
_train_metrics={k:AverageMeter(f'train_{k}') for k in _metrics_to_collect.keys()}
_valid_metrics={k:AverageMeter(f'valid_{k}') for k in _metrics_to_collect.keys()}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=SomeNet(input_shape=trainset[0][0].shape, output_shape=len(trainset.classes))
model=nn.DataParallel(model).to(device) if torch.cuda.device_count() > 1 else model.to(device) 
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)
# lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-2, cycle_momentum=False)

epochs=100
for e in range(epochs+1):
    wandb.log({'epoch':e, 'lr':[p['lr'] for p in optimizer.param_groups][-1]})

    model=model.train()
    with tqdm(train_loader, desc=f'[{e}/{epochs}] Train', ascii=True) as progress, torch.enable_grad():
        for i, l in progress:
            i, l = i.to(device), l.to(device)
            optimizer.zero_grad()
            predictions=model(i)
            losses=criterion(predictions, l)
            losses.backward()
            optimizer.step()
            
            for i,(k,v) in enumerate(_metrics_to_collect.items()):
                _train_metrics[k].update(v(predictions, l))

            log_dict={v.name:v.avg for v in _train_metrics.values()}
            wandb.log(log_dict)
            progress.set_postfix(log_dict)
        for m in _train_metrics.values():
            m.commit()

    model=model.eval()
    with tqdm(valid_loader, desc=f'[{e}/{epochs}] Valid', ascii=True) as progress, torch.no_grad():
        for i, l in progress:
            i, l = i.to(device), l.to(device)
            optimizer.zero_grad()
            predictions=model(i)
            
            for i,(k,v) in enumerate(_metrics_to_collect.items()):
                _valid_metrics[k].update(v(predictions, l))
            
            log_dict={v.name:v.avg for v in _valid_metrics.values()}
            wandb.log(log_dict)
            progress.set_postfix(log_dict)
        for m in _valid_metrics.values():
            m.commit()
    lr_scheduler.step(_valid_metrics['loss'].avgs[-1])
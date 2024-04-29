import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

# TODO: Import models and datasets
from utils.util import AverageMeter
from utils.options import parse_args
from dataset import create_dataset

import pdb

###################
# Configs / Setup
###################

opt = parse_args()
start_epoch = 0

# Make directories
save_dir = Path('results/chkpts') / opt.expr_name
save_dir.mkdir(exist_ok=True, parents=True)

# Setting up Logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] [%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'train.log'),
            logging.StreamHandler()
        ])

########
# Data
########

logging.info('Creating dataloaders')

trainset, testset = create_dataset(opt)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                          shuffle=opt.shuffle, num_workers=opt.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=opt.num_workers)

#########
# Model
#########

logging.info('Creating models')

model = # TODO
model = model.to(opt.device[0])
optimizer = Adam(model.parameters(), lr=lr)
criterion = # TODO

if resume:
    chkpt_path = str(save_dir / 'chkpts' / 'latest.pth')
    logging.info(f'Resuming from {chkpt_path}')

    chkpt = torch.load(chkpt_path)
    model.load_state_dict(chkpt['state_dict'])
    start_epoch = chkpt['epoch'] + 1
    model.load_state_dict(state_dict['state_dict'])
    model.load_optimizer_state_dict(state_dict['optimizer_state_dict'])

################
# Init Logging
################

loss_hist = AverageMeter()

# Tensorboard
if opt.save:
    logging.info('Setting up tensorboard')
    if resume:
        writer = SummaryWriter(logdir=state_dict['logdir'])
    else:
        # Save in format: 'runs/{expr_name}/{run_idx}'
        runs_dir = Path('runs') / opt.expr_name
        if runs_dir.exists():
            run_idx = len(os.listdir(runs_dir))
        else:
            run_idx = 0
        logdir = runs_dir / f'{run_idx:03d}'
        writer = SummaryWriter(logdir=logdir)

for epoch in range(start_epoch, num_epochs):
    logging.info(f'Starting epoch {epoch}')

    for data_idx, (im, tgt) in tqdm(enumerate(trainloader), total=len(trainloader)):
        im, tgt = im.to(opt.device[0]), tgt.cuda(opt.device[0])

        pred = model(im)

        optimizer.zero_grad()
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()

        ###########
        # Logging
        ###########
        loss_hist.update(loss.item())

        #######
        # Log
        #######

        # Write to tensorboard
        writer.add_scalar(f'loss', loss.item(), i + epoch * len(trainloader))

    # Write to stdout
    print(f'Epoch: {epoch} | Loss: {loss_hist.avg:.04f}')
    loss_hist.reset()

    ########
    # Save
    ########
    if opt.save:
        logging.info(f'Saving epoch {epoch} model')
        to_save = {'state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'epoch': epoch,
                   'logdir': writer.logdir,
                   'expr_name': expr_name}
        torch.save(to_save, save_dir / 'chkpts' / f'latest.pth')

        if epoch % save_freq == 0:
            torch.save(to_save, save_dir / 'chkpts' / f'epoch_{epoch:04d}.pth')

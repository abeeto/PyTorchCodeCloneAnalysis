# %%
import time, os, json, string, re, datetime
import argparse
import numpy as np

import torch
from torch.nn.modules.activation import GELU
from torch.nn.modules.linear import Linear
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from functools import partial

from torchvision.transforms.functional import InterpolationMode
# from xcit import XCiT

# from utils_progressbar import *
from utils_args import ARGS
# from utils_smi import NVIDIA_SMI

from utils_datasets import Datasets
from utils_dataset_tire import get_tire_dataset
from models.vision_all import VisionModelZoo
from utils_network import Network

from multiprocessing import Pool

# %%

time_stamp = time.strftime('%y%m%d_%H%M%S')

config = [
    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  100,  int, None, 'number of training epochs'),
    ('dataset', 'stl10', str, None, 'name of the dataset'),
    ('data_path', '/host/ubuntu/torch', str, None, 'path to the local image folder'),
    ('bs', 128, int, None, 'batch size'),
    ('root_path', '/host/ubuntu/torch', str, None, 'path of the folder to put the pretrained models and download datasets'),
    ('arch', 'swin_base_patch4_window7_224', str, None, 'backbone network architecture'),
    ('lr', 0.001, float, None, 'initial learning rate'),
    ('lr_scheduler', 'step', str, ['none', 'step', 'exp', 'cos', 'ca', 'cos_exp'], 'type of lr scheduler'),
    ('lr_step', 10, int, None, 'the number of epochs between each scheduling step'),
    ('lr_gamma', 0.5, float, None, 'the rate of reducing for the learning rate'),
    ('lr_scale', 0.1, float, None, 'the min scale ratio for some scheduler'),
    ('limit_train', 0, int, None, 'set to int >0 to limit the number of training samples'),
    ('limit_test', 0, int, None, 'set to int >0 to limit the number of testing samples'),
    # ('stats_json', ''.format(time_stamp), str),
    ('stats_fp', './logs/massA/stats_{}.json'.format(time_stamp), str),
    ('lineareval', False, bool, None, 'include to set training mode to be similar to lineareval protocol, the backbone model is frozen, only the classifier head is finetuned'),
    ('earlystop_epoch', 5, int, None, 'the number of epochs without improvement to stop the training process early'),
    ('pretrained', False, bool, None, 'include to load the pretrained model from arch, note that it is not available for all archs'),
    ('note', '', str, None, 'note to recognize the run'),
    ('opt', 'sgd', str, None, 'set the optimizer'),
    ('fc', [], int, None, 'the units for the additional fc layers'),
    # ('aug_auto_imagenet', 1, int, [0, 1]),
    # ('aug_random_crop', 1, int, [0, 1]),
    # ('aug_color_jitter', 1, int, [0, 1]),
    ('image_size', 0, int, None, 'size to resize the input image to, defaults to 0 meaning image is untouch'),
    # ('tire_settings', 0, int, None, 'settings [0-3] for tire dataset preprocessing'),
]

A = ARGS(config=config)

A.set_and_parse_args('')

args = A.args
print('args:', json.dumps(A.info, indent=4))



# %%
ds = Datasets(
    dataset=args['dataset'],
    image_size=args['image_size'],
    root_path=args['root_path'],
    batchsize=args['bs'],
    # transform_pre=[],
    download=True,
    # splits=['train', 'test'],
    shuffle=True,
    num_workers=8,
    limit_train=args['limit_train'],
    limit_test=args['limit_test'],
)

print(ds.info)

# %%
models = VisionModelZoo.get_model(
    arch=args['arch'],
    pretrained=args['pretrained'],
    image_channels=3,
    classifier=[
        *args['fc'],
        ds.num_labels,
    ],
    root_path=args['root_path'],
    return_separate=bool(args['lineareval']),
)
_run_mode = '<unknown>'
if args['lineareval']:
    # models is tuple(model_bb, cls_head)
    frozen_model_bottom, model = models
    _run_mode = 'lineareval'
else:
    frozen_model_bottom = []
    model = models
    _run_mode = 'finetune'

# %%

_telem = {
    'hardware': '1x3090',
    'sample_count_train': ds.info['sample_count']['train'],
    'sample_count_val': ds.info['sample_count']['test'],
    'completed': False,
    'time_stamp': time_stamp,
    'time_start': None,
    'time_finish': None,
    'time_elapsed': None,
    'mode': _run_mode,
}


# %%
net = Network(
    model=model,
    frozen_model_bottom=frozen_model_bottom,
    # opt='adamw',
    opt=args['opt'],
    loss_fn=nn.CrossEntropyLoss(),
    lr=args['lr'],
    lr_type=args['lr_scheduler'],
    lr_step=args['lr_step'],
    lr_gamma=args['lr_gamma'],
    lr_scale=args['lr_scale'],
    device=args['device'],
    stats_fp=args['stats_fp'],
    epochs=args['epoch'],
    earlystop_epoch=args['earlystop_epoch'],
    
    metrics=['lr'],
    info={**args},
    telem=_telem,
    splits=['train', 'val'],
    use_default_acc=True,
    use_default_loss=True,
    
    ds=ds,
)
net

# %%

# %%
net.fit(
    # dataloader_train=ds.loaders['train'],
    # dataloader_val=ds.loaders['test'],
    # epochs=args['epoch'],
    # # time_stamp=time_stamp,
    # earlystop_epoch=args['earlystop_epoch'],
    
    # dataloaders={},
    # epochs=10,
    # epoch_start=0,
    # earlystop_epoch=10,
    # val_splits=['train'],
)

# %%
print('\n\nDONE')
# %%



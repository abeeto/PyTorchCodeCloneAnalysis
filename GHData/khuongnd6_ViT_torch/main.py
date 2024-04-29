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


# %%
time_start_master = time.time()
time_stamp = time.strftime('%y%m%d_%H%M%S')

# %%
# config = [
#     ('device',  'cuda',  str,  ['cuda', 'cpu']),
#     ('epoch',  100,  int, None, 'number of training epochs'),
#     ('dataset', 'tire', str, None, 'name of the dataset'),
#     ('data_path', '/host/ubuntu/torch/tire/tire_500', str, None, 'path to the local image folder'),
#     ('bs', 512, int, None, 'batch size'),
#     ('root_path', '/host/ubuntu/torch', str, None, 'path of the folder to put the pretrained models and download datasets'),
#     ('arch', 'dino_vits16', str, None, 'backbone network architecture'),
#     ('lr', 0.001, float, None, 'initial learning rate'),
#     ('lr_schedule_half', 10, int, None, 'number of epochs between halving the learning rate'),
#     ('limit_train', 0, int, None, 'set to int >0 to limit the number of training samples'),
#     ('limit_test', 0, int, None, 'set to int >0 to limit the number of testing samples'),
#     ('stats_json', ''.format(time_stamp), str),
#     ('master_stats_json', './logs/stats_master.json', str),
#     # ('test_only', False, bool),
#     # ('train_only', False, bool),
#     ('lineareval', False, bool),
#     # ('earlystopping', True, bool),
#     ('pretrained', False, bool),
#     ('note', '', str, None, 'note to recognize the run'),
#     ('opt', 'sgd', str, None, 'set the optimizer'),
#     ('fc', [], int, None, 'the units for the additional fc layers'),
#     # ('multiple_frozen', 1, int, None, 'multiply the frozen train dataset by this amount'),
#     # ('update_frozen_rate', 2, int, None, 'set to int >0 to update the frozen values every <this> epochs'),
#     ('aug_auto_imagenet', 1, int, [0, 1]),
#     # ('aug_random_crop', 1, int, [0, 1]),
#     ('aug_color_jitter', 1, int, [0, 1]),
#     ('image_size', 0, int),
#     ('tire_settings', 0, int, None, 'settings [0-3] for tire dataset preprocessing'),
# ]

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

def main():
    A = ARGS(config=config)
    
    A.set_and_parse_args('')

    args = A.args
    print('args:', json.dumps(A.info, indent=4))
    
    os.environ['TORCH_HOME'] = args['root_path'] # '/host/ubuntu/torch'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # smi = NVIDIA_SMI(device_id=0)
    
    _image_channels = 3
    if args['dataset'] in Datasets.available_datasets:
        # _transform_resize = []
        # if args['image_size'] > 0:
        #     _transform_resize = [transforms.Resize(args['image_size'],InterpolationMode.BICUBIC)]
        ds = Datasets(
            dataset=args['dataset'],
            image_size=args['image_size'],
            root_path=args['root_path'],
            batchsize=args['bs'],
            # transform_pre=[],
            download=True,
            splits=['train', 'test'],
            shuffle=True,
            num_workers=4,
            limit_train=args['limit_train'],
            limit_test=args['limit_test'],
        )
    elif args['dataset'] == 'tire':
        tire_kwargs = {
            'zoom_amount': 2.0,
            'random_crop_amount': 1.2,
            'random_vflip': True,
            'random_hflip': True,
        }
        lbp_methods = ['l', 'default', 'uniform']
        if args['tire_settings'] == 1:
            pass
        elif args['tire_settings'] == 2:
            tire_kwargs['zoom_amount'] = 2.4
        elif args['tire_settings'] == 3:
            tire_kwargs['zoom_amount'] = 2.4
            tire_kwargs['random_crop_amount'] = 1.6
        elif args['tire_settings'] == 0 or True:
            args['tire_settings'] == 0
            lbp_methods = ['r', 'g', 'b', 'default', 'uniform', 'ror', 'nri_uniform']
        _lbp_dict = {'radius': 2, 'point_mult': 8, 'methods': lbp_methods}
        
        print('tire settings:', json.dumps(tire_kwargs), json.dumps(_lbp_dict))
        _image_channels = len(lbp_methods)
        assert args['image_size'] > 0, 'must provide arg `image_size` of int >0'
        ds = get_tire_dataset(
            data_path=args['data_path'],
            batchsize=args['bs'],
            shuffle=True,
            num_workers=16,
            train_ratio=0.8,
            limit=0,
            image_size=args['image_size'],
            # force_reload=False,
            lbp=_lbp_dict,
            # random_crop=bool(args['aug_random_crop']),
            color_jitter=bool(args['aug_color_jitter']),
            autoaugment_imagenet=bool(args['aug_auto_imagenet']),
            
            fill=128,
            
            **tire_kwargs,
            # zoom_amount=2.0,
            # random_crop_amount=1.2,
            # random_hflip=True,
            # random_vflip=True,
        )
    else:
        raise ValueError('arg `dataset` [{}] is not supported!'.format(args['dataset']))
    print('dataset: {}'.format(json.dumps(ds.info, indent=4)))
    
    _run_mode = '<unknown>'
    if args['lineareval']:
        _run_mode = 'lineareval'
        _model_backbone = VisionModelZoo.get_model(
            args['arch'],
            pretrained=args['pretrained'],
            image_channels=_image_channels,
            classifier=None,
            root_path=args['root_path']
        )
        _input_shape = [1, _image_channels, args['image_size'], args['image_size']]
        _output_dim = VisionModelZoo.get_output_shape(_model_backbone, _input_shape, args['device'])[-1]
        
        frozen_model_bottom = [_model_backbone]
        model = VisionModelZoo.get_classifier_head(
            _output_dim,
            [*args['fc'], ds.num_labels],
            # GELU(),
        )
    else:
        _run_mode = 'finetune'
        frozen_model_bottom = []
        model = VisionModelZoo.get_model(
            args['arch'],
            pretrained=args['pretrained'],
            image_channels=_image_channels,
            classifier=[*args['fc'], ds.num_labels],
        )
    

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
        
        # 'hardware': '1x3090',
        # # 'gpu_total_memory': float(smi.info['total']),
        # 'sample_count_train': ds.info['sample_count']['train'],
        # 'sample_count_val': ds.info['sample_count']['test'],
        # 'completed': False,
        # 'time_stamp': time_stamp,
        # 'time_start': time_start,
        # 'time_finish': None,
        # 'time_elapsed': None,
        # 'mode': run_mode,
        # **({
        #     **tire_kwargs,
        #     'lbp': {**_lbp_dict},
        # } if args['dataset'] == 'tire' else {}),
    }
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
    
    run_mode = 'finetune'
    if args.get('lineareval'):
        run_mode = 'lineareval'

    # time_start = time.time()
    # stats = {
    #     'info': {**args},
    #     'telem': _telem,
    #     **{_split: [] for _split in ['train', 'val']},
    # }
    net.fit(
        # dataloader_train=ds.loaders['train'],
        # dataloader_val=ds.loaders['test'],
        # epochs=args['epoch'],
        # # epoch_start=0,
        # # fp_json_master=args['master_stats_json'],
        # time_stamp=time_stamp,
        # stats=stats,
    )

    
# %%
if __name__ == '__main__':
    main()
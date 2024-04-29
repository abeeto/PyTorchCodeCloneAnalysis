#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
# Created on 27 15:09

"""
Configuration of model specification, training and tracking

For most of the time, DO NOT modify the configurations within this file.
Use the configurations here as the default configurations and only update
them following the examples in the `experiments` directory.
"""

import os.path as osp
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

HOME = os.environ['HOME']
PROJ_ROOT = osp.dirname(__file__)
CONF_FN = osp.abspath(__file__)

RUN_NAME = 'GOTURN-ALEXNET-ORI' # Idenitifier of the experiment
EXP_DIR = osp.join(PROJ_ROOT, 'experiments')      # Where checkpoints, logs are saved
DATASET_PREFIX = osp.join(PROJ_ROOT, 'data')

DATA_CONFIG = {
    'alov_vdroot': osp.join(DATASET_PREFIX, 'ALOV300/train/images'),
    'alov_adroot': osp.join(DATASET_PREFIX, 'ALOV300/train/gt'),
    'det2014_imroot': osp.join(DATASET_PREFIX, 'DET2014/images'),
    'det2014_adroot': osp.join(DATASET_PREFIX, 'DET2014/gt'),
    'det_dbfn': osp.join(DATASET_PREFIX, 'det_dbfn.pickle'),
    'vot2015_imroot': osp.join(DATASET_PREFIX, 'VOT2015'),
}


MODEL_CONFIG = {
    'input_size': 227,          # The network input size
    'bbox_scale': 10,
    #'model_id': 'goturn',
    'model_id': 'gocorr',
    'init_model_dir': osp.join(PROJ_ROOT, "models", "init"),     # Contains alexnet and fc layers
    'size_average': False
}

TRAIN_CONFIG = {
    'exp_dir': osp.join(EXP_DIR, RUN_NAME),
    'seed': 800,        # Fix seed for reproducing experiments
    'init_lr': 1e-6,
    'bs': 50,
    'kGenPerImage': 10,
    'shuffle': True,
    'resume_checkpoint': None,
    #'resume_checkpoint': osp.join(EXP_DIR, RUN_NAME, "Iteration_14000.pth.tar"),
    'n_iter': 500000,
    'stepsize': 100000,
    'num_workers': 8,           # Decraped
    'use_gpu': True,
    'logfile': osp.join(EXP_DIR, RUN_NAME, "experiment.log"),
    'momentum': 0.9,
    'weight_decay': 5e-4,       # Note only w has weight_decay, bias has no weight_decay
    'lr_gamma': 0.1,
    'dump_freq': 10000,     # How many iteration interval to dump model param
    'valid_freq': 5,
    'motionparam': {
        'lambda_scale_frac': 15,
        'lambda_shift_frac': 5,
        'min_scale': -0.4,
        'max_scale': 0.4
    },
}

TRACK_CONFIG = {
    'model': osp.join(EXP_DIR, RUN_NAME, "model_best.pth.tar"),
    'use_pretrained_model': False,
    'logfile': osp.join(EXP_DIR, RUN_NAME, "track.log"),
    'use_gpu': True
}

if __name__ == "__main__":
    import json
    print (json.dumps(DATA_CONFIG, indent=1))


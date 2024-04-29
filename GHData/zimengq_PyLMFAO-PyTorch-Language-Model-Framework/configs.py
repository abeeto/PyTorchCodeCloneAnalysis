#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
Training and model configurations
"""

import os
import argparse
import torch
from datetime import datetime

# training configurations
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='Model type.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batch', type=int, default=20, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--pad', action='store_false', default=True, help='Padding each sentence in batch to same length.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--tie_weights', action='store_true', default=False, help='Optionally tie weights.')
parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=2, help='Number of RNN/Transformer layers.')
parser.add_argument('--head', type=int, default=2, help='Number of head for multi-head attention.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU id, -1 for using cpu.')
parser.add_argument('--embed', type=str, default=None, help='Pre-trained embedding path.')
parser.add_argument('--embed_size', type=int, default=300, help='Embedding size.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature for generating sentences - higher will increase diversity')
args = parser.parse_args()

if not os.path.exists('logs'):
    try:
        os.mkdir('logs')
    except IOError:
        print("Can not create log file directory.")
args.save = 'logs/{}-lr{}-hid{}-dp{}'.format(
    args.model,
    args.lr,
    args.hidden,
    args.dropout
)

args.models_dir = 'models/{}-lr{}-hid{}-dp{}'.format(
    args.model,
    args.lr,
    args.hidden,
    args.dropout
)

# running configurations
log_dir = args.save
timestamp = datetime.now().strftime('%m%d-%H%M%S')
# names = ('train', 'dev') if args.mode == 'train' else ('test',)

use_cuda = args.gpu_id != -1 and torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')
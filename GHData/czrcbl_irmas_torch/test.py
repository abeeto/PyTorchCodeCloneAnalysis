import os
from os.path import join as pjoin
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as tdata
import torchvision
# from torchsummary import summary
# from ignite.metrics import Accuracy

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, 
    confusion_matrix, classification_report)
import logging
import argparse
from copy import deepcopy
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import pickle

from src.datasets import IRMAS
from src.utils import get_network
from src import config as cfg
from src import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--dataset-name',type=str,
                        help='Dataset to use.')
    parser.add_argument('--model-name',type=str,
                        help='Experiment to use.')
    parser.add_argument('--strategy', type=str, default='all',
                        help='Evaluation strategy.')
#    parser.add_argument('--batch-size', type=int, default=1,
#                        help='Inference batch size.')
    parser.add_argument('--device',type=str, default='cuda',
                        help='Device to use.')
    parser.add_argument('--is-test', type=int, default=0,
                        help='Whether it is a test.')
    parser.add_argument('--store-activations', action='store_true')
    
    args = parser.parse_args()
    return args

def save_output(output, args, strategy):
    with open(pjoin(args.exp_path, f'output_strategy-{strategy}.pkl'), 'wb') as f:
        pickle.dump(output, f)

def compute_metrics(y_pred, y_true, strategy, logger):

    logger.info('Computing Metrics')
    logger.info('Strategy %d' % strategy)
    logger.info(classification_report(y_true, np.round(y_pred), target_names=cfg.classes))
    logger.info('Accuracy: %.3f' % accuracy_score(y_true, np.round(y_pred)))
    logger.info('mAP: %.3f' % average_precision_score(y_true, y_pred))


def test_strategy1(net, test_ds, device, args, logger):
    """Apply the network to the whole audio excerpt"""
    logger.info('Starting strategy 1')
    net = net.to(device)
    output = []
    net.eval()
    with torch.no_grad():
        for data, target in tqdm(test_ds):
            data = data[None, :, :, :]
            data = data.to(device)
            y = net(data)
            y = F.sigmoid(y)
            output.append((y.cpu().numpy(), target.numpy()))
    
    save_output(output, args, 1)

    # reshape Data
    y_pred = np.vstack(o[0] for o in output)
    y_true = np.vstack(o[1] for o in output)

    compute_metrics(y_pred, y_true, 1, logger)

   
def test_strategy2(net, test_ds, device, n_frames, args, logger):
    """Apply the network to slices of size of the time_slice of the train dataset."""
    logger.info('Starting strategy 2')
    net = net.to(device)
    output = []
    net.eval()
    with torch.no_grad():
        for data, target in tqdm(test_ds):
            data = data[None, :, :, :]
            data = data.to(device)
            ns = data.shape[-1] // n_frames
            out = []
            for i in range(ns):  
                y = net(data[:, :, :, i * n_frames: (i + 1) * n_frames])
                y = F.sigmoid(y)
                out.append(y.cpu().numpy())
#           
            output.append((out, target.numpy()))
            
    save_output(output, args, 2)

    y_pred = []
    y_true = []
    
    # # Simplily stack the data
    # for ele in output:
    #     t = ele[1]
    #     for p in ele[0]:
    #         y_pred.append(p)
    #         y_true.append(t)



    # Vote - Most common label
    for ele in output:
        counts = np.bincount(ele[0])
        y_pred.append(np.argmax(counts))
        y_true.append(ele[1])
                      
    
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)            
    
    compute_metrics(y_pred, y_true, 2, logger)
    
      
#def load_experiment(exp_name):
#    exp_path = pjoin(cfg.models_path, exp_name)
#    with open(pjoin(exp_path, 'parameters.json'), 'r') as f:
#        params = json.load(f)
#    audio_params = ['fs', 'n_fft', 'hop_length', 'n_mels']
#    audio_params = {key:params[key] for key in audio_params}
#    test_ds = IRMAS(mode='test', **audio_params)
#    net_params = edict({
#            'base_network': params['base_network'],
#            'transfer': False,
#            'mono':params['mono']})
#    net = get_network(net_params)
#    net.load_state_dict(torch.load(pjoin(exp_path, 'best_model.pth')))
#    return net, test_ds


def load_experiment(args):
    
    try:
        ds_num = int(args.dataset_name)
        ds_list = sorted(os.listdir(cfg.results_path))
        dataset_name = ds_list[ds_num]
    except ValueError:
        dataset_name = args.dataset_name
    
    try:
        model_num = int(args.model_name)
        model_list = sorted(os.listdir(pjoin(cfg.results_path, dataset_name)))
        model_name = model_list[model_num]
    except ValueError:
        model_name = args.model_name
    
    exp_path = pjoin(cfg.results_path, dataset_name, model_name)
    with open(pjoin(exp_path, 'parameters.json'), 'r') as f:
        params = json.load(f)
        
    audio_params = {key:params[key] for key in cfg.dataset_params}
    audio_params['time_slice'] = None
    net_params = {key:params[key] for key in cfg.network_params}

    if net_params['transfer']:
        trans = [transforms.AsImageTrans()]
    else:
        trans = None
   
    test_ds = IRMAS(mode='test', is_test=args.is_test, transforms=trans, **audio_params)
    net = get_network(net_params['base_network'], net_params['transfer'], net_params['mono'])
    net.load_state_dict(torch.load(pjoin(exp_path, 'best_model.pth')))
    args.exp_path = exp_path

    return net, test_ds, params


def main():
    
    args = parse_args()
    net, test_ds, params = load_experiment(args)
    # setup logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = pjoin(args.exp_path, 'test.log')
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)

    device = torch.device(args.device)
    
    ts = params['time_slice']
    hop_length = params['hop_length']
    fs = params['fs']
    n_frames = (fs * ts) // hop_length + 1
    
    if args.strategy == '1':
        test_strategy1(net, test_ds, device, args, logger)
    elif args.strategy == '2':
        test_strategy2(net, test_ds, device, n_frames, args, logger)
    elif args.strategy == 'all':
        test_strategy1(net, test_ds, device, args, logger)
        test_strategy2(net, test_ds, device, n_frames, args, logger)


if __name__ == '__main__':
    sys.exit(main())
    


import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as tdata
import torchvision
from torchsummary import summary
from ignite import metrics
from sklearn.metrics import average_precision_score, f1_score
import numpy as np
import os
from os.path import join as pjoin
import sys
import subprocess
import logging
import argparse
from copy import deepcopy
import json
from tqdm import tqdm

from src.utils import get_network
from src.datasets import IRMAS
from src import config as cfg
from src import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # misc
    parser.add_argument('--is-test',type=int, default=0,
                        help='Whether it is a test.')
    parser.add_argument('--results-folder', default=cfg.results_path,
                        help='Where to save the model.')
    parser.add_argument('--device', default='cuda',
                        help='Train device, default: cuda.')
    parser.add_argument('--model-name', default='',
                        help='Name of the model')
    parser.add_argument('--dataset-name', default='',
                        help='Name of the dataset.')
    ## Network args
    parser.add_argument('--base-network', default='resnet18',
                        help='Network to use as base.')
    parser.add_argument('--transfer', action='store_true',
                        help='Whether to use transfer learning.')
    
    ## Preprocessing/dataset params
    parser.add_argument('--fs', type=float, default=22050,
                        help='Final sampling rate.')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT size.')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop Length.')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel coeficients.')
    parser.add_argument('--mono', type=bool, default=True,
                        help='Whether to convert the audio to mono.')
    parser.add_argument('--time-slice', type=float, default=1,
                        help='Size of the spectrum in seconds.')
    parser.add_argument('--normalize', type=bool, default=True,
                        help='Whether to normilize the melspec.')
    ## Train 
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Train learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Train and validation batch size.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of cpu workers.')
    # Test
    parser.add_argument('--run_test', default=1, type=int,
                        help='Whether to run the test script.')
    args = parser.parse_args()

    # Validate args
    if args.transfer and not args.mono:
        raise ValueError('Transfer shoud be used with mono.')

    return args


class ModelSaver:
    
    def __init__(self, save_path):
        self.best_score = 0 
        self.best_model = None
        self.save_path = save_path
        
    def update(self, net, score):
        if score > self.best_score:
            self.best_score = score
            self.best_params = deepcopy(net.state_dict())
            
    def close(self):
        torch.save(self.best_params, pjoin(self.save_path, 'best_model.pth'))
        
        
def get_datasets(args):
    if args.transfer:
        trans = [transforms.AsImageTrans()]
    else:
        trans = None
    trn_ds = IRMAS(mode='train', is_test=args.is_test, transforms=trans)
    val_ds = IRMAS(mode='val', is_test=args.is_test, transforms=trans)
    return trn_ds, val_ds


def get_dataloaders(trn_ds, val_ds, args):
    trn_loader = tdata.DataLoader(
        trn_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = tdata.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    return trn_loader, val_loader


def train(net, trn_loader, val_loader, optmizer, criterion, epochs, device, args):
    
    net = net.to(device)
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.output_path + '/train.log'
    log_dir = os.path.dirname(log_file_path)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    
    saver = ModelSaver(log_dir)
    accu_metric = metrics.Accuracy()

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        logger.info(f'Epoch {epoch} training starting.')
        for i, (data, target) in tqdm(enumerate(trn_loader)):
            data, target = data.to(device), target.to(device)
            optmizer.zero_grad()
            y = net(data)
            loss = criterion(y, target.float())
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
        train_loss /= i
        
        accu=0
        net.eval()
        y_true = []
        y_pred = []
        # accu_metric.reset()
        with torch.no_grad():
            val_loss = 0
            logger.info(f'Epoch {epoch} validation starting.')
            for i, (data, target) in tqdm(enumerate(trn_loader)):
                data, target = data.to(device), target.to(device)
                y = net(data)
                loss = criterion(y, target.float())
                val_loss += loss.item()
                y = torch.sigmoid(y)
                y_pred.append(y.cpu().numpy())
                y_true.append(target.cpu().numpy())
                accu_metric.update((y.round(), target))
                accu += torch.mean((y.argmax(axis=-1) == target.argmax(axis=-1)).float())
            val_loss /= i
            # TODO: Proper weight the accuracy
            accu /= i
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        mAP = average_precision_score(y_true, y_pred)
        f1score = f1_score(y_true, np.round(y_pred), average='weighted')
        msg = f'Epoch: {epoch}\n'\
            f'train_loss: {train_loss:.3f}\n'\
            f'validation_loss: {val_loss:.3f}\n'\
            f'single_class_accuracy: {accu:.3f}\n'\
            f'multiclass_class_accuracy: {accu_metric.compute():.3f}\n'\
            f'F1-score: {f1score:.3f}\n'\
            f'mAP {mAP:.3f}'
        logger.info(msg)
        saver.update(net, accu)
    
    saver.close()


def prepare_experiment(args):
    
    if not args.dataset_name:
        dataset_name = ''
        for param in cfg.dataset_params:
            dataset_name = dataset_name + '~{}-{}'.format(param, vars(args)[param])
        dataset_name = dataset_name[1:]
        args.dataset_name = dataset_name
    else:
        dataset_name = args.dataset_name

    # Change dataset name if it is a test
    if args.is_test:
        dataset_name = 'test'
        args.dataset_name = dataset_name
    
    dataset_path = pjoin(args.results_folder, dataset_name)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    
    
    if not args.model_name:
        model_name = ''
        for param in cfg.network_params:
            model_name = model_name + '~{}-{}'.format(param, vars(args)[param])
        model_name = model_name[1:]
        args.model_name = model_name
    else:
        model_name = args.model_name

    # Change model name if it is a test
    if args.is_test:
        model_name = 'test_0'
        i = 1
        while model_name in os.listdir(dataset_path):
            model_name = f'test_{i}'
            i += 1
    
    model_path = pjoin(dataset_path, model_name)
    if os.path.isdir(model_path):
        raise ValueError(f'Model {model_name} for dataset {dataset_name} already exist.')
    else:
        os.makedirs(model_path)
    args.model_name = model_name
    args.output_path = model_path
    
    
def main():
    
    args = parse_args()
    prepare_experiment(args)

    if args.is_test:
        args.epochs = 3
    with open(pjoin(args.output_path, 'parameters.json'), 'w') as f:
        json.dump(args.__dict__, f)
    trn_ds, val_ds = get_datasets(args)
    trn_loader, val_loader = get_dataloaders(trn_ds, val_ds, args)
    device = torch.device(args.device)
    net = get_network(args.base_network, args.transfer, args.mono)
    optmizer = optim.Adamax(net.parameters())
    criterion = nn.BCEWithLogitsLoss()
    train(net, trn_loader, val_loader, optmizer, criterion, args.epochs, device, args)

    if args.run_test:
        command = f"""
            python test.py
            --dataset-name {args.dataset_name}
            --model-name {args.model_name}
            --strategy all
            --device {args.device}
            --is-test {args.is_test}
            --store-activations
        """
        subprocess.call(command.split())

if __name__ == '__main__':
    sys.exit(main())
    


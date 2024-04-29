import csv
import logging
import math
import time
from  _datetime import datetime
import os
from collections import OrderedDict
import json

import torch
import numpy as np
import torchvision.datasets
from torchvision.utils import make_grid

from modules import tensorboard_utils
from modules.csv_utils_2 import CsvUtils2
from modules.file_utils import FileUtils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
from torch.utils.data import DataLoader
import torch_optimizer as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-sequence_name', default='sequence', type=str)
parser.add_argument('-run_name', default='run', type=str)

parser.add_argument('-model', default='model_3', type=str)
parser.add_argument('-dataset', default='dataset_1_emnist', type=str)

parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)

parser.add_argument('-optimizer', default='radam', type=str)

parser.add_argument('-loss_rec', default='mse', type=str)

parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-debug_batch_count', default=0, type=int) # 0 = release version

parser.add_argument('-embedding_size', default=32, type=int)

parser.add_argument('-gamma', default=0.0, type=float)
parser.add_argument('-C_0', default=0.0, type=float)
parser.add_argument('-C_n', default=5.0, type=float)
parser.add_argument('-C_interval', default=10000, type=int)
parser.add_argument('-C_start', default=0, type=int)

args, args_other = parser.parse_known_args()

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
path_artifacts = f'./artifacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_artifacts)
FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

CsvUtils2.create_global(path_sequence)
CsvUtils2.create_local(path_sequence, args.run_name)

summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=path_run
)

rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
rootLogger.level = logging.INFO #level

base_name = os.path.basename(path_sequence)
fileHandler = logging.FileHandler(f'{path_run}/log-{base_name}.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
DataSet = getattr(__import__('modules_core.' + args.dataset, fromlist=['DataSet']), 'DataSet')

logging.info(f'path_run: {path_run}')
logging.info(json.dumps(args.__dict__, indent=4))

if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    logging.info('cuda devices: {}'.format(torch.cuda.device_count()))

datasets = OrderedDict({
    'train': DataSet(is_train=True),
    'test': DataSet(is_train=False)
})

x, x_noisy, y = datasets['train'][0]
args.input_size = x.size()

dataloaders = OrderedDict({
    'train': DataLoader(datasets['train'], shuffle=True, batch_size=args.batch_size),
    'test': DataLoader(datasets['test'], shuffle=False, batch_size=args.batch_size)
})
model = Model(args).to(args.device)


# https://pypi.org/project/torch-optimizer/#radam
if args.optimizer == 'radam':
    optimizer = optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

def dict_list_append(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

metrics_best = {
    'best_test_loss': float('Inf'),
    'best_test_loss_dir': -1
}

count_batches = 0
for epoch in range(1, args.epochs+1):
    logging.info(f'epoch: {epoch}')

    metric_mean = {}
    metrics_list = {}
    timer_epoch = time.time()
    for mode, dataloader in dataloaders.items():

        if mode == 'train':
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            model = model.eval()
            torch.set_grad_enabled(False)

        if args.debug_batch_count != 0:
            count_batches = 0

        z_hist = []
        for x, x_noisy, _ in dataloader:

            # plt.subplot(2, 1, 1)
            # plt.imshow(np.transpose(make_grid(x).numpy(), (1, 2, 0)))
            # plt.subplot(2, 1, 2)
            # plt.imshow(np.transpose(make_grid(x_noisy).numpy(), (1, 2, 0)))
            # plt.show()

            if mode == 'train':
                count_batches += 1
            if args.debug_batch_count != 0 and count_batches > args.debug_batch_count: # for debugging
                break

            z, z_mu, z_sigma, y_prim = model.forward(x_noisy.to(args.device))

            if args.loss_rec == 'bce':
                loss_rec = torch.mean(x.to(args.device)*torch.log(y_prim+1e-8))
            else:
                loss_rec = torch.mean((x.to(args.device) - y_prim)**2)

            C = 0
            if args.C_n > args.C_0 and count_batches >= args.C_start:
                C = min(args.C_n, (args.C_n - args.C_0) * ((count_batches - args.C_start) / args.C_interval) + args.C_0)

            # univariate mu = 0, sigma = 1, but could be also either other mu & sigma or could use multivariate
            kl = -0.5*(2*torch.log(z_sigma + 1e-8) - z_sigma - z_mu**2 + 1)
            kl_means = torch.mean(kl, dim=0) # (32, )
            loss_kl = args.gamma * torch.abs(C - torch.sum(kl_means))

            loss = loss_rec + loss_kl

            loss_scalar = loss.cpu().item()
            loss_rec_scalar = loss_rec.cpu().item()
            loss_kl_scalar = loss_kl.cpu().item()
            if np.isnan(loss_scalar) or np.isinf(loss_scalar):
                logging.error(f'loss_scalar: {loss_scalar} loss_rec_scalar: {loss_rec_scalar} loss_kl_scalar: {loss_kl_scalar}')
                exit()

            if mode == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            z_hist.append(z.cpu().data.numpy())
            dict_list_append(metrics_list, f'{mode}_kl_means', torch.mean(kl_means).cpu().item())
            dict_list_append(metrics_list, f'{mode}_loss_rec', loss_rec_scalar)
            dict_list_append(metrics_list, f'{mode}_loss_kl', loss_kl_scalar)
            dict_list_append(metrics_list, f'{mode}_loss', loss_scalar)
            dict_list_append(metrics_list, f'{mode}_z_mu', torch.mean(z_mu).cpu().item())
            dict_list_append(metrics_list, f'{mode}_z_sigma', torch.mean(z_sigma).cpu().item())
            dict_list_append(metrics_list, f'{mode}_z', torch.mean(z).cpu().item())
            dict_list_append(metrics_list, f'{mode}_c', C)

        z_std = np.std(np.concatenate(z_hist, axis=0), axis=0).tolist()
        z_std_sort = sorted(z_std)
        summary_writer.add_histogram(f'{mode}_z_std', z_std_sort, global_step=epoch, bins=len(z_std_sort))

        fig = plt.figure()
        plt.imshow(torchvision.utils.make_grid(x.cpu().detach(), normalize=True).permute(1, 2, 0))
        summary_writer.add_figure(f'{mode}_x', fig, global_step=epoch)

        fig = plt.figure()
        plt.imshow(torchvision.utils.make_grid(x_noisy.cpu().detach(), normalize=True).permute(1, 2, 0))
        summary_writer.add_figure(f'{mode}_x_noisy', fig, global_step=epoch)

        fig = plt.figure()
        plt.imshow(torchvision.utils.make_grid(y_prim.cpu().detach(), normalize=True).permute(1, 2, 0))
        summary_writer.add_figure(f'{mode}_y_prim', fig, global_step=epoch)

    for key, value in metrics_list.items():
        value = np.mean(value)
        logging.info(f'{key}: {value}')
        metric_mean[key] = value

        for key_best in metrics_best.keys():
            if f'best_{key}' == key_best:
                direction = metrics_best[f'{key_best}_dir']
                if metrics_best[key_best] * direction < value * direction:
                    torch.save(model.state_dict(), f'{path_artifacts}/best-{key_best}-{args.run_name}.pt')
                    with open(f'{path_artifacts}/best-{key_best}-{args.run_name}.json', 'w') as fp:
                        json.dump(args.__dict__, fp, indent=4)
                    metrics_best[key_best] = value
                metric_mean[key_best] = metrics_best[key_best]

    metric_mean['epoch_hours'] = (time.time() - timer_epoch) / (60.0 * 60.0)
    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metric_mean,
        name=args.run_name,
        global_step=epoch
    )
    CsvUtils2.add_hparams(
        path_sequence,
        args.run_name,
        args.__dict__,
        metric_mean,
        epoch
    )
    summary_writer.flush()
summary_writer.close()





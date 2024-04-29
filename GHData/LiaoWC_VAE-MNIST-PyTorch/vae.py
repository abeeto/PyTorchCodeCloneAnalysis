import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse
from datetime import datetime


#
class DigitsDataset(Dataset):
    def __init__(self, images):
        self.data = images

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx] / 255.).view(1, self.data[0].shape[0], self.data[0].shape[1])


class Encoder(nn.Module):
    def __init__(self, z_dim, nef=2):
        super().__init__()
        self.z_dim = z_dim
        self.conv0 = nn.Conv2d(1, nef, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 14x14
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(nef, nef * 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))  # 6x6
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(nef * 2, nef * 4, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))  # 2x2
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(nef * 4, nef * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 1x1
        self.relu3 = nn.ReLU()
        self.conv_mu = nn.Conv2d(nef * 4, z_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))  # 1x1
        self.conv_logvar = nn.Conv2d(nef * 4, z_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.fc0 = nn.Linear(nef * 8 * (1 * 1), nef * 8)
        self.fc_mu = nn.Linear(nef * 8, z_dim)
        self.fc_logvar = nn.Linear(nef * 8, z_dim)
        #
        # self.fcs = nn.Sequential(
        #     nn.Linear(28 * 28, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        # )
        # self.fc_mu = nn.Linear(256, z_dim)
        # self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc0(x.view(bs, -1))
        return self.fc_mu(x), self.fc_logvar(x)
        #
        # x = self.fcs(x.view(x.shape[0], -1))
        # return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, z_dim, nef=2):
        super().__init__()
        self.z_dim = z_dim
        self.transpose_conv0 = nn.ConvTranspose2d(z_dim, nef * 8, kernel_size=(1, 1), stride=(2, 2),
                                                  padding=(0, 0))  # 1x1
        self.relu0 = nn.ReLU()
        self.transpose_conv1 = nn.ConvTranspose2d(nef * 8, nef * 4, kernel_size=(4, 4), stride=(2, 2),
                                                  padding=(1, 1))  # 6x6
        self.relu1 = nn.ReLU()
        self.transpose_conv2 = nn.ConvTranspose2d(nef * 4, nef * 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
        self.relu2 = nn.ReLU()
        self.transpose_conv3 = nn.ConvTranspose2d(nef * 2, nef, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
        self.relu3 = nn.ReLU()
        self.transpose_conv4 = nn.ConvTranspose2d(nef, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()
        #
        # self.fcs = nn.Sequential(
        #     nn.Linear(z_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 28 * 28),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1)
        x = self.transpose_conv0(x)
        x = self.relu0(x)
        x = self.transpose_conv1(x)
        x = self.relu1(x)
        x = self.transpose_conv2(x)
        x = self.relu2(x)
        x = self.transpose_conv3(x)
        x = self.relu3(x)
        x = self.transpose_conv4(x)
        x = self.sigmoid(x)
        return x
        #
        # x = x.view(x.shape[0], -1)
        # return self.fcs(x).view(x.shape[0], 1, 28, 28)


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def train(encoder,
          decoder,
          optimizer,
          device,
          start_epoch,
          n_epoch,
          train_dataloader,
          test_dataloader,
          test_epoch_freq: int,
          save_model_epoch_freq: int,
          use_dataparallel: bool,
          writer,
          model_path,
          plot_path,
          kld_opt,
          args):
    if use_dataparallel:
        encoder = nn.DataParallel(encoder).to(device)
        decoder = nn.DataParallel(decoder).to(device)
    else:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
    start_time = datetime.now()
    assert 1 <= start_epoch <= n_epoch
    for epoch in range(start_epoch, n_epoch + 1):
        encoder.train()
        decoder.train()
        train_loss = 0
        train_rec_loss = 0
        train_kld_loss = 0
        # KLD loss mode
        if kld_opt['KLD_MODE'] == 'monotonic':
            kld_weight = kld_opt['MAX_KLD_W'] * np.max((0., epoch - kld_opt['KLD_W_FROM'] + 1)) / (
                    kld_opt['KLD_W_TO'] - kld_opt['KLD_W_FROM'] + 1)
        elif kld_opt['KLD_MODE'] == 'fixed':
            kld_weight = kld_opt['FIXED_W']
        elif kld_opt['KLD_MODE'] == 'cyclical':
            if epoch >= kld_opt['KLD_START']:
                kld_weight = kld_opt['MAX_KLD_W'] * (
                        (epoch - kld_opt['KLD_START'] + 1) % kld_opt['KLD_LEN'] / kld_opt['KLD_LEN'])
                kld_weight = kld_opt['MAX_KLD_W'] if kld_weight == 0. else kld_weight
            else:
                kld_weight = 0.
        else:
            raise ValueError(f'Invalid kld loss mode: {KLD_MODE}.')
        # Iteration
        for ite, inputs in enumerate(train_dataloader):
            if (epoch - 1) % test_epoch_freq == 0:
                encoder.eval()
                decoder.eval()
                tensors = None
                for test_inputs in test_dataloader:
                    test_inputs = test_inputs.to(device).float()
                    mu, logvar = encoder(test_inputs)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    latent = eps * std + mu
                    test_rec = decoder(latent)
                    if tensors is None:
                        tensors = torch.cat((test_inputs, test_rec))
                    else:
                        tensors = torch.cat((tensors, test_inputs, test_rec))
                grid = make_grid(tensors, padding=1, nrow=len(test_dataloader))
                save_image(grid, os.path.join(plot_path, f'epoch{epoch - 1:04}.jpg'))
            if (epoch - 1) % save_model_epoch_freq == 0:
                torch.save({'encoder': encoder.module if use_dataparallel else encoder,
                            'decoder': decoder.module if use_dataparallel else decoder,
                            'optimizer': optimizer, 'args': args, 'completed_epoch': epoch-1},
                           os.path.join(model_path, f'epoch{epoch - 1:07}.pt'))
            #
            optimizer.zero_grad()
            #
            inputs = inputs.to(device).float()
            mu, logvar = encoder(inputs)
            #
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = eps * std + mu
            #
            rec = decoder(latent)
            #
            rec_loss = nn.BCELoss(reduce='sum')(rec + 1e-10, inputs)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.shape[0]
            print('--- Epoch: {}/{}\t Iter: {}/{}\tTime: {}\t{:.4f} {:.4f} {}'.format(
                epoch, n_epoch, ite + 1, len(train_dataloader), str(datetime.now() - start_time), rec_loss.item(),
                kld_loss.item(), kld_weight))
            loss = rec_loss + kld_loss * kld_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            optimizer.step()
            #
            train_loss += loss.item()
            train_rec_loss += rec_loss.item()
            train_kld_loss += kld_loss.item()
        avg_total_loss = train_loss / len(train_dataloader)
        avg_rec_loss = train_rec_loss / len(train_dataloader)
        avg_kld_loss = train_kld_loss / len(train_dataloader)
        print(
            f'Epoch: {epoch}/{n_epoch}\tTotal loss: {avg_total_loss:.3f}\t'
            f'Rec loss: {avg_rec_loss:.3f}\tKld loss: {avg_kld_loss:.3f}')
        writer.add_scalar('Total Loss/Total Loss', avg_total_loss, epoch)
        writer.add_scalar('Loss/Reconstruction Loss', avg_rec_loss, epoch)
        writer.add_scalar('Loss/KL-divergence Loss', avg_kld_loss, epoch)
        writer.add_scalar('Loss/KL-divergence Weight', kld_weight, epoch)


if __name__ == '__main__':
    # Default values
    default_root_path = os.path.realpath(os.curdir)

    # Arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cont-train', required=False, type=int,
                        help='Provide a epoch number to continue training from it.')
    parser.add_argument('-r', '--root', default=default_root_path, type=str,
                        help='Repo path. Default: current work directory.')
    parser.add_argument('-x', '--experiment', required=True, type=str, help='Experiment name.')
    parser.add_argument('-e', '--epoch', required=True, type=int, help='Number of epochs.')
    parser.add_argument('-b', '--batch-size', '--bs', required=True, type=int, help='Batch size.')
    parser.add_argument('--lr', '--learning-rate', required=True, type=float, help='Learning rate.')
    parser.add_argument('--z-dim', '--zd', required=True, type=int, help='Number of latent dimensions.')
    parser.add_argument('--nef', required=True, type=int, help='Number of channels of first layer of encoder.')
    parser.add_argument('-d', '--dataparallel', type=bool, default=True, help='Whether use all GPUs.')
    parser.add_argument('--test-epoch-freq', '--test-freq', required=True, type=int,
                        help='Testing frequency.(unit: epoch)')
    parser.add_argument('--save-model-epoch-freq', '--save-freq', required=True, type=int,
                        help='Model saving frequency.(unit: epoch)')
    KLD_MODES = ['monotonic', 'fixed', 'cyclical']  # TODO: make 'fixed', 'cyclical'
    parser.add_argument('--kld-mode', required=True, type=str, help=f'KLD weight mode. ({", ".join(KLD_MODES)})')
    #
    ARGS = parser.parse_args()
    print(ARGS)

    print(f'Repo root path: {ARGS.root}')
    os.chdir(ARGS.root)

    # Get constants
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_DATAPARALLEL = ARGS.dataparallel
    ROOT_PATH = ARGS.root
    EXPERIMENTS_PATH = os.path.join(ROOT_PATH, 'experiments')
    EXPERIMENT_PATH = os.path.join(EXPERIMENTS_PATH, ARGS.experiment)
    LOG_PATH = os.path.join(EXPERIMENT_PATH, 'log')
    MODEL_PATH = os.path.join(EXPERIMENT_PATH, 'model')
    PLOT_PATH = os.path.join(EXPERIMENT_PATH, 'plot')
    N_EPOCH = ARGS.epoch
    BATCH_SIZE = ARGS.batch_size
    LR = ARGS.lr
    Z_DIM = ARGS.z_dim
    NEF = ARGS.nef
    TEST_EPOCH_FREQ = ARGS.test_epoch_freq
    SAVE_MODEL_EPOCH_FREQ = ARGS.save_model_epoch_freq
    KLD_MODE = ARGS.kld_mode

    # Data
    MNIST = torchvision.datasets.MNIST('data',
                                       download=True if not os.path.exists(
                                           os.path.join(ROOT_PATH, 'data')) else False).data
    TRAIN_DATALOADER = DataLoader(DigitsDataset(MNIST[:59992]), batch_size=BATCH_SIZE)  # 59992 images for training
    TEST_DATALOADER = DataLoader(DigitsDataset(MNIST[59992:]), batch_size=1)  # 8 images for testing

    #
    ENCODER = Encoder(Z_DIM, NEF)
    DECODER = Decoder(Z_DIM, NEF)
    OPTIMIZER = optim.Adam(list(ENCODER.parameters()) + list(DECODER.parameters()), lr=LR)

    # KLD modes
    KLD_OPT = {'KLD_MODE': KLD_MODE}
    if KLD_MODE == 'monotonic':
        print('=== Setting of monotonic KLD loss mode ===')
        KLD_W_FROM = int(input('Increase KLD loss weight from epoch: '))
        KLD_W_TO = int(input('Reach KLD loss weight 1.0 in epoch: '))
        KLD_OPT['KLD_W_FROM'] = KLD_W_FROM
        KLD_OPT['KLD_W_TO'] = KLD_W_TO
        KLD_OPT['MAX_KLD_W'] = float(input('Max KLD loss weight: '))
    elif KLD_MODE == 'fixed':
        KLD_OPT['FIXED_W'] = float(input('Fixed weight: '))
    elif KLD_MODE == 'cyclical':
        print('=== Setting of monotonic KLD loss mode ===')
        KLD_START = int(input('Start cycle from epoch: '))
        KLD_LEN = int(input('Each cycle length. (#epochs): '))
        KLD_OPT['KLD_START'] = KLD_START
        KLD_OPT['KLD_LEN'] = KLD_LEN
        KLD_OPT['MAX_KLD_W'] = float(input('Max KLD loss weight: '))
    else:
        raise ValueError(f'Invalid KLD mode "{KLD_MODE}" (valid: {", ".join(KLD_MODES)})')

    # Make directories and record args
    make_dir_if_not_exists(EXPERIMENTS_PATH)
    make_dir_if_not_exists(EXPERIMENT_PATH)
    make_dir_if_not_exists(LOG_PATH)
    make_dir_if_not_exists(MODEL_PATH)
    make_dir_if_not_exists(PLOT_PATH)
    with open(os.path.join(EXPERIMENT_PATH, 'args.txt'), 'w+') as args_file:  # Save arguments
        args_file.write(ARGS.__repr__())
        args_file.write('\n')
        args_file.write(KLD_OPT.__repr__())
    WRITER = SummaryWriter(log_dir=LOG_PATH, flush_secs=60)

    #
    START_EPOCH = 1
    if ARGS.cont_train is not None:
        loaded = torch.load(os.path.join(MODEL_PATH, f'epoch{ARGS.cont_train:07}.pt'))
        ENCODER = loaded['encoder'].cpu()
        DECODER = loaded['decoder'].cpu()
        OPTIMIZER = loaded['optimizer']
        START_EPOCH = loaded['completed_epoch'] + 1

    #
    train(encoder=ENCODER,
          decoder=DECODER,
          optimizer=OPTIMIZER,
          device=DEVICE,
          start_epoch=START_EPOCH,
          n_epoch=N_EPOCH,
          train_dataloader=TRAIN_DATALOADER,
          test_dataloader=TEST_DATALOADER,
          test_epoch_freq=TEST_EPOCH_FREQ,
          save_model_epoch_freq=SAVE_MODEL_EPOCH_FREQ,
          use_dataparallel=USE_DATAPARALLEL,
          writer=WRITER,
          model_path=MODEL_PATH,
          plot_path=PLOT_PATH,
          kld_opt=KLD_OPT,
          args=ARGS)

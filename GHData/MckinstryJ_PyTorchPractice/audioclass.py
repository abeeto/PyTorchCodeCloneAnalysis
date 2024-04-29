import numpy as np
import librosa
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34

from tqdm import tqdm
import os


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512,
                          n_mels=128, fmin=20, fmax=8300, top_db=80):
    """

    :param file_path:
    :param sr: sampling rate (default 44.1KHz) to load the data
    :param n_fft: number of samples per window
    :param hop_length: number of samples to skip
    :param n_mels: dim of spectro image
    :param fmin: lowest frequency
    :param fmax: highest frequency
    :param top_db: threshold for output
    :return:
    """
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels, fmin=fmin, fmax=fmax)
    return librosa.power_to_db(spec, top_db=top_db)


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    return spec_scaled.astype(np.uint8)


class ESC50Data(Dataset):
    def __init__(self, base, df, in_col, out_col):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i = {}
        self.i2c = {}
        self.categories = sorted(df[out_col].unique())
        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category

        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[in_col])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis, ...])
            self.labels.append(self.c2i[row['category']])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]


class ResNet():
    def __init__(self, lr, opt, epochs, loss):
        # Building the Model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 50)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7),
                                     stride=(2, 2), padding=(3, 3), bias=False)
        self.model = self.model.to(self.device)

        self.lr = lr
        self.opt = opt(self.model.parameters(), lr=self.lr)
        self.epochs = epochs
        self.loss = loss
        self.train_losses = []
        self.test_losses = []

    def lr_decay(self, epoch):
        if epoch % 10:
            self.lr /= 10 ** (epoch // 10)
            self.opt = self.opt.setlr(self.lr)
            print('Changed lr to: {}'.format(self.lr))

    def train(self, train_load, test_load):
        change_lr = False
        for ep in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            batch_losses = []

            if change_lr:
                self.lr_decay(ep)

            for i, data in enumerate(train_loader):
                x, y = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.long)
                self.opt.zero_grad()
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                loss.backward()
                batch_losses.append(loss.item())
                self.opt.step()

            self.train_losses.append(batch_losses)
            print('Epoch - {} <---> Train-Loss: {}'.format(ep, np.mean(self.train_losses[-1])))

            self.model.eval()
            batch_losses = []
            trace_y = []
            trace_yhat = []
            for i, data in enumerate(test_load):
                x, y = data[0].to(self.device, dtype=torch.float32), data[1].to(self.device, dtype=torch.long)
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                trace_y.append(y.cpu().detach().numpy())
                trace_yhat.append(y_hat.cpu().detch().numpy())
                batch_losses.append(loss.item())
            self.test_losses.append(batch_losses)
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            print('Epoch - {} <---> Test-Loss: {}'.format(ep, np.mean(trace_yhat.argmax(axis=1) == trace_y)))


# Getting the Data
df = pd.read_csv('ESC-50/meta/esc50.csv')
df['is_valid'] = (df['fold'] == 5)

# TODO: adjusting df to be train / test

train_data = ESC50Data('audio', train, 'filename', 'category')
test_data = ESC50Data('audio', test, 'filename', 'category')
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

res = ResNet(lr=.01, opt=optim.Adam, epochs=50, loss=nn.CrossEntropyLoss)
res.train(train_loader, test_loader)

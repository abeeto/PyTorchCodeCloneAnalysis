import os

import torch.optim
from keras.datasets import cifar10
from torch import nn
from tqdm import tqdm
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader

from models.cnn import Detector
from utils import addnoise, randshadow
from scipy.fftpack import dct, idct

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

weight_decay = 1e-4
num_classes = 2

device = torch.device('cuda')
model = Detector()
model.to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.05, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()


def patching_train(clean_sample):
    '''
    this code conducts a patching procedure with random white blocks or random noise block
    '''
    attack = np.random.randint(0, 5)
    pat_size_x = np.random.randint(2, 8)
    pat_size_y = np.random.randint(2, 8)
    output = np.copy(clean_sample)
    if attack == 0:
        block = np.ones((pat_size_x, pat_size_y, 3))
    elif attack == 1:
        block = np.random.rand(pat_size_x, pat_size_y, 3)
    elif attack == 2:
        return addnoise(output)
    elif attack == 3:
        return randshadow(output)
    if attack == 4:
        randind = np.random.randint(x_train.shape[0])
        tri = x_train[randind]
        mid = output + 0.3 * tri
        mid[mid > 1] = 1
        return mid

    margin = np.random.randint(0, 6)
    rand_loc = np.random.randint(0, 4)
    if rand_loc == 0:
        output[margin:margin + pat_size_x, margin:margin + pat_size_y, :] = block  # upper left
    elif rand_loc == 1:
        output[margin:margin + pat_size_x, 32 - margin - pat_size_y:32 - margin, :] = block
    elif rand_loc == 2:
        output[32 - margin - pat_size_x:32 - margin, margin:margin + pat_size_y, :] = block
    elif rand_loc == 3:
        output[32 - margin - pat_size_x:32 - margin, 32 - margin - pat_size_y:32 - margin, :] = block  # right bottom

    output[output > 1] = 1
    return output


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


for i in tqdm(range(5)):
    poi_train = np.zeros_like(x_train)
    for i in range(x_train.shape[0]):
        poi_train[i] = patching_train(x_train[i])

    # 3channel dct
    x_dct_train = np.vstack((x_train, poi_train))
    y_dct_train = (np.vstack((np.zeros((x_train.shape[0], 1)), np.ones((x_train.shape[0], 1))))).astype(np.int)
    for i in range(x_dct_train.shape[0]):
        for channel in range(3):
            x_dct_train[i][:, :, channel] = dct2((x_dct_train[i][:, :, channel] * 255).astype(np.uint8))

    # SHUFFLE TRAINING DATA
    x_dct_train = np.transpose(x_dct_train, (0, 3, 1, 2))
    x_dct_train, y_dct_train = torch.tensor(x_dct_train, dtype=torch.float), torch.tensor(y_dct_train,
                                                                                          dtype=torch.long).view(
        (-1,))
    dataloader = DataLoader(TensorDataset(x_dct_train, y_dct_train), batch_size=64, shuffle=True)
    model.train()
    print(f'size: {len(dataloader.dataset)}')
    for ep in range(10):
        for load in dataloader:
            d = load[0].to(device)
            t = load[1].to(device)
            optimizer.zero_grad()
            pred = model(d)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()

    os.makedirs('./detector/', exist_ok=True)
    torch.save(model.state_dict(), './detector/6_CNN_CIFAR10.pth')

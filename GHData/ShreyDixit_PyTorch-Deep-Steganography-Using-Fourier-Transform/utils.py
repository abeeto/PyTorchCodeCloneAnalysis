import numpy as np
import os
import torch
import cv2
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import torchvision.transforms as transforms

def encrypt(img_channel, seed = 42):
    np.random.seed(seed)
    f = np.fft.fft2(img_channel)
    mag, ang = np.abs(f), np.arctan2(f.imag, f.real)
    ns = np.random.uniform(0, 6.28, size = f.shape)
    ns = np.fft.fft2(ns)
    ns = np.arctan2(ns.imag, ns.real)
    ang_new = ang+ns
    noise_img = np.fft.ifft2(mag*np.exp((ang_new)*1j)).real
    return noise_img

def decrypt(img_channel, seed = 42):
    np.random.seed(seed)
    f = np.fft.fft2(img_channel)
    mag, ang = np.abs(f), np.arctan2(f.imag, f.real)
    ns = np.random.uniform(0, 6.28, size = f.shape)
    ns = np.fft.fft2(ns)
    ns = np.arctan2(ns.imag, ns.real)
    ang_new = ang-ns
    noise_img = np.fft.ifft2(mag*np.exp((ang_new)*1j)).real
    return noise_img

def image_crypto(f_path, size, func):
    img = cv2.imread(f_path)[:, :, [2, 1, 0]]
    img = cv2.resize(img, size)
    return np.stack([func(img[..., i]) for i in range(3)], 2)

def encode(f_path, idx, model, size, fourier_func):
    path, fname = '/'.join(f_path.split('/')[:-2]), f_path.split('/')[-1]
    x = cv2.resize(cv2.imread(f_path)[:, :, [2, 1, 0]], size).transpose((2,0,1))
    y = image_crypto(path + '/secret/'+ fname, size, fourier_func).transpose((2,0,1))
    x, y = torch.tensor(x), torch.tensor(y)
    X = torch.cat((x.float(), y.float()), 0)
    out = model(X[None]).squeeze().permute(1, 2, 0)[:, :, [2, 1, 0]].detach().numpy()
    cv2.imwrite(path + '/encoded/' + fname, out)
                                                                        
def decode(f_path, idx, model, size, fourier_func):
    path, fname = f_path.split('/')[:-2].join('/'), f_path.split('/')[-1]
    x = cv2.resize(cv2.imread(f_path)[:, :, [2, 1, 0]], size).transpose((2,0,1))
    x = torch.tensor(x)
    out = model(x[None]).squeeze().permute(1, 2, 0)[:, :, [2, 1, 0]].detach().numpy()
    out = np.stack([fourier_func(out[..., i]) for i in range(3)], 2)                             
    cv2.imwrite(path + '/decoded/' + fname, out)                                                                        
                                                                            
def mse(y_pred, y): 
    return ((y_pred - y)**2).mean()
def mse_cov(y_pred, y): 
    return ((y_pred[:, 0] - y[:, 0])**2).mean()
def mse_hidden(y_pred, y): 
    return ((y_pred[:, 1] - y[:, 1])**2).mean()                                                                    

class Image_Data(Dataset):
    def __init__(self, path, data_len, fourier_seed, size):
        self.path = path
        self.encrypt_partial = partial(encrypt, seed=fourier_seed)
        self.size = size
        f_paths = [path + '/'+ f for f in os.listdir(path)]
        self.file_pairs = list(zip(random.choices(f_paths, k=data_len), random.choices(f_paths, k=data_len)))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        file_pair= self.file_pairs[idx]
        x = cv2.imread(file_pair[0])[:, :, [2, 1, 0]]
        x = cv2.resize(x, self.size).transpose((2,0,1))
        y = image_crypto(file_pair[1], self.size, self.encrypt_partial).transpose((2,0,1))
        x, y = torch.tensor(x), torch.tensor(y)
        X = torch.stack((x.float(), y.float()))
        return X, X
    
def ImageLoader(path, data_len, fourierSeed = 42, size=300, bs=64):
    return DataLoader(Image_Data(path, data_len, fourierSeed, size), bs)
                                                                            
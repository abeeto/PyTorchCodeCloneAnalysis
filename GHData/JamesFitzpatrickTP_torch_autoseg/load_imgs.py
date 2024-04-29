import numpy as np
import torch
import os
from PIL import Image

train_dir = '/mnt/c/Users/James Fitzpatrick/Downloads/train/train/'
truth_dir = '/mnt/c/Users/James Fitzpatrick/Downloads/train_masks/train_masks/'


def img_paths(dir_path):
    fnames = os.listdir(dir_path)
    paths = [os.path.join(dir_path, fname) for fname in fnames]
    return paths


def img_loader(path):
    return np.array(Image.open(path))


def data_loader(dir_path, idx_list=None):
    if idx_list is None:
        idx_list = list(range(os.listdir(dir_path)))
    paths = np.array(img_paths(dir_path))[idx_list]
    imgs = [img_loader(path) for path in paths]
    return np.array(imgs)


def no_idxs(idx_len):
    idx_list = list(range(5000))
    np.random.shuffle(idx_list)
    return idx_list[:idx_len]


def set_loader(train_dir, truth_dir, idx_list=None):
    if idx_list is None:
        idx_list = no_idxs(100)
    print("Loading ", len(idx_list), "train and truth image(s)...")
    train_imgs = data_loader(train_dir, idx_list)
    truth_imgs = data_loader(truth_dir, idx_list)
    return train_imgs, truth_imgs
    

def roll_imgs(img_arr):
    return np.rollaxis(img_arr, -1, 1)


def torch_loader(train_dir, truth_dir, idx_list):
    train_imgs, truth_imgs = set_loader(train_dir, truth_dir, idx_list)
    train_imgs = torch.tensor(roll_imgs(train_imgs)).float()
    truth_imgs = torch.tensor(np.expand_dims(truth_imgs, 1)).float()
    return train_imgs, truth_imgs

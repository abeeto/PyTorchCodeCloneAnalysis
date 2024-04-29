from scipy.sparse.construct import random
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import collections
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import torch
import random

torch.manual_seed(0)

def show(p_img, n_img, name, norm, m, s):
	# convert tensor to numpy array
    p_npimg = p_img.numpy()
    n_npimg = n_img.numpy()
    if norm:
        p_npimg = m + s * p_npimg
        n_npimg = m + s * n_npimg
	# Convert to H*W*C shape
    p_npimg_tr=np.transpose(p_npimg, (1,2,0))
    n_npimg_tr=np.transpose(n_npimg, (1,2,0))
    plt.subplot(1,2,1)
    plt.imshow(p_npimg_tr)
    plt.title("COVID-19")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(n_npimg_tr)
    plt.title("Non COVID-19")
    plt.axis('off')
    plt.savefig(name)
    plt.clf()

def make_random_sample(data_set, name, norm, m, s):
    sample_size = 4
    pos_len = 2
    neg_len = 2
    neg_sample = []
    pos_sample = []
    neg = []
    pos = []
    idx_list = list(range(len(data_set)))
    random.shuffle(idx_list)

    for i in idx_list:
        _, idx = data_set[i]
        if idx == 0:
            neg_len -= 1
            neg_sample.append(i)
        if neg_len == 0:
            break

    for i in neg_sample:
        neg.append(data_set[i][0])

    for i in idx_list:
        _, idx = data_set[i]
        if idx == 1:
            pos_len -= 1
            pos_sample.append(i)
        if pos_len == 0:
            break

    for i in pos_sample:
        pos.append(data_set[i][0])


    pos_sample = make_grid(pos, nrow=2, padding=3)
    neg_sample = make_grid(neg, nrow=2, padding=3)
    show(pos_sample, neg_sample, name, True, m, s)

def label_statistics(data_set):
    if isinstance(data_set, Subset):
        labels = np.array(data_set.dataset.targets)[data_set.indices]
    else:
        labels = data_set.targets
    counter_stat = collections.Counter(labels)
    return counter_stat

def mean(data_set):
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in data_set]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    return [meanR, meanG, meanB]

def std(data_set):
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in data_set]

    stdR = np.mean([m[0] for m in stdRGB])
    stdG = np.mean([m[1] for m in stdRGB])
    stdB = np.mean([m[2] for m in stdRGB])

    return [stdR, stdG, stdB]

def get_data_loader(path_to_data, batch_size, result, max_data):
    # Transformers
    temp_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor()
    ])

    # Data set
    covid_ds = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    print("Total dataset class :", label_statistics(covid_ds))

    # index of list
    indices = list(range(len(covid_ds)))
    max_len = min(max_data, len(covid_ds))

    train_size = int(max_len*0.7)
    val_size = int(max_len * 0.2)
    test_size = int(max_len * 0.1)
    etc = len(covid_ds) - train_size - val_size - test_size

    torch.manual_seed(0)
    train_ds, val_ds, test_ds, _ = random_split(covid_ds, [train_size, val_size, test_size, etc])
    print("Train dataset class :", label_statistics(train_ds))
    print("val dataset class :", label_statistics(val_ds))
    print("test dataset class :", label_statistics(test_ds))

    # Sample images
    sample_size = 4

    train_sample = [train_ds[i][0] for i in range(sample_size)]
    val_sample = [val_ds[i][0] for i in range(sample_size)]
    test_sample = [test_ds[i][0] for i in range(sample_size)]

    train_sample = make_grid(train_sample, nrow=8, padding=1)
    val_sample = make_grid(val_sample, nrow=8, padding=1)
    test_sample = make_grid(test_sample, nrow=8, padding=1)

    mean_val = mean(train_ds)
    std_val = std(train_ds)

    norm = transforms.Normalize(mean=mean(train_ds), std=std(train_ds))
    make_random_sample(train_ds, os.path.join(result, "train_sample.png"), True, mean_val[0], std_val[0])
    make_random_sample(val_ds, os.path.join(result, "val_sample.png"), True, mean_val[0], std_val[0])
    make_random_sample(test_ds, os.path.join(result, "test_sample.png"), True, mean_val[0], std_val[0])
 
    # Transformers
    norm = transforms.Normalize(mean=mean_val, std=std_val)
    global_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        norm
    ])

    # Change transformer
    covid_ds.transform = global_transformer
 
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
 
    return train_dl, val_dl, test_dl


import torch as tc
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def loss_acc_plot(loss_arr, acc_tr, acc_v=None):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8,4))
    ax1.set_title("Accuracy")
    ax1.plot(acc_tr, c='b', label='Train set')
    if acc_v is not None:
        ax1.plot([len(acc_tr)/len(acc_v)*i for i in range(len(acc_v))],
                  acc_v, c='r', label='Validation set')
    ax1.legend()
    ax2.set_title("Loss")
    ax2.plot(loss_arr, c='g')
    fig.tight_layout()
    plt.show()

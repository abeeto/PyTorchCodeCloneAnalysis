from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time

text = "The action scenes were top notch in this movie. Thor has never been this epic in the MCU. He does some pretty epic sh*t in this movie and he is definitely not under-powered anymore. Thor in unleashed in this, I love that. Converting text into characters"


class Dictionary(object):
    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.length = 0

    def add_word(self,word):
        if word not in self.id2word:
            self.id2word.append(word)
            self.word2id[word] = self.length+1
            self.length += 1

    def __len__(self):
        return len(self.id2word)

    def onehot_encode(self,word):
        vec = np.zeros(self.length)
        vec[self.word2id[word]] = 1
        return vec


dict = Dictionary()

for tok in text.split():
    dict.add_word(tok)

print(dict.word2id)
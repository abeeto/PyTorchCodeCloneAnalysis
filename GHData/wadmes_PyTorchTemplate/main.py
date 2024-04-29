import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import  WeightedRandomSampler
import time
from model import *
from data import *
from utils import *
import os
import time
from sklearn.metrics import confusion_matrix
def train():


def test():

if __name__ == "__main__":
    print("Arguments: ", opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    model = TemplateModel() 
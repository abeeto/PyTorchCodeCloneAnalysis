import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from config import *


class TemplateModel(nn.Module):
    def __init__(self):
        super(TemplateModel, self).__init__()

    def forward(self, x):
        # print(x.size())
        return 
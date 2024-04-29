import torch as t
import numpy as np
from models.VGG16 import VGG16
from models.RCNN import RCNN
from models.RPN import RPN
from train_helper import TrainHelper
from Faster_RCNN import Faster_RCNN

# V = VGG16(pretrained=True)
# RC = RCNN(21, 0, 1)
# RP = RPN('VGG16')
FR = Faster_RCNN()
TH = TrainHelper(FR)

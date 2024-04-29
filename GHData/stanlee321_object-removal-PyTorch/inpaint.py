import argparse
import torch
import cv2
import numpy as np
import torchvision.utils as vutils
from torch.utils.serialization import load_lua
from torchsummary import summary

import os, sys
from glcic.glcic import glcic

# load Completion Network
#model = glcic(in_ch=3, out_ch=3, ch=64)
data = load_lua('./glcic/completionnet_places2.t7')
model = data.model
model.evaluate()

print(model)
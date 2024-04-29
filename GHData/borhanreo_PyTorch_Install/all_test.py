import torch
from __future__ import print_function
import cv2
import tensorflow
print("Torch version: ",torch.__version__)
print("OpenCV version",cv2.__version__)
print("TensorFlow version",tensorflow.__version__)
print(torch.rand(5,4))
torch.cuda.is_available()

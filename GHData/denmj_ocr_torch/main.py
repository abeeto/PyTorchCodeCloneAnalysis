
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

print(torch.__version__)

if torch.cuda.is_available():
    print("GPU is available: {}, {}".format(torch.cuda.device_count(), torch.cuda.get_device_name()))
else:
    print("GPU is not available")

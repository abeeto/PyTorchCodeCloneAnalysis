# Tut 7

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available() == True:
	print("yes")
	print(torch.cuda.device_count())

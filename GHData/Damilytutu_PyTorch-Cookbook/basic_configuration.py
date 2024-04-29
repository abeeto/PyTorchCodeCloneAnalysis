import collections
import os
import shutil
import tqdm

import numpy as np
import PIL.Image
import torch
import torchvision


# check pytorch version
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))


# update pytorch
#conda update pytorch torchvision -c pytorch


# fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# assign gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


# determine if there is a cuda support
torch.cuda.is_available()


# cudnn benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# eliminate gpu memory
torch.cuda.empty_cache()

import torch
from torch.utils.data import DataLoader
import os
from os.path import join as pjoin
import time
import numpy as np
from src import transforms
from src.datasets import IRMAS

project_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

fn = pjoin(project_path, 'data/IRMAS/IRMAS-TrainingData/cel/[cel][cla]0001__1.wav')

fns = [fn] * 100

#trans_rosa = transforms.MelSpecTransformRosa()
#pool_rosa = transforms.TransformPool(trans_rosa)
#
#tic = time.time()
#pool_rosa(fns)
#print(f'librosa time {time.time() - tic}')


trans_torch = transforms.MelSpecTransformTorchAudio()
#pool_torch = transforms.TransformPool(trans_rosa)

tic = time.time()
for fn in fns:
    trans_torch(fn)
print(f'Torch time {time.time() - tic}')


ds = IRMAS(mono=False, time_slice=1)
loader = DataLoader(ds, batch_size=32, num_workers=16)
#times = []
tic = time.time()
for i, batch in enumerate(loader):
    if i == 100:
        break
    
print((time.time() - tic)/i)
#print(f'Loader time {time.time() - tic}')
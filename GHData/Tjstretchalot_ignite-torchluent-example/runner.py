"""Runs mnist"""
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTOBLAS_MAIN_FREE'] = '1'

import torch
torch.set_num_threads(3)

import mnist

if __name__ == '__main__':
    mnist.train()

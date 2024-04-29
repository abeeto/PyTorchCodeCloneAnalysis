import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data


from dataset import Dataset

root = '/data/yanjianhao/nlp/torch/torch_NRE/data/'
datasets = Dataset(root)


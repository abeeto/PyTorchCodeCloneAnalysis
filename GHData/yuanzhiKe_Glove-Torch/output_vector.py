from collections import Counter, defaultdict
import matplotlib.pyplot as pltb
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pickle
from struct import unpack
import argparse
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
from glove_torch import GloveModel
import pandas as pd


def main(args):
    words = {}
    with open(args.vocab_file) as f:
        for i, line in enumerate(f):
            word = line.split(' ')[0]
            words[i] = word
    vocab_size = len(words)
    words[vocab_size] = '<unk>'
    print(vocab_size)
    EMBED_DIM = args.emb_size
    if args.epoch is None:
        saved_state = torch.load(args.model_name)
        print(f'load parameters from {args.model_name}.')
    else:
        checkpoint = torch.load(args.model_name + '.e' + str(args.epoch))
        assert checkpoint['epoch'] == args.epoch
        saved_state = checkpoint['model_state_dict']
        print(f'load checkpoint from {args.model_name}.e{args.epoch}. checkpoint epoch: {checkpoint["epoch"]}, loss: {checkpoint["loss"]}')
    glove = GloveModel(vocab_size + 1, EMBED_DIM)
    glove.load_state_dict(saved_state)
    #glove.cuda()
    emb_i = glove.wi.weight.data.numpy()
    emb_j = glove.wj.weight.data.numpy()
    emb = emb_i + emb_j
    with open(args.output_file, 'w') as f:
         for i, i_emb in enumerate(emb):
             line = words[i] + ' '
             values = ' '.join([str(v) for v in np.round(i_emb, decimals=6)])
             line += values
             line += '\n'
             f.write(line)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_file', type=str, help='path to the vocab file', required=True)
    parser.add_argument('-e', '--epoch', type=int, default=None)
    parser.add_argument('-m', '--model_name', type=str, help='saving model path', required=True)
    parser.add_argument('-o', '--output_file', type=str, help='file to output', required=True)
    parser.add_argument('--emb_size', type=int, help='embedding size', default=200)
    args = parser.parse_args()
    main(args)

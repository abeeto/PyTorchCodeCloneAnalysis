#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""
Train POS models
"""

import sys
import time
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from dataset import build_vocab
from dataset import get_iterators
from config import VocabConfig, ModelConfig, TrainConfig

from lstmcrf import LSTMCRF

parser = argparse.ArgumentParser(sys.argv[0], description='build vocab by passing pos train file. (each line contains a char and a processed tag)')
parser.add_argument('input_file', help='processed input file, each line has two columns, one is char and another is processed tag')

args = parser.parse_args()

train_iterator, valid_iterator, test_iterator = get_iterators(args.input_file)

build_vocab(train_iterator.dataset)

src_vocab = train_iterator.dataset.fields['TEXT'].vocab
#src_vocab.load_vectors()
#src_embedding = src_vector
tgt_vocab = train_iterator.dataset.fields['TAG'].vocab

src_size = len(src_vocab.stoi)
tgt_size = len(tgt_vocab.stoi)

print('src_size: ', src_size)
print('tgt_size: ', tgt_size)

tgt_pad_idx = tgt_vocab.stoi[VocabConfig.PAD]
#print('src pad: ', src_vocab.stoi[VocabConfig.PAD])
print('tgt pad: ', tgt_vocab.stoi[VocabConfig.PAD])

def build_model():
    if TrainConfig.model == 'lstm':
        model = LSTMCRF(src_size, tgt_size,
                        tgt_vocab.stoi[VocabConfig.PAD], ModelConfig)
    elif TrainConfig.model == 'bert':
        pass
    return model

model = build_model().to(TrainConfig.device)
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    TrainConfig.lr,
    betas=(0.9, 0.98),
    eps=1e-09
)


def epochs():
    for epoch in range(1, TrainConfig.epochs + 1):
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_loss = train_epoch(epoch)

        print(' (Training)   loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'.format(
                    loss=train_loss,
                    elapse=(time.time()-start)/60))

        start = time.time()
        valid_accu, valid_loss = valid_epoch(epoch)
        print(' (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss,
                    accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        start = time.time()
        test_accu = test(epoch)
        print(' (Test) accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(
                    accu=100*test_accu,
                    elapse=(time.time()-start)/60))

def train_epoch(epoch):
    model.train()

    total_loss = 0

    for i, batch in tqdm(enumerate(train_iterator), mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        optimizer.zero_grad()

        (inputs, inputs_length), tgts = batch
        #print('inputs: ', inputs.shape)
        #print(inputs)
        #print('inputs_length: ', inputs_length.shape)
        #print('tgts: ', tgts.shape)
        inputs_mask = torch.arange(0, inputs.shape[0]).long() \
							.repeat(TrainConfig.batch_size, 1) \
							.lt(inputs_length.unsqueeze(1)) \
                            .T \
							.to(TrainConfig.device)
        #print('inputs_mask: ', inputs_mask.shape)
        #print(inputs_mask)

        loss = -model(inputs, inputs_length, inputs_mask, tgts)
        #print('loss: ', loss)

        #if i % 50 == 0:
        #    print('loss: ', loss.item())

        total_loss += loss.item()

        loss.backward()

         # update parameters
        optimizer.step()

    return total_loss / len(train_iterator)

def valid_epoch(epoch):
    model.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_iterator), mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):

            (inputs, inputs_length), tgts = batch
            #print('inputs: ', inputs.shape)
            #print('tgts: ', tgts.shape)
            inputs_mask = torch.arange(0, inputs.shape[0]).long() \
                                .repeat(TrainConfig.batch_size, 1) \
                                .lt(inputs_length.unsqueeze(1)) \
                                .T \
                                .to(TrainConfig.device)

            loss = -model(inputs, inputs_length, inputs_mask, tgts)
            #print('loss: ', loss)
            total_loss += loss.item()

            # List[List[int]], [batch_size, seq_len]
            pred_tgts = model.decode(inputs, inputs_length, inputs_mask)

            accu = call_performace(tgts.T.tolist(), pred_tgts, inputs_length.tolist())
            total_acc += accu

    return total_acc / len(valid_iterator), total_loss / len(valid_iterator)


def test(epoch):
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_iterator), mininterval=2,
                desc=' (Teste: %d) ' % epoch, leave=False):

            (inputs, inputs_length), tgts = batch
            inputs_mask = torch.arange(0, inputs.shape[0]).long() \
                                .repeat(TrainConfig.batch_size, 1) \
                                .lt(inputs_length.unsqueeze(1)) \
                                .T \
                                .to(TrainConfig.device)

            # List[List[int]], [batch_size, seq_len]
            pred_tgts = model.decode(inputs, inputs_length, inputs_mask)
            #pred_tgts = torch.LongTensor(pred_tgts).to(TrainConfig.device)

            accu = call_performace(tgts.T.tolist(), pred_tgts, inputs_length.tolist())
            total_acc += accu

    return total_acc / len(test_iterator)

def call_performace(gold, pred, lens):
    '''
    gold: List[List[int]]
    pred: List[List[int]], removed pad
    '''
    tmp_gold, tmp_pred = list(), list()

    for i, l in enumerate(lens):
        assert(len(pred[i]) == l)

        tmp_gold.extend(gold[i][:l])
        tmp_pred.extend(pred[i])

    accu = accuracy_score(tmp_gold, tmp_pred)

    return accu

if __name__ == '__main__':
    epochs()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
Train and test RNN/LSTM/GRU/Transformer Language Model
"""

from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

from utils.data_utils import load_data
from models import RNNLM, TransformerLM
from configs import *
from log import init_logger
from vocab import Vocab
from dataset import TextDataset

# init logger
logger = init_logger()

# set random seed for reproduce
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def generate_sent(model, max_len):
    """
    Generate a sentence
    """
    sent = []
    if isinstance(model, RNNLM):
        model.hidden = model.init_hidden(1)
    init = torch.randint(len(vocab), (1, 1), dtype=torch.long).to(device)
    with torch.no_grad():  # no tracking history
        for i in range(max_len):
            if isinstance(model, TransformerLM):
                output = model(init, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                init = torch.cat([init, word_tensor], 0)
            else:
                output = model(init)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                init.fill_(word_idx)

            if word_idx == vocab['</s>']:
                break

            sent.append(vocab.itos[word_idx])

    print(' '.join(sent))


def train(model, epoch, dataset, train_loader, optimizer, criterion):
    """
    Train model
    Args:
        epoch: train epoch index
        dataset: input dataset
        train_loader: input data loader

    Returns:
        train loss
        train accuracy
    """
    t = time.time()
    model.train()
    tot_train_node = 0
    trained_batch = 0
    tot_loss = 0.
    tot_batches = int(len(dataset) / args.batch)

    for sents in train_loader:
        trained_batch += 1
        tot_train_node += len(sents)
        optimizer.zero_grad()
        output = model(sents)
        if isinstance(model, RNNLM):
            # detach RNN caches
            model.detach()
        targets = sents.view(-1).to(device)
        loss = criterion(output.view(-1, len(vocab)), targets)
        loss.backward()
        tot_loss += loss.item()

        # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-args.lr, p.grad.data)

        optimizer.step()

        if trained_batch % 100 == 0:
            logger.info("{} batches trained, eta {}s".format(
                trained_batch, tot_batches * (time.time() - t) / trained_batch))

    logger.info('| Epoch: {:4d} | loss_train: {:.4f} | ppl_train: {:.4f} | time: {:.4f}s |'.format(
            epoch + 1, tot_loss, np.exp(tot_loss / len(train_loader)), time.time() - t))

    return tot_loss


def evaluate(model, epoch, val_loader, criterion):
    """
    Evaluate validation set performance separately
    deactivates dropout during validation run
    Args:
        epoch: evaluation epoch index
        dataset: input dataset
        val_loader: input data loader

    Returns:
        evaluation loss
        evaluation accuracy
    """
    model.eval()
    tot_val_node = 0
    tot_loss = 0.
    with torch.no_grad():
        for sents in val_loader:
            tot_val_node += len(sents)
            output = model(sents)
            if isinstance(model, RNNLM):
                # detach RNN caches
                model.detach()
            targets = sents.view(-1).to(device)
            loss = criterion(output.view(-1, len(vocab)), targets)
            tot_loss += loss.item()

    logger.info('| Epoch: {:4d} | loss_val : {:.4f} | ppl_val  : {:.4f} |'.format(
        epoch + 1, tot_loss, np.exp2(tot_loss / len(val_loader))))

    return tot_loss


def test(model, test_loader, criterion):
    """
    Test model
    Args:
        model: model to test
        dataset: input dataset
        test_loader: input data loader
    """
    model.eval()
    tot_loss = 0.
    with torch.no_grad():
        for sents in test_loader:
            output = model(sents)
            targets = sents.view(-1).to(device)
            loss = criterion(output.view(-1, len(vocab)), targets)
            tot_loss += loss.item()

    logger.info("Test set results:")
    logger.info("loss = {:.4f}".format(tot_loss))
    logger.info("ppl = {:.4f}".format(np.exp2(tot_loss / len(test_loader))))


if __name__ == '__main__':
    # print configurations
    logger.info(args)

    # load data from file
    train_data = load_data('../dataset/train.csv')
    test_data = load_data('../dataset/test.csv')
    val_data = load_data('../dataset/val.csv')

    # build vocab
    if args.embed is not None:
        logger.info("Loading pre-trained embeddings...")
        vocab = Vocab(vectors=args.embed)
    else:
        vocab = Vocab(words=train_data)
    logger.info("Vocabulary length: {}".format(len(vocab)))

    # build dataset and data loaders
    train_set = TextDataset(vocab=vocab, examples=train_data,
                            padding=args.pad, sort=False)
    test_set = TextDataset(vocab=vocab, examples=test_data,
                            padding=args.pad, sort=False)
    val_set = TextDataset(vocab=vocab, examples=val_data,
                            padding=args.pad, sort=False)

    train_loader = tud.DataLoader(train_set,
                                  batch_size=args.batch,
                                  shuffle=True,
                                  collate_fn=train_set.collate,
                                  drop_last=True)
    test_loader = tud.DataLoader(test_set,
                                 batch_size=args.batch,
                                 shuffle=False,
                                 collate_fn=test_set.collate,
                                 drop_last=True)
    val_loader = tud.DataLoader(val_set,
                                batch_size=args.batch,
                                shuffle=False,
                                collate_fn=val_set.collate,
                                drop_last=True)

    # build model
    if args.model.lower() in ['lstm', 'gru', 'rnn_relu', 'rnn_tanh']:
        model = RNNLM(rnn_type=args.model.upper(), ntoken=len(vocab), ninp=args.embed_size, nhid=args.hidden,
                      nlayers=args.layers, dropout=args.dropout, bsz=args.batch, tie_weights=args.tie_weights)
    elif args.model.lower() == 'transformer':
        model = TransformerLM(ntoken=len(vocab), ninp=args.embed_size, nhead=args.head,
                              nhid=args.hidden, nlayers=args.layers, dropout=args.dropout)
    else:
        raise NotImplemented

    # init optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # transfer data to device
    if use_cuda:
        model = model.to(device)

    # train model
    t_total = time.time()
    last_dev = 1e20
    best_dev = 1e20
    best_model = None
    for epoch in range(args.epochs):
        loss_train = train(model, epoch, train_set, train_loader, optimizer, criterion)
        loss_val = evaluate(model, epoch, val_loader, criterion)

        # keep track of the development accuracy and reduce the learning rate if it got worse
        if last_dev < loss_val and hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate /= 2
        last_dev = loss_val

        # keep track of the best development loss, and save the model only if it's the best one
        if best_dev > loss_val:
            if not os.path.exists(args.models_dir):
                try:
                    os.makedirs(args.models_dir)
                except Exception as e:
                    print("Can not create models directory, %s" % e)
            torch.save(model, "{}/best.pt".format(args.models_dir))
            best_dev = loss_val
            best_model = model

    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # testing
    test(best_model, test_loader, criterion)
    for i in range(10):
        generate_sent(best_model, 20)

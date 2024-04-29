#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""
Dataset
"""
import os
import sys
import argparse
from torchtext.data import interleave_keys
from torchtext.data import Field
from torchtext.data import Example
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import BucketIterator

from config import VocabConfig
from config import TrainConfig
#from torch.utils.data import DataLoader


# TODO
def load_data2(input_file):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        data = ['', '']
        for line in f:
            line = line.strip()
            if not line:
                datas.append(data)
                data = ['', '']
                continue
            else:
                # char, processed tag
                c, t = line.split()
                data[0] += c + ' '
                data[1] += t + ' '

    return datas

def load_data(input_file):
    datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            else:
                data = tuple(line.split('\t'))
                datas.append(data)
    return datas

def build_fields():
    tokenize = lambda x: x.split()
    TEXT = Field(
        sequential=True,
        use_vocab=True,
        init_token=VocabConfig.SOS,
        eos_token=VocabConfig.EOS,
        include_lengths=True,
        tokenize=tokenize,
        pad_token=VocabConfig.PAD,
        unk_token=VocabConfig.UNK,
        batch_first=False,
    )
    TAG = Field(
        sequential=True,
        use_vocab=True,
        init_token=VocabConfig.SOS,
        eos_token=VocabConfig.EOS,
        tokenize=tokenize,
        pad_token=VocabConfig.PAD,
        unk_token=VocabConfig.UNK,
        is_target=True,
        batch_first=False,
    )

    fields = [
        ('TEXT', TEXT),
        ('TAG', TAG)
    ]

    return fields

def build_examples(datas, fields):
    examples = []
    for data in datas:
        example = Example.fromlist(data, fields)
        examples.append(example)

    return examples


def build_dataset(examples, fields, split_ratio):
    dataset = Dataset(examples, fields)
    #dataset.sort_key = lambda x: len(x['TEXT'])
    train_dataset, valid_dataset, test_dataset = dataset.split(split_ratio)

    return train_dataset, valid_dataset, test_dataset

def build_iterator(train_dataset, valid_dataset, test_dataset, batch_size, device):

    """

    train_iterator = Iterator(
        train_dataset,
        batch_size=batch_size,
        #sort_key=lambda x: len(x.TEXT),
        sort_key=lambda x: len(x[0][0]),
        train=True,
        device=device,
    )

    valid_iterator = Iterator(
        valid_dataset,
        batch_size=batch_size,
        sort_key=lambda x: len(x.TEXT),
        train=False,
        device=device
    )

    test_iterator = Iterator(
        test_dataset,
        batch_size=batch_size,
        sort_key=lambda x: len(x.TEXT),
        train=False,
        device=device
    )
    """
    # return train_iterator, valid_iterator, test_iterator
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_dataset, valid_dataset, test_dataset),
            batch_sizes=(batch_size, batch_size, batch_size),
            sort_within_batch=True,
            sort_key=lambda x: (len(x.TEXT)),
            repeat=False,
            device=device)
    return train_iterator, valid_iterator, test_iterator



def build_vocab(train_dataset):
    train_dataset.fields['TEXT'].build_vocab(train_dataset,
                                             max_size=VocabConfig.max_size,
                                             min_freq=VocabConfig.min_freq)

    train_dataset.fields['TAG'].build_vocab(train_dataset,
                                             max_size=VocabConfig.max_size,
                                             min_freq=VocabConfig.min_freq)
def get_iterators(input_file):
    datas = load_data(input_file)
    print('datas: ', len(datas))
    fields = build_fields()
    examples = build_examples(datas, fields)
    train_dataset, valid_dataset, test_dataset = build_dataset(examples,
                                                               fields,
                                                               TrainConfig.split_ratio)
    train_iterator, valid_iterator, test_iterator = build_iterator(train_dataset,
                                                                   valid_dataset,
                                                                   test_dataset,
                                                                   batch_size=TrainConfig.batch_size,
                                                                   device=TrainConfig.device)
    return train_iterator, valid_iterator, test_iterator



if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], description='build vocab by passing pos train file. (each line contains a char and a processed tag)')
    parser.add_argument('input_file', help='processed input file, each line has two columns, one is char and another is processed tag')

    args = parser.parse_args()

    train_iterator, valid_iterator, test_iterator = get_iterators(args.input_file)

    print('train_iterator: ', len(train_iterator))


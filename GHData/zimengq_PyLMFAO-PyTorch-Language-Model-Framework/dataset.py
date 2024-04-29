#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

import torch

from torch.utils import data
from configs import device


class TextDataset(data.Dataset):
    def __init__(self, vocab, examples=None, padding=True, sort=False, sort_key=None):
        super(TextDataset, self).__init__()
        self.examples = examples if examples is not None else []
        self.vocab = vocab
        self.padding = padding

        if sort and sort_key is not None:
            self.examples.sort(key=sort_key)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        assert item < len(self), 'index out of range'
        return self.examples[item]

    def get_subset(self, start, end):
        assert start < end, 'start index should be less than end index'
        return self.examples[start:end]

    def add(self, example):
        self.examples.append(example)

    def collate(self, batch):
        max_len = max([len(text) for text in batch])
        # [batch_size, max_len]
        text_batch = [torch.LongTensor([self.vocab[tok] for tok in text]) for text in batch]
        text_batch = torch.stack(
            [torch.cat((text, torch.full(((max_len - len(text)), ), self.vocab['<pad>'], dtype=torch.int64)))
             if self.padding and len(text) < max_len else text for text in text_batch])

        return text_batch.to(device)




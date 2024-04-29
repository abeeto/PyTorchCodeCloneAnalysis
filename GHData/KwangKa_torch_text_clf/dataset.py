# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 10:26
# @Author  : kaka

import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import build_vocab
from utils.tokenizer import Tokenizer


class ClassificationDataset(Dataset):
    def __init__(self,
                 fname,
                 tokenizer,
                 vocab):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.samples = []
        self._init_dataset(fname)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _init_dataset(self, fname):
        with open(fname, 'r', encoding='utf8') as h:
            for line in h:
                data = json.loads(line)
                text = data.get('text', '')
                label = data.get('category_id', 0)
                tokens = self.tokenizer(text)
                ids = self.vocab.tokens_to_ids(tokens)
                self.samples.append((ids, label))


class ClfPadCollator(object):
    def __init__(self,
                 max_len=32):
        self.max_len = max_len

    def collate(self, batch):
        ids = [torch.tensor(sample[0][:self.max_len], dtype=torch.int64) for sample in batch]
        ids = pad_sequence(ids, batch_first=True)
        labels = [sample[1] for sample in batch]
        labels = torch.tensor(labels, dtype=torch.int64)
        return ids, labels


def get_data(args):
    path = Path(args.data_path)
    f_train = path / 'train.json'
    f_test = path / 'test.json'
    f_val = path / 'val.json'

    tokenizer = Tokenizer()
    vocab = build_vocab([f_train, f_test, f_val], tokenizer=tokenizer.tokenize, min_freq=2, max_size=50000)

    train_ds = ClassificationDataset(fname=f_train, tokenizer=tokenizer.tokenize, vocab=vocab)
    test_ds = ClassificationDataset(fname=f_test, tokenizer=tokenizer.tokenize, vocab=vocab)
    val_ds = ClassificationDataset(fname=f_val, tokenizer=tokenizer.tokenize, vocab=vocab)

    collator = ClfPadCollator(args.max_seq_length)
    train_iter = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator.collate)
    test_iter = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    val_iter = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator.collate)
    return train_iter, val_iter, test_iter, vocab

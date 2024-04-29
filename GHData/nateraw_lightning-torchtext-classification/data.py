import argparse
import logging
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification
import pytorch_lightning as pl


class TorchTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "./",
        dataset_name: str = "AG_NEWS",
        ngrams: int = 2,
        train_val_split: float = 0.05,
        batch_size: int = 32,
        num_workers = 8,
        n_train = None,
        n_val = None,
        n_test = None,
    ):
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.ngrams = ngrams
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_train = None
        self.n_val = None
        self.n_test = None

    @property
    def splits(self):
        """dict where keys are split name and values are length of assiciated dataset"""
        s = dict()
        for name, ds in zip(('train', 'validation', 'test'), (self.train_ds, self.val_ds, self.test_ds)):
            s[name] = 0 if ds is None else len(ds)
        return s

    def setup(self, stage=None):
        self.init_datasets(stage)
        self.split_datasets(stage)
        self.limit_datasets(stage)
        self.transform_datasets(stage)

    def init_datasets(self, stage=None):
        """Use this to set initial values for self.{train/val_test}_dataset"""
        ds_cls = text_classification.DATASETS[self.dataset_name]
        self.train_ds, self.test_ds = ds_cls(root=self.root, ngrams=self.ngrams)
        self.vocab = self.train_ds.get_vocab()
        self.labels = self.train_ds.get_labels()
        self.val_ds = None

    def split_datasets(self, stage=None):
        """Split init datasets and reassign to self.{train/val/test}_dataset"""
        if not self.train_val_split or stage == 'test':
            return
        train_split_size = int(len(self.train_ds) * (1 - self.train_val_split))
        val_split_size = len(self.train_ds) - train_split_size
        self.train_ds, self.val_ds = random_split(
            self.train_ds, [train_split_size, val_split_size]
        )

    def limit_datasets(self, stage=None):

        if stage == 'fit':
            if self.n_train is not None:
                self.train_ds = Subset(self.train_ds, range(self.n_train))

            if self.n_val is not None and self.val_ds is not None:
                self.val_ds = Subset(self.val_ds, range(self.n_val))

        if stage == 'test' and self.n_test is not None:
            self.test_ds = Subset(self.test_ds, range(self.n_test))

    def transform_datasets(self, stage=None):
        pass

    def get_dataloader(self, ds):
        return (
            DataLoader(ds, self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
            if ds is not None
            else None
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_ds)

    def val_dataloader(self):
        return self.get_dataloader(self.val_ds)

    def test_dataloader(self):
        return self.get_dataloader(self.test_ds)

    @staticmethod
    def collate_fn(batch):
        r"""
        Since the text entries have different lengths, a custom function
        generate_batch() is used to generate data batches and offsets,
        which are compatible with EmbeddingBag. The function is passed
        to 'collate_fn' in torch.utils.data.DataLoader. The input to
        'collate_fn' is a list of tensors with the size of batch_size,
        and the 'collate_fn' function packs them into a mini-batch.
        Pay attention here and make sure that 'collate_fn' is declared
        as a top level def. This ensures that the function is available
        in each worker.

        Output:
            text: the text entries in the data_batch are packed into a list and
                concatenated as a single tensor for the input of nn.EmbeddingBag.
            offsets: the offsets is a tensor of delimiters to represent the beginning
                index of the individual sequence in the text tensor.
            cls: a tensor saving the labels of individual text entries.
        """
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label


if __name__ == "__main__":
    dm = TorchTextDataModule()
    dm.setup()

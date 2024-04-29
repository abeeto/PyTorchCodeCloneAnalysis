#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build dataset for model training and testing.

   Author: Meng Cao
"""
import pickle

from os.path import join
from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
from torch.nn.init import uniform_


class DatasetBuilder:
    """Class for dataset generation.
    """

    def __init__(self, args):
        self.inputs = None
        self.labels = None
        self.args = args

    def create_dataset(self):
        """Read dataset and convert to PyTorch Examples.
        """
        # define how text and label will be processed
        self.inputs = data.Field(lower=True, tokenize='spacy', batch_first=True)
        self.labels = data.Field(sequential=False, batch_first=True, unk_token=None)

        # create datasets
        train, dev, test = datasets.SNLI.splits(self.inputs, self.labels,
                                                root=self.args.data_folder)

        return train, dev, test

    def _build_vocab(self, train, dev, test):
        """Build vocabulary: mapping word string to ids.
        """
        # load pre-trained vectors
        vectors = Vectors(name=self.args.word_vectors,  # name of the vector file
                          cache=self.args.vector_cache,  # directory path
                          unk_init=uniform_)  # initialization for unknown words

        # build vocabulary
        self.inputs.build_vocab(train, dev, test, vectors=vectors)
        self.labels.build_vocab(train)

        # save vocabulary
        self.input_vocab = self.inputs.vocab
        self.label_vocab = self.labels.vocab

    def get_iterator(self, train, dev, test):
        """Get iterator for the given dataset.
        """
        # build vocabuary
        self._build_vocab(train, dev, test)

        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_sizes=(self.args.batch_size, 256, 256))

        return train_iter, dev_iter, test_iter


class BuildLPCDataset:
    """Read and pre-process LPC dataset.
    """

    def __init__(self, args):
        self.args = args
        self.inputs = data.Field(lower=True, tokenize='spacy', batch_first=True)
        self.labels = data.Field(sequential=False, batch_first=True, use_vocab=False)

    def _load_pickle_data(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _create_dataset(self, data_list, fields):
        """Get a single dataset.
        """
        examples = []
        for sample in tqdm(data_list):
            label = sample['label']
            premise = ' '.join(sample['related_sentences'][:2])
            hypothesis = sample['claim']
            examples.append(data.Example.fromlist([label, premise, hypothesis], fields))

        return data.Dataset(examples, fields)

    def create_dataset(self):
        """Read dataset and convert to PyTorch Examples.
        """
        # load pickle data
        train_list = self._load_pickle_data(join(self.args.data_folder, 'train.pickle'))
        dev_list = self._load_pickle_data(join(self.args.data_folder, 'dev.pickle'))

        # create datasets
        fields = [("label", self.labels), ("premise", self.inputs), ("hypothesis", self.inputs)]
        train = self._create_dataset(train_list, fields)
        dev = self._create_dataset(dev_list, fields)

        self._build_vocab(train, dev)

        return train, dev

    def _build_vocab(self, train, dev):
        """Build vocabulary: mapping word string to ids.
        """
        # load pre-trained vectors
        vectors = Vectors(name=self.args.word_vectors,  # name of the vector file
                          cache=self.args.vector_cache,  # directory path
                          unk_init=uniform_)  # initialization for unknown words

        # build vocabulary
        self.inputs.build_vocab(train, dev, vectors=vectors)
        self.labels.build_vocab(train)

        # save vocabulary
        self.input_vocab = self.inputs.vocab

    def get_iterator(self, train, dev):
        """Get iterator for the given dataset.
        """
        # train_iter, dev_iter = data.BucketIterator.splits(
        #     (train, dev), batch_sizes=(self.args.batch_size, 256))

        train_iter = data.BucketIterator(train, batch_size=self.args.batch_size)
        dev_iter = data.BucketIterator(dev, batch_size=256)

        return train_iter, dev_iter

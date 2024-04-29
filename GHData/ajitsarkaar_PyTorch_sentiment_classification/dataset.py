import os
from os.path import join as pjoin
import random
SEED = 407

import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets

class IMDBDataset(object):
	'''
	Wrapper for imdb dataset. Used for abstracting away helper functions.
	'''
	def __init__(self):
		super(IMDBDataset, self).__init__()

	def get_vocab_size(self):
		return len(self.TEXT.vocab)

	def get_pad_idx(self):
		return self.TEXT.vocab.stoi[self.TEXT.pad_token]
		
	def create_dataset(self, sequence_length, batch_size=1):
		self.TEXT = data.Field(tokenize='spacy', lower=True, fix_length=sequence_length)
		self.LABEL = data.LabelField(dtype=torch.float)
		MAX_VOCAB_SIZE = 10000
		EMBEDDING_DIM = 300
		BATCH_SIZE = batch_size
		
		train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

		self.TEXT.build_vocab(train_data, 
			max_size=MAX_VOCAB_SIZE, 
			vectors='glove.6B.300d', 
			unk_init=torch.Tensor.normal_)
		self.LABEL.build_vocab(train_data)

		self.train_iter = data.BucketIterator(train_data, BATCH_SIZE, train=True)
		self.test_iter = data.BucketIterator(test_data, BATCH_SIZE, train=False)

		return self.train_iter, self.test_iter
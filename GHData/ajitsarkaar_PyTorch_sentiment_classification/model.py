import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentNet(nn.Module):
	'''
	A sequential model for sentiment analysis. 
	'''
	def __init__(self, unit='LSTM', num_layers=1, vocab_size=None, word_size=None, sequence_length=None, pad_idxs=None, embedding_size=32, num_classes=2, hidden_layers=[16, 16]):
		super(SentimentNet, self).__init__()
		
		if vocab_size is None or word_size is None or pad_idxs is None or sequence_length is None:
			raise ValueError('All parameters not specified.')

		assert len(hidden_layers) > 0, 'At least one hidden layer has to be specified.'

		self.sequence_length = sequence_length
		self.unit = unit
		self.word_size = word_size
		self.embedding_size = embedding_size
		self.embedding = nn.Embedding(vocab_size, word_size, padding_idx=pad_idxs)
		
		if unit == 'GRU':
			self.sequence_model = nn.GRU(word_size, embedding_size, num_layers, bidirectional=False)
			self.ndirections = 1
		elif unit == 'BiLSTM':
			self.sequence_model = nn.LSTM(word_size, embedding_size, num_layers, bidirectional=True)
			self.ndirections = 2
		else:
			self.sequence_model = nn.LSTM(word_size, embedding_size, num_layers, bidirectional=False)
			self.ndirections = 1

		self.flatten_dim = self.ndirections * self.sequence_length * self.embedding_size
		self.dense = nn.Sequential(
			nn.Linear(self.flatten_dim, 32),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(32, 32),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(32, 32),
			nn.ReLU(inplace=True),
			nn.Linear(32, 2)
		)

	def initialize(self, batch_size=1, dev=torch.device('cpu')):
		def random_tensor(bs, hs):
			return torch.randn(self.ndirections, bs, hs, device=dev)
		
		if self.unit == 'GRU':
			hidden_state = random_tensor(batch_size, self.embedding_size)
			return hidden_state
		else:
			hidden_state = random_tensor(batch_size, self.embedding_size)
			cell_state = random_tensor(batch_size, self.embedding_size)
			return (hidden_state, cell_state)

	def forward(self, text):
		embedded = self.embedding(text)
		emb_vector, next_hidden_state = self.sequence_model(embedded)
		flattened_features = torch.flatten(emb_vector.permute(1, 0, 2), 1)
		logits = self.dense(flattened_features)

		return logits, next_hidden_state
# coding:utf-8
import logging
import numpy as np

import torch
from torch import nn

# pylint: disable=W0221
class Network(nn.Module):
	def __init__(self, emb, rnn_size=200, mode='GRU'):
		super(Network, self).__init__()
		'''
		mode: 'GRU', 'LSTM', 'Attention'
		'''
		self.mode = mode

		self.embLayer = EmbeddingLayer(emb)
		self.encoder = Encoder(embedding_size=emb.shape[1], rnn_size=rnn_size, mode=mode)
		if mode == 'Attention':
			self.selfAttention = SelfAtt(hidden_size=rnn_size)
		self.predictionNetwork = PredictionNetwork(rnn_size=rnn_size)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, sent, sent_length, label=None):

		embedding = self.embLayer.forward(sent)
		hidden_states = self.encoder.forward(embedding, sent_length)
		if self.mode == 'Attention':
			sentence_representation, penalization_loss = self.selfAttention.forward(hidden_states)
		else:
			sentence_representation = hidden_states.mean(dim=1)
		logit = self.predictionNetwork.forward(sentence_representation)

		if label is None:
			return logit

		classification_loss = self.loss(logit, label)
		if self.mode == 'Attention':
			return logit, classification_loss + penalization_loss * .0
		else:
			return logit, classification_loss

class EmbeddingLayer(nn.Module):	# Embedding data as input to the RNN
	def __init__(self, emb):
		super(EmbeddingLayer, self).__init__()

		vocab_size, embedding_size = emb.shape
		self.embLayer = nn.Embedding(vocab_size, embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(emb))

	def forward(self, sent):
		'''
		inp: data
		output: post
		'''
		return self.embLayer(sent)

class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self, input_size, hidden_size):
		super(LSTM, self).__init__()
		# Implement LSTM
		self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.c_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.hidden_size = hidden_size

	def forward(self, embedding, init_h=None, init_c=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		# Implement LSTM
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h

		if init_c is None:
			c = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			c = init_c

		hidden_states = []
		cell_states = []
		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			#𝑓_𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊_𝑓 𝑥_𝑡+𝑈_𝑓 ℎ_(𝑡−1)+𝑏_𝑓)
			#𝑖_𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊_𝑖 𝑥_𝑡+𝑈_𝑖 ℎ_(𝑡−1)+𝑏_𝑖)
			#𝑜_𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊_𝑜 𝑥_𝑡+𝑈_𝑜 ℎ_(𝑡−1)+𝑏_𝑜)
			f = torch.sigmoid(self.forget_gate(_input)) # [batch_size, hidden_size]
			i = torch.sigmoid(self.input_gate(_input)) # [batch_size, hidden_size]
			o = torch.sigmoid(self.output_gate(_input)) # [batch_size, hidden_size]
			
			
			# Update cell state
			#𝑐_𝑡=𝑓_𝑡∘𝑐_(𝑡−1)+𝑖_𝑡∘tanh⁡(𝑊_𝑐 𝑥_𝑡+𝑈_𝑐 ℎ_(𝑡−1)+𝑏_𝑐 )
			c = f*c + i*torch.tanh(self.c_gate(_input))

			# ℎ_𝑡=𝑜_𝑡∘tanh⁡(𝑐_𝑡)
			h = o*torch.tanh(c)
			hidden_states.append(h)

		# prediction history
		return torch.stack(hidden_states, dim=1) # [batch_size, sentence_length, hidden_size]

class GRU(nn.Module):
	"""docstring for GRU"""
	def __init__(self, input_size, hidden_size):
		super(GRU, self).__init__()
		self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.update_gate.bias.data.fill_(0.0)
		self.reset_gate.bias.data.fill_(0.0)
		self.memory_gate.bias.data.fill_(0.0)

		self.hidden_size = hidden_size
		
	def forward(self, embedding, init_h=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h

		hidden_states = []
		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			# 𝑧_𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊_𝑧 𝑥_𝑡+𝑈_𝑧 ℎ_(𝑡−1)+𝑏_𝑧) 
			# 𝑟_𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊_𝑟 𝑥_𝑡+𝑈_𝑟 ℎ_(𝑡−1)+𝑏_𝑟 )
			z = torch.sigmoid(self.update_gate(_input)) # [batch_size, hidden_size]
			r = torch.sigmoid(self.reset_gate(_input)) # [batch_size, hidden_size]
			
			# TODO: Update hidden state h
			# ℎ ̂_𝑡=tanh⁡(𝑊_ℎ 𝑥_𝑡+𝑈_ℎ (𝑟_𝑡∘ℎ_(𝑡−1) )+𝑏_ℎ)
			h_input = torch.cat([embedding[t], r*h], dim=1)
			h_hat = torch.tanh(self.memory_gate(h_input))

			# ℎ_𝑡=(1−𝑧_𝑡 )∘ℎ_(𝑡−1)+𝑧_𝑡∘ℎ ̂_𝑡
			h = (1-z)*h + z*(h_hat) 
			hidden_states.append(h)

		return torch.stack(hidden_states, dim=1) # [batch_size, sentence_length, hidden_size]

class Encoder(nn.Module):
	def __init__(self, embedding_size, rnn_size, mode='GRU'):
		super(Encoder, self).__init__()

		if mode == 'GRU':
			self.rnn = GRU(embedding_size, rnn_size)
		else:
			self.rnn = LSTM(embedding_size, rnn_size)

	def forward(self, embedding, sent_length=None):
		'''
		sent_length is not used
		'''
		hidden_states = self.rnn(embedding.transpose(0, 1)) # [batch_size, sentence_length, hidden_size]
		# you can add dropout here
		hidden_states = nn.functional.dropout(hidden_states, p = 0.2)

		return hidden_states

class SelfAtt(nn.Module):
	"""docstring for SelfAtt"""
	def __init__(self, hidden_size):
		super(SelfAtt, self).__init__()
		# TODO: Implement Self-Attention
		self.w_1 = nn.Linear(hidden_size, hidden_size, False)
		self.w_2 = nn.Linear(hidden_size, 1, False)

	
	def forward(self, H, add_penalization=True):
		'''
		H: [batch_size, sentence_length, hidden_size]
		'''
		# TODO: Implement Self-Attention
		batch_size, sentence_length, hidden_size = H.size()

		# batchsize, 1, sentence_length
		A = nn.functional.softmax(self.w_2(torch.tanh(self.w_1(H))), dim=1).transpose(1,2)	

		# batchsize, 1, sentence_length
		M = torch.bmm(A, H) 

		# M = [batchsize, sentence_length]
		M = torch.squeeze(M)	# Removes a tensor with all the dimensions of input of size 1 removed.
		
		if add_penalization:
			# batchsize, 1, 1
			I = torch.ones(batch_size, 1, 1).cuda()
			P = pow(torch.norm(torch.bmm(A, torch.transpose(A, 1, 2)) - I), 2)
		else:
			P = 0

		return M, P

class PredictionNetwork(nn.Module):
	def __init__(self, rnn_size, hidden_size=64, class_num=5):
		super(PredictionNetwork, self).__init__()
		self.predictionLayer = nn.Sequential(nn.Linear(rnn_size, hidden_size),
											nn.ReLU(),
											nn.Linear(hidden_size, class_num))

	def forward(self, h):

		return self.predictionLayer(h)

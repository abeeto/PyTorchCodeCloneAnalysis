import numpy as numpy

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

with open('data/test.txt', 'r') as f:
	rest = f.read()

############################################## tokenization ############################################
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])  # encode the text

############################################## pre_prpcessing the data #################################
'''one-hot'''

##################################### making training mini-batchs #######################################
'''
starting sequence:
[1 2 3 4 5 6 7 8 9 10 11 12]
batch size = 2
[1 2 3 4 5 6]
[7 8 9 10 11 12]
sequence length = 3
1 2 3   4 5 6
7 8 9   101112
'''
def get_batches(arr, batch_size, seq_length):

	batch_size_total = batch_size * seq_length
	n_batches = len(arr)//batch_size_total
	arr = arr[:n_batches * batch_size_total]
	arr = arr.reshape((batch_size, -1))              # reshape
	for n in range(0, arr.shape[1], seq_length):
		x = arr[:, n:n+seq_length]                   # features
		y = np.zeros_like(x)                         # target, shift by one
		try:
			y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
		except IndexError:
			y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
		yield x, y


batches = get_batches(encoded, 8, 50)
x, y = next(batches)

##################################### define the model ####################################################
# check gpu

class CharRNN(nn.Module):
	def __init__(self, tokens, n_hidden = 256, n_layers = 2, drop_prob = 0.5, lr = 0.001):
		super().__init__()
		self.drop_prob = drop_prob
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.lr = lr

		self.chars = tokens
		self.int2char = dict(enumerate(self.chars))
		self.char2int = {ch: ii for ii, ch in self.int2char.items()}

		# define the layers
		self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout = drop_prob, batch_first = True)
		self.dropout = nn.Dropout(drop_prob)
		self.fc = nn.Linear(n_hidden, len(self.chars))

		def forward(self, x, hidden):
			r_output, hidden = self.lstm(x, hidden)
			out = self.dropout(r_output)
			out = self.fc(out)
			return out, hidden

		def init_hidden(self, batch_size):
			'''
			create two new tensors with sizes (n_layers * batch_size * n_hidden)
			initialized to zero, for hidden state and cell state of lstm
			'''
			weight = next(self.parameters()).data

			if (train_on_gpu):
				hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
					weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
			else:
				hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
					weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

			return hidden

		def train(net, data, epochs=10, batch_size=10, seq_length=10, lr=0.001, clip=5, val_frac, print_every):

			opt = torch.optim.Adam(net.parameters(), lr = lr)
			criterion = nn.CrossEntropyLoss()

			val_idx = int(len(data) * (1-val_frac))
			data, val_data = data[:val_idx], data[val_idx:]

			if (train_on_gpu):
				net.cuda()

			counter = 0
			n_chars = len(net.chars)
			for e in range(epochs):
				h = net.init_hidden(batch_size)

				for x, y in get_batches(data, batch_size, seq_length):
					counter += 1

					x = one_hot_encode(x, n_chars)
					inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

					if (train_on_gpu):
						inputs, targets = inputs.cuda(), targets.cuda()

					h = tuple([each.data for each in h])

					net.zero_grad()
					output, h = net(inputs, h)

					loss = criterion(output, targets.view(batch_size*seq_length))
					loss.backward()
					nn.utils.clip_grad_norm_(net.parameters(), clip).   # prevent the exploding
					opt.step()

					if counter % print_every == 0:
						pass

##################################### instantiating the model #########################################


##################################### checkpoints #########################################


##################################### make predictions #########################################
'''top k'''




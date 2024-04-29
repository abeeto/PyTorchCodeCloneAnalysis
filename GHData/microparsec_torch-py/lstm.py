import torch
import random
import copy
import numpy as np


class LSTM:

	# N = input length
	# D = hidden layer width
	# Z = Total state length N + D

	# z block i input gate o output gate f forget gate (vectors)
	# W_ input weights
	# R_ recurrent weights


	# p_ peephole (vector)
	# b_ bias (vector)


	def __init__(self, N, D, b = None):

		self.activation_g = torch.tanh
		self.activation_i = torch.sigmoid
		self.activation_dg = dtanh
		self.activation_di = dsigmoid

		Z = N + D
		self.N = N
		self.D = D
		self.Z = Z

		self.Wz = torch.div(torch.cuda.FloatTensor(D,Z).normal_(0,1), Z ** (.5) / 2.)
		self.Wi = torch.div(torch.cuda.FloatTensor(D,Z).normal_(0,1), Z ** (.5) / 2.)
		self.Wf = torch.div(torch.cuda.FloatTensor(D,Z).normal_(0,1), Z ** (.5) / 2.)
		self.Wo = torch.div(torch.cuda.FloatTensor(D,Z).normal_(0,1), Z ** (.5) / 2.)
		self.Wy = torch.div(torch.cuda.FloatTensor(N,D).normal_(0,1), D ** (.5) / 2.)

		self.bz = torch.cuda.FloatTensor(D).zero_()
		self.bi = torch.cuda.FloatTensor(D).zero_()
		self.bf = torch.cuda.FloatTensor(D).zero_()
		self.bo = torch.cuda.FloatTensor(D).zero_()
		self.by = torch.cuda.FloatTensor(N).zero_()



	def forward(self, x, h, c):
		
		
		X = torch.cat( (h, x) )

		z = self.activation_g(torch.mv(self.Wz, X) + self.bz)
		i = self.activation_i(torch.mv(self.Wi, X) + self.bi)
		f = self.activation_i(torch.mv(self.Wf, X) + self.bf)
		o = self.activation_i(torch.mv(self.Wo, X) + self.bo)
		c_old = c.clone()
		c = z * i + c_old * f
		h = self.activation_g(c) * o

		y = torch.mv(self.Wy, h) + self.by

		probs = softmax(y)

		cache = (X, z, i, f, o, c_old)
		return (probs, y, h, c, cache)

	def backward(self, probs, y_, y, h, c, cache, d_next):
		X, z, i, f, o, c_old = cache
		dh, dz, di, df, do, dc = d_next

		#get the error
		dy = probs.clone()
		dy -= y
		
		#propage to hidden layer
		dWy = torch.ger(dy, h)
		dby = dy
		dh = torch.t(self.Wy) @ dy + dh

		#propagate error to gates
		do = self.activation_di(o) * self.activation_g(c) * dh
		dc = do * dh * self.activation_dg(c) + dc.clone()

		df = self.activation_di(f) * (c_old * dc)

		di = z * dc * self.activation_di(i)

		dz = i * dc * self.activation_dg(z)

		#propagate to input weights
		dWz = torch.ger(dz, X)
		dbz = dz
		dxz = torch.t(self.Wz) @ dz

		dWi = torch.ger(di, X)
		dbi = di
		dxi = torch.t(self.Wi) @ di

		dWf = torch.ger(df, X)
		dbf = df
		dxf = torch.t(self.Wf) @ df

		dWo = torch.ger(do, X)
		dbo = do
		dxo = torch.t(self.Wo) @ do

		#propagate to hidden + 1 (for the next timestep)
		dh = dxz + dxi + dxf + dxo
		dh = dh[:self.D].clone()

		dc = f * dc.clone()

		d = (dh, dz, di, df, do, dc)

		grad = (dWz, dWi, dWf, dWo, dWy, dbz, dbi, dbf, dbo, dby)

		return grad, d



def training(lstm, data, iterations, alpha = 1e-3):



	for i in range(0, iterations):

		r = random.randint(0, len(data) - 51)
		batch = data[r:r+50]
		grads, loss = train(lstm, batch)

		alpha = alpha * (1 - 0.0001)

		lstm.Wz -= alpha * grads[0]
		lstm.Wi -= alpha * grads[1]
		lstm.Wf -= alpha * grads[2]
		lstm.Wo -= alpha * grads[3]
		lstm.Wy -= alpha * grads[4]
		lstm.bz -= alpha * grads[5]
		lstm.bi -= alpha * grads[6]
		lstm.bf -= alpha * grads[7]
		lstm.bo -= alpha * grads[8]
		lstm.by -= alpha * grads[9]

		if i % 100 == 0:
			samp = sample(lstm, 50, "a")
			print("Iteration %i - Loss: %f - alpha: %f" % (i, loss, alpha))
			print("sample:        ", samp)
			#print (grads[:5])



def train(lstm, batch):
	loss = 0.
	results = []
	h = torch.cuda.FloatTensor(lstm.D).normal_()

	c = torch.cuda.FloatTensor(lstm.D).zero_()
	for ix in range(0, len(batch) - 1):
		char = batch[ix]

		x = torch.cuda.FloatTensor(lstm.N).zero_()
		x[getindex(char)] = 1.0
		y = torch.cuda.FloatTensor(lstm.N).zero_()
		y[getindex(batch[ix+1])] = 1.0


		probs, y_, h, c, cache = lstm.forward(x, h, c)

		results.append((probs, y_, h, c, cache))
		loss += cross_entropy(y, probs)

	loss /= len(batch)
	grads = [torch.cuda.FloatTensor(lstm.D, lstm.Z).zero_() for x in range(0,4)] + \
			[torch.cuda.FloatTensor(lstm.N, lstm.D).zero_() for x in range(0,1)] + \
			[torch.cuda.FloatTensor(lstm.D).zero_() for x in range(0,4)] + \
			[torch.cuda.FloatTensor(lstm.N).zero_() for x in range(0,1)]
	d_next = tuple(torch.cuda.FloatTensor(lstm.D).normal_() for x in range(0,6))

	for il in reversed(range(0, len(batch) - 1)):
		char = batch[il]
		y = torch.cuda.FloatTensor(lstm.N).zero_()
		y[getindex(batch[il+1])] = 1.0
		probs, y_, h, c, cache = results[il]
		grad, d_next = lstm.backward(probs, y_, y, h, c, cache, d_next)

		for i in range(0, len(grads)):
			grads[i] += grad[i]



	return tuple(grads), loss

def sample(lstm, length = 50, start = None):
	if not start:
		start = random.choice(list(dic.keys()))

	x = torch.cuda.FloatTensor(lstm.N).zero_()
	x[char2idx[start]] = 1.0
	h = torch.cuda.FloatTensor(lstm.D).normal_()
	c = torch.cuda.FloatTensor(lstm.D).normal_()
	string = ""
	for i in range(0,length):
		probs, y_, h, c, cache = lstm.forward(x, h, c)
		probs = probs.cpu().numpy()
		probs /= np.sum(probs)

		index = np.random.choice(range(0,vocab_size), p=probs)
		sample = idx2char[index]
		string += sample[0]
	return string

def cross_entropy(y, y_):
	epsilon = 1e-15
	y_ = torch.clamp(y_, epsilon, 1-epsilon)
	return -1. * torch.sum(y * torch.log(y_))

dic = { "a" : 0, "b" : 1, "c" : 2, "d" : 3, "e": 4, "f" : 5, "g" : 6, "h" : 7, "i" : 8, "j" : 9, "k" : 10, "l" : 11, "m" : 12, "n" : 13, "o" :14, "p": 15, "q" : 16, "r" : 17, "s" : 18, "t" : 19, "u" : 20, "v" : 21, "w" : 22, "x" : 23, "y": 24, "z" : 25, " " : 26, "\n" : 27}

def getindex(cha):
	return dic[cha]

def dtanh(tensor):
	return torch.cuda.FloatTensor(len(tensor)).fill_(1.) - (torch.tanh(tensor) * torch.tanh(tensor))

def dsigmoid(tensor):
	return torch.sigmoid(tensor) * (torch.cuda.FloatTensor(len(tensor)).fill_(1.) - torch.sigmoid(tensor))

def softmax(tensor):
	eX = torch.exp(tensor - torch.max(tensor))
	return eX / eX.sum()


print("Loading training data")
data = open("shakey.txt").read()
char2idx = {char : i for i, char in enumerate(set(data)) }
idx2char = {i : char for i, char in enumerate(set(data)) }
vocab_size = len(char2idx)


print("Init LSTM")
x = LSTM(vocab_size, 64)
print("Training")

training(x, data, 25000)
print("Sample incoming")
print(sample(x))
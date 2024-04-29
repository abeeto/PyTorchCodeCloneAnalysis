# !usr/bin/python
# coding=utf-8

from __future__ import print_function
import torch
import json
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 超参数
# EPOCH = 50000
# BATCH_SIZE = 64
# LR = 0.01
# DOWNLOAD_MNIST = False
# N_TEST_IMG = 5

# 下载MNIST数据
# train_data = torchvision.datasets.MNIST(
# 	root='./mnist/',
# 	train=True,
# 	transform=torchvision.transforms.ToTensor(),
# 	download=DOWNLOAD_MNIST,
# )

# 输出一个样本
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()


def load_data():
	return load_data_fft()
	csv_data = pd.read_csv('data.csv', index_col = 0)
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# csv_data = csv_data.drop(['chroma_stft'], axis = 1)
	# csv_data = csv_data.drop(['spec_cent'], axis = 1)
	# csv_data = csv_data.drop(['spec_bw'], axis = 1)
	# csv_data = csv_data.drop(['rolloff'], axis = 1)
	# csv_data = csv_data.drop(['zcr'], axis = 1)
	# for i in range(20):
	# 	csv_data = csv_data.drop(['mfcc_%s'%i], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return total_X, filenameList, 0.018

def load_data_android():
	content = json.load(open('data.json', 'r'))
	filenameList = list(content.keys())
	filenameList.sort()
	total = []
	for i in filenameList:
		total.append(content[i])
	total = np.array(total)
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(total)
	return total_X, filenameList, 0.0285

def load_data_fft():
	csv_data = pd.read_csv('data_peak.csv', index_col = 0)
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return total_X, filenameList, 0.0006

class MyDataset(Data.Dataset):#这是一个Dataset子类
	def __init__(self, data_x, data_y):
		self.data_x = data_x
		self.data_y = data_y
 
	def __getitem__(self, index):
		data = self.data_x[index]
		label = self.data_y[index]
		return data, label #返回标签
 
	def __len__(self):
		return len(self.data_x)

total_X, filenameList, threshold = load_data()
# print(total_X.shape)
# x_train, x_test, y_train, y_test = train_test_split(total_X, total_X, test_size = 0.2)
data_x = torch.from_numpy(total_X)
data_y = torch.from_numpy(total_X)

learning_rate = 0.005
training_epochs = 10000
batch_size = 512
display_step = 100
# examples_to_show = 10
n_input = total_X.shape[1]

# Dataloader
dataset = Data.TensorDataset(data_x, data_y)
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# print(train_loader)

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Linear(n_input, 16),
			nn.Tanh(),
			nn.Linear(16, 8),
			nn.Tanh(),
			nn.Linear(8, 2),
		)

		self.decoder = nn.Sequential(
			nn.Linear(2, 8),
			nn.Tanh(),
			nn.Linear(8, 16),
			nn.Tanh(),
			nn.Linear(16, n_input),
			nn.Sigmoid(),
		)

	# def forward(self, x):
	# 	encoded = self.encoder(x)
	# 	decoded = self.decoder(encoded)
	# 	return encoded, decoded

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
# autoencoder.load_state_dict(torch.load('autoencoder.pkl'))

for epoch in range(training_epochs):
	for step, (x, y) in enumerate(train_loader):
		b_x = Variable(x)
		b_y = Variable(y)
		b_x = b_x.float()
		b_y = b_y.float()
		# b_label = Variable(y)

		encoded, decoded = autoencoder(b_x)

		loss = loss_func(decoded, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if epoch % display_step == 0:
		print('Epoch: ', epoch, '| train loss: %s' %loss.data.numpy())
	if loss.data.numpy() < 0.001:
		break
print('Epoch: ', epoch, '| train loss: %s' %loss.data.numpy())
torch.save(autoencoder.state_dict(), 'autoencoder.pkl')


print(data_x.shape)
encoded, decoded = autoencoder(Variable(data_x).float())
encoded = encoded.data.numpy()
print(encoded.shape)

idx = 822
print(filenameList[idx])
dev = encoded[idx]
result = []
count = 0
# for i in csv_data.iterrows():
for i in encoded:
	devb = i
	dist2 = np.sqrt(np.sum(np.square(dev - devb)))
	result.append((filenameList[count], dist2))
	count += 1
result = sorted(result, key = lambda x: x[1])
for k, v in enumerate(result):
	print(k, *v)

print('*'*100)

idx = 292
print(filenameList[idx])
dev = encoded[idx]
result = []
count = 0
# for i in csv_data.iterrows():
for i in encoded:
	devb = i
	dist2 = np.sqrt(np.sum(np.square(dev - devb)))
	result.append((filenameList[count], dist2))
	count += 1
result = sorted(result, key = lambda x: x[1])
for k, v in enumerate(result):
	print(k, *v)
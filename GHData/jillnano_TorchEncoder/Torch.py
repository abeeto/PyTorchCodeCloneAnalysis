# !usr/bin/python
# coding=utf-8

# 使用原始特征计算

import numpy as np
import librosa
import json
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from pandas.io.json import json_normalize

from glob import glob

sample_length = 30

def getSample(path):
	x, sr = librosa.load(path)
	sample_size = sr * sample_length
	start = (x.shape[0] - sample_size) / 2
	sample = x[start:start + sample_size]
	return sample, sr

def sample2data(path, y, sr):
	data = {'filename': path}
	chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
	spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
	spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
	rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	mfcc = librosa.feature.mfcc(y = y, sr = sr)
	data['chroma_stft'] = np.mean(chroma_stft)
	data['spec_cent'] = np.mean(spec_cent)
	data['spec_bw'] = np.mean(spec_bw)
	data['rolloff'] = np.mean(rolloff)
	data['zcr'] = np.mean(zcr)
	for k, v in enumerate(mfcc):
		data['mfcc_%s'%k] = np.mean(v)
	return data

def writeData():
	header = ['filename', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr']
	for i in xrange(20):
		header.append('mfcc_%s'%i)

	data = []
	fileList = glob('music/*.mp3')
	fileList.sort()
	for path in fileList:
		print(path)
		y, sr = getSample(path)
		d = sample2data(path, y, sr)
		data.append(d)
		# break
	data = json_normalize(data)
	data.to_csv('data.csv', columns = header)

def load_data():
	csv_data = pd.read_csv('data.csv')
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return X, filenameList

def load_data_android():
	content = json.load(open('data.json', 'r'))
	filenameList = list(content.keys())
	filenameList.sort()
	total = []
	for i in filenameList:
		v = np.array(content[i])
		v = v.T
		value = []
		for x in v:
			value.append(np.mean(x))
		value = np.array(value)
		total.append(value)
	total = np.array(total)
	print(total.shape)
	print(total)
	print(total[0])
	scaler = MinMaxScaler()
	total_X = scaler.fit_transform(total)
	return total_X, filenameList

def load_data_fft():
	csv_data = pd.read_csv('data_peak.csv')
	filenameList = csv_data.filename
	csv_data = csv_data.drop(['filename'], axis = 1)
	# scaler = StandardScaler()
	scaler = MinMaxScaler()
	X = scaler.fit_transform(np.array(csv_data, dtype = float))
	return X, filenameList

def readData(idx):
	# X, filenameList = load_data()
	# X, filenameList = load_data_android()
	X, filenameList = load_data_fft()
	print(X.shape)
	# target = csv_data.iloc[136]
	# dev = target.as_matrix()
	print(filenameList[idx])
	dev = X[idx]
	result = []
	count = 0
	# for i in csv_data.iterrows():
	for i in X:
		devb = i
		dist2 = np.sqrt(np.sum(np.square(dev - devb)))
		result.append((filenameList[count], dist2))
		count += 1
	result = sorted(result, key = lambda x: x[1])
	for k, v in enumerate(result):
		print(k, *v)

if __name__ == '__main__':
	readData(292)
	# scaler = StandardScaler()
	# X = scaler.fit_transform(np.array(data, dtype = float))
	# print X